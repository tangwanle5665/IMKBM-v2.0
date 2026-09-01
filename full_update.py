#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键增量更新知识库（支持教材 + 药物说明书混合更新）
- 检测新增/修改的教材文件
- 检测新增/修改的说明书文件
- 增量更新向量库（不重建）
- 支持断点续传和重试
"""

import os
import json
import time
import pickle
import hashlib
import traceback
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    BASE_DIR, PARSED_DIR, DRUG_LABEL_DIR, VECTOR_DB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP
)
from utils import logger, get_bge_model, get_tfidf_model
from drug_label_parser import parse_drug_label_file

# 增量更新记录文件
UPDATE_LOG = os.path.join(VECTOR_DB_DIR, "update_log.json")


def _get_file_hash(filepath: str) -> str:
    """计算文件 MD5"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def _load_update_log() -> dict:
    """加载更新记录"""
    if os.path.exists(UPDATE_LOG):
        with open(UPDATE_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_update_log(log: dict):
    """保存更新记录"""
    with open(UPDATE_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def _detect_new_files(source_dir: str, log: dict, suffix: tuple = (".txt",)) -> List[str]:
    """检测新增或修改的文件"""
    new_files = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.lower().endswith(suffix):
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, BASE_DIR)
            current_hash = _get_file_hash(fpath)

            if rel_path not in log or log[rel_path] != current_hash:
                new_files.append(fpath)
                log[rel_path] = current_hash
    return new_files


def _load_existing_index() -> dict:
    """加载现有向量索引"""
    index_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl")
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_index(index_data: dict):
    """保存向量索引"""
    with open(os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl"), "wb") as f:
        pickle.dump(index_data, f)


def _build_faiss_index(bge_vectors):
    """重建 FAISS 索引"""
    try:
        import faiss
        import numpy as np
        dim = bge_vectors.shape[1]
        bge_vectors_norm = bge_vectors / (np.linalg.norm(bge_vectors, axis=1, keepdims=True) + 1e-10)

        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(bge_vectors_norm)
        faiss.write_index(flat_index, os.path.join(VECTOR_DB_DIR, "bge_index.faiss"))

        # 删除旧的 IVFPQ（需要重新训练）
        ivfpq_path = os.path.join(VECTOR_DB_DIR, "bge_index_ivfpq.faiss")
        if os.path.exists(ivfpq_path):
            os.remove(ivfpq_path)
            logger.info("🗑️ 旧 IVFPQ 索引已删除，下次启动时会自动重建")

        logger.info("✅ FAISS Flat 索引已重建")
    except Exception as e:
        logger.warning(f"FAISS 索引重建失败: {e}")


def _chunk_textbook_docs(docs: List[Document]) -> List[Document]:
    """教材按字数分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, add_start_index=True,
    )
    return splitter.split_documents(docs)


def _chunk_drug_label_docs(docs: List[Document]) -> List[Document]:
    """说明书按章节分块"""
    result = []
    for doc in docs:
        chunks = parse_drug_label_file(doc.metadata.get("source", ""))
        result.extend(chunks)
    return result


def full_update() -> Tuple[bool, str]:
    """
    一键增量更新知识库
    返回: (success, message)
    """
    logger.info("🔄 开始一键增量更新...")

    # 1. 加载更新记录和现有索引
    log = _load_update_log()
    existing = _load_existing_index()

    if existing is None:
        logger.info("📦 未找到现有索引，将执行全量构建")
        from build_vector_db import build_vector_db
        build_vector_db(source_dir=PARSED_DIR, drug_label_dir=DRUG_LABEL_DIR)
        return True, "全量构建完成"

    # 2. 检测新增文件
    new_textbook_files = _detect_new_files(PARSED_DIR, log, (".txt",)) if os.path.exists(PARSED_DIR) else []
    new_drug_files = _detect_new_files(DRUG_LABEL_DIR, log, (".txt", ".md")) if os.path.exists(DRUG_LABEL_DIR) else []

    if not new_textbook_files and not new_drug_files:
        return True, "没有新增或修改的文件，无需更新"

    logger.info(f"📚 新增教材: {len(new_textbook_files)} 个")
    logger.info(f"💊 新增说明书: {len(new_drug_files)} 个")

    # 3. 解析新增文件
    new_docs = []

    # 教材
    for fpath in new_textbook_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            if len(text.strip()) < 50:
                continue
            new_docs.append(Document(
                page_content=text,
                metadata={"source": fpath, "source_type": "textbook"}
            ))
        except Exception as e:
            logger.warning(f"读取失败 {fpath}: {e}")

    # 说明书
    for fpath in new_drug_files:
        try:
            chunks = parse_drug_label_file(fpath)
            new_docs.extend(chunks)
        except Exception as e:
            logger.warning(f"说明书解析失败 {fpath}: {e}")

    if not new_docs:
        return True, "新增文件解析后无有效内容"

    # 4. 分块
    textbook_docs = [d for d in new_docs if d.metadata.get("source_type") != "drug_label"]
    drug_docs = [d for d in new_docs if d.metadata.get("source_type") == "drug_label"]

    new_chunks = []
    if textbook_docs:
        logger.info(f"🔪 教材分块: {len(textbook_docs)} 篇")
        new_chunks.extend(_chunk_textbook_docs(textbook_docs))
    if drug_docs:
        logger.info(f"🔪 说明书分块: {len(drug_docs)} 篇")
        new_chunks.extend(drug_docs)  # 说明书已在 parser 中分块

    logger.info(f"📦 新增分块: {len(new_chunks)}")

    # 5. 编码新增块
    logger.info("🧮 编码新增内容...")
    tfidf_model = existing["tfidf_model"]
    bge = get_bge_model()

    texts = [c.page_content for c in new_chunks]
    new_tfidf = tfidf_model.transform(texts).toarray()
    new_bge = bge.embed_documents(texts)

    # 6. 合并索引
    logger.info("🔄 合并索引...")
    existing["chunks"].extend(new_chunks)
    existing["tfidf_vectors"] = _safe_vstack(existing["tfidf_vectors"], new_tfidf)
    existing["bge_vectors"] = _safe_vstack(existing["bge_vectors"], new_bge)

    # 7. 保存
    _save_index(existing)
    _save_update_log(log)

    # 8. 重建 FAISS
    _build_faiss_index(existing["bge_vectors"])

    # 清理缓存
    from utils import clear_vector_db_cache
    clear_vector_db_cache()

    msg = f"增量更新完成！新增 {len(new_chunks)} 个分块（教材 {len(textbook_docs)} 篇，说明书 {len(drug_docs)} 篇）"
    logger.info(f"✅ {msg}")
    return True, msg


def _safe_vstack(a, b):
    """安全合并向量，处理维度不匹配"""
    import numpy as np
    if a is None or a.size == 0:
        return b
    if b is None or b.size == 0:
        return a
    # 确保维度一致
    if a.shape[1] != b.shape[1]:
        logger.error(f"向量维度不匹配: {a.shape} vs {b.shape}")
        raise ValueError("向量维度不匹配，可能需要重建索引")
    return np.vstack([a, b])


if __name__ == "__main__":
    success, msg = full_update()
    print(msg)
