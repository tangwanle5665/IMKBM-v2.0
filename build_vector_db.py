#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建向量数据库（多策略分块版）
- 教材/PPT：按字数分块（chunk_size=500）
- 药物说明书：按章节分块（drug_label_parser）
- 统一编码为 TF-IDF + BGE 双向量
"""

import os
import pickle
import gc
import numpy as np
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    TFIDF_MAX_FEATURES, BGE_BATCH_SIZE, BGE_DEVICE
)
from utils import logger, get_bge_model, get_tfidf_model
from drug_label_parser import parse_drug_label_dir, split_drug_label

os.makedirs(VECTOR_DB_DIR, exist_ok=True)


def split_text_documents(docs: list, chunk_size: int = CHUNK_SIZE, 
                         chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """普通教材文本：按字数分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


def split_drug_label_documents(docs: list) -> list:
    """药物说明书：按章节分块"""
    result = []
    for doc in docs:
        # 如果已经是 drug_label 类型，直接按章节分块
        if doc.metadata.get("source_type") == "drug_label":
            chunks = split_drug_label(doc.page_content, 
                                       doc.metadata.get("source", "unknown"))
            result.extend(chunks)
        else:
            # 普通文本，回退到字数分块
            result.append(doc)
    return result


def build_vector_db(source_dir: str = None, drug_label_dir: str = None):
    """构建向量数据库，支持多来源混合"""
    all_docs = []

    # 1. 加载普通教材
    if source_dir and os.path.exists(source_dir):
        logger.info(f"📚 加载教材目录: {source_dir}")
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                if len(text.strip()) < 50:
                    continue
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": fpath, "source_type": "textbook"}
                ))

    # 2. 加载药物说明书
    if drug_label_dir and os.path.exists(drug_label_dir):
        logger.info(f"💊 加载说明书目录: {drug_label_dir}")
        drug_docs = parse_drug_label_dir(drug_label_dir)
        all_docs.extend(drug_docs)
        logger.info(f"💊 说明书共 {len(drug_docs)} 个块")

    if not all_docs:
        logger.warning("⚠️ 未找到任何文档")
        return

    logger.info(f"📄 总文档数: {len(all_docs)}")

    # 3. 分块策略路由
    textbook_docs = [d for d in all_docs if d.metadata.get("source_type") != "drug_label"]
    drug_docs = [d for d in all_docs if d.metadata.get("source_type") == "drug_label"]

    chunks = []
    if textbook_docs:
        logger.info(f"🔪 教材按字数分块: {len(textbook_docs)} 篇")
        chunks.extend(split_text_documents(textbook_docs))
    if drug_docs:
        logger.info(f"🔪 说明书按章节分块: {len(drug_docs)} 篇")
        chunks.extend(split_drug_label_documents(drug_docs))

    logger.info(f"📦 总分块数: {len(chunks)}")

    # 4. TF-IDF 编码（保持稀疏矩阵，避免内存爆炸）
    logger.info("🧮 训练 TF-IDF 模型...")
    tfidf_model = get_tfidf_model()
    texts = [c.page_content for c in chunks]
    tfidf_model.fit(texts)
    tfidf_vectors = tfidf_model.embed_documents(texts)

    # 5. BGE 编码（分批 + 断点续传）
    logger.info("🧠 开始 BGE 语义编码...")
    bge = get_bge_model()
    dim = bge.embed_query("test").shape[0]

    bge_vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    checkpoint_dir = os.path.join(VECTOR_DB_DIR, "bge_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i in tqdm(range(0, len(chunks), BGE_BATCH_SIZE), desc="BGE编码"):
        cp_path = os.path.join(checkpoint_dir, f"batch_{i}.npy")
        if os.path.exists(cp_path):
            bge_vectors[i:i+BGE_BATCH_SIZE] = np.load(cp_path)
            continue

        batch_texts = texts[i:i+BGE_BATCH_SIZE]
        batch_vecs = bge.embed_documents(batch_texts)
        bge_vectors[i:i+BGE_BATCH_SIZE] = batch_vecs
        np.save(cp_path, batch_vecs)

    # 6. 保存索引
    logger.info("💾 保存向量索引...")
    index_data = {
        "chunks": chunks,
        "tfidf_vectors": tfidf_vectors,
        "bge_vectors": bge_vectors,
        "tfidf_model": tfidf_model,
    }

    with open(os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl"), "wb") as f:
        pickle.dump(index_data, f)

    # 7. 构建 FAISS 索引
    try:
        import faiss
        logger.info("🔍 构建 FAISS 索引...")
        bge_vectors_norm = bge_vectors / (np.linalg.norm(bge_vectors, axis=1, keepdims=True) + 1e-10)

        # 构建 Flat 索引
        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(bge_vectors_norm)
        faiss.write_index(flat_index, os.path.join(VECTOR_DB_DIR, "bge_index.faiss"))

        # 构建 IVFPQ 量化索引（加速检索）
        nlist = min(100, len(chunks) // 10)
        if nlist >= 10:
            quantizer = faiss.IndexFlatIP(dim)
            ivfpq_index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
            ivfpq_index.train(bge_vectors_norm)
            ivfpq_index.add(bge_vectors_norm)
            faiss.write_index(ivfpq_index, os.path.join(VECTOR_DB_DIR, "bge_index_ivfpq.faiss"))
            logger.info("✅ IVFPQ 量化索引已保存")
    except Exception as e:
        logger.warning(f"FAISS 索引构建失败: {e}")

    # 清理 checkpoint
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    gc.collect()
    logger.info("✅ 向量数据库构建完成")


if __name__ == "__main__":
    from config import PARSED_DIR, DRUG_LABEL_DIR
    build_vector_db(source_dir=PARSED_DIR, drug_label_dir=DRUG_LABEL_DIR)
