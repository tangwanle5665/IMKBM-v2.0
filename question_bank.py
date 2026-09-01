#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题库加载与相似题检索（支持多 CSV 合并：教材题库 + 考研真题）
- 自动合并 题库.docx.csv 与 题库.pdf.csv
- BGE 向量缓存到 question_bank/ 目录，重启不重复编码
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import QUESTION_BANK_PATH, QUESTION_TOP_K
from utils import logger, get_bge_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 题库 CSV 文件列表（按顺序合并；文件不存在则跳过）
QUESTION_BANK_PATHS = [
    QUESTION_BANK_PATH,                                  # 题库.docx.csv（教材题库）
    os.path.join(BASE_DIR, "题库.pdf.csv"),              # 题库.pdf.csv（考研真题）
]
# 向量缓存（避免每次启动重复 BGE 编码数千道真题）
_Q_VEC_CACHE = os.path.join(BASE_DIR, "question_bank", "q_vectors.npy")
_Q_FP_CACHE = os.path.join(BASE_DIR, "question_bank", "q_texts.fp")

_q_data, _q_texts, _q_vecs = [], [], []


def _load_all_rows():
    """合并所有存在的题库 CSV，返回 record 列表"""
    rows = []
    for path in QUESTION_BANK_PATHS:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            required = ["question", "options", "answer", "analysis", "know_point"]
            if not all(col in df.columns for col in required):
                logger.warning(f"题库文件列名不匹配，跳过：{path}")
                continue
            rows.extend(df.to_dict("records"))
        except Exception as e:
            logger.warning(f"题库文件加载失败，跳过：{path} -> {e}")
    return rows


def _texts_fingerprint(texts):
    """题库文本指纹，用于判断缓存是否过期"""
    joined = "\x01".join(texts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _load_vec_cache(texts, expected_dim=None):
    """尝试从磁盘加载向量缓存；指纹或维度不一致则返回 None"""
    try:
        if not os.path.exists(_Q_VEC_CACHE) or not os.path.exists(_Q_FP_CACHE):
            return None
        with open(_Q_FP_CACHE, "r", encoding="utf-8") as f:
            fp = f.read().strip()
        if fp != _texts_fingerprint(texts):
            logger.info("题库内容有变化，重新编码向量...")
            return None
        vecs = np.load(_Q_VEC_CACHE)
        if vecs.shape[0] != len(texts):
            return None
        if expected_dim and vecs.shape[1] != expected_dim:
            logger.info(f"题库向量维度变化（{vecs.shape[1]} → {expected_dim}），重新编码...")
            return None
        logger.info(f"✅ 题库向量缓存命中: {vecs.shape[0]} 题")
        return vecs
    except Exception as e:
        logger.debug(f"题库缓存加载失败: {e}")
        return None


def _save_vec_cache(texts, vecs):
    try:
        os.makedirs(os.path.dirname(_Q_VEC_CACHE), exist_ok=True)
        np.save(_Q_VEC_CACHE, vecs)
        with open(_Q_FP_CACHE, "w", encoding="utf-8") as f:
            f.write(_texts_fingerprint(texts))
    except Exception as e:
        logger.debug(f"题库缓存保存失败: {e}")


def init_bank():
    global _q_data, _q_texts, _q_vecs
    if _q_data:
        return
    _q_data = _load_all_rows()
    if not _q_data:
        logger.warning("题库为空（未找到 CSV 或文件为空），跳过加载")
        return
    _q_texts = [r["question"] for r in _q_data]
    try:
        emb = get_bge_model()
        expected_dim = int(emb.embed_query("维度探测").shape[0])
        vecs = _load_vec_cache(_q_texts, expected_dim)
        if vecs is None:
            vecs = emb.embed_documents(_q_texts)
            if hasattr(vecs, "toarray"):
                vecs = vecs.toarray()
            vecs = np.asarray(vecs, dtype=np.float32)
            _save_vec_cache(_q_texts, vecs)
            logger.info(f"✅ 题库向量编码完成: {len(_q_data)} 题")
        _q_vecs = vecs
        if _q_vecs.ndim == 1:
            _q_vecs = _q_vecs.reshape(1, -1)
    except Exception as e:
        logger.warning(f"题库向量编码失败: {e}，相似题检索将不可用")
        _q_vecs = np.array([])


def load_question_bank():
    init_bank()
    return _q_data


def search_similar_question(query, top_k=None):
    top_k = top_k or QUESTION_TOP_K
    init_bank()
    if not _q_data or _q_vecs is None or len(_q_vecs) == 0:
        return []

    try:
        emb = get_bge_model()
        q_vec = emb.embed_query(query)
        if len(q_vec.shape) == 1:
            q_vec = q_vec.reshape(1, -1)
        vecs = _q_vecs
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        sims = cosine_similarity(q_vec, vecs)[0]
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [_q_data[i] for i in idxs]
    except Exception as e:
        logger.error(f"相似题检索失败: {e}")
        return []


def bank_stats():
    """题库统计信息（GUI 展示用；只读 CSV 行数，不触发向量编码）"""
    rows = _load_all_rows()
    return {
        "file_total": len(rows),
        "loaded": len(_q_data),
    }


if __name__ == "__main__":
    init_bank()
    print(f"题库条目数: {len(_q_data)}")
    if _q_data:
        sample = search_similar_question("高血压")
        print("相似题示例:", sample[:2] if sample else "无")
