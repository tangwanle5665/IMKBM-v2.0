import os
import pickle
import re
import numpy as np
from scipy.sparse import csr_matrix
import joblib  # 新增
from config import (
    VECTOR_DB_DIR, RETRIEVE_TOP_K, RETRIEVE_FINAL_K,
    MAX_CONTEXT_LENGTH, TFIDF_WEIGHT, SEMANTIC_WEIGHT,
    QUERY_EXPAND_ENABLED
)
from utils import logger, get_tfidf_model, get_bge_model, expand_medical_query

# 标量量化(SQ8)开关：True=内存降75%、检索更快（精度损失极小）；False=全精度 FlatIP
USE_SQ8_INDEX = True

# ========== 全局缓存 ==========
_CACHED_BGE = None
_FAISS_INDEX = None
_CACHED_CHUNKS = None
_CACHED_TFIDF_VECTORS = None
_TFIDF_MODEL_CACHE = None   # 检索用的 TF-IDF 模型缓存

def get_cached_bge():
    global _CACHED_BGE
    if _CACHED_BGE is None:
        _CACHED_BGE = get_bge_model()
    return _CACHED_BGE

def load_faiss_index():
    global _FAISS_INDEX
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    # 从 numpy 向量文件在内存构建 FAISS（避免 faiss 文件读写的中文路径/序列化问题）
    npy_path = os.path.join(VECTOR_DB_DIR, "bge_vectors.npy")
    try:
        if not os.path.exists(npy_path):
            with open(os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl"), "rb") as f:
                data = pickle.load(f)
            bv = np.ascontiguousarray(data["bge_vectors"], dtype=np.float32)
            np.save(npy_path, bv)
        bv = np.load(npy_path)
        import faiss
        bv = bv / (np.linalg.norm(bv, axis=1, keepdims=True) + 1e-10)
        bv = np.ascontiguousarray(bv, dtype=np.float32)
        if USE_SQ8_INDEX:
            # 标量量化(8bit)：内存降约 75%，检索更快，精度损失极小
            idx = faiss.IndexScalarQuantizer(bv.shape[1], faiss.ScalarQuantizer.QT_8bit)
            idx.train(bv)
            idx.add(bv)
        else:
            idx = faiss.IndexFlatIP(bv.shape[1])
            idx.add(bv)
        _FAISS_INDEX = idx
        logger.info(f"✅ FAISS 索引已在内存构建: {idx.ntotal} 条 (SQ8={USE_SQ8_INDEX}) from {npy_path}")
        return _FAISS_INDEX
    except Exception as e:
        logger.warning(f"FAISS 构建失败: {e}")
    logger.warning("⚠️ 未找到可用的 FAISS 索引，将使用传统 np.dot 检索")
    return None

# ⭐ 改进：支持 joblib 加速
def load_vector_data():
    """加载 chunks 和 tfidf_vectors（优先使用 joblib 缓存）"""
    global _CACHED_CHUNKS, _CACHED_TFIDF_VECTORS
    if _CACHED_CHUNKS is not None and _CACHED_TFIDF_VECTORS is not None:
        return _CACHED_CHUNKS, _CACHED_TFIDF_VECTORS

    joblib_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.joblib")
    index_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl")

    # 1. 优先 joblib
    if os.path.exists(joblib_path):
        try:
            data = joblib.load(joblib_path)
            _CACHED_CHUNKS = data["chunks"]
            _CACHED_TFIDF_VECTORS = data["tfidf_vectors"]
            logger.info("✅ 向量库已加载 (joblib 加速)")
            return _CACHED_CHUNKS, _CACHED_TFIDF_VECTORS
        except Exception as e:
            logger.warning(f"joblib 加载失败: {e}，尝试 pickle")

    # 2. 回退 pickle
    if not os.path.exists(index_path):
        return None, None

    with open(index_path, "rb") as f:
        data = pickle.load(f)

    _CACHED_CHUNKS = data["chunks"]
    vecs = data["tfidf_vectors"]
    # 转为稀疏
    if not isinstance(vecs, csr_matrix):
        vecs = csr_matrix(vecs)
    _CACHED_TFIDF_VECTORS = vecs

    # 生成 joblib 缓存
    try:
        joblib.dump({"chunks": _CACHED_CHUNKS, "tfidf_vectors": vecs}, joblib_path)
        logger.info("✅ 已生成 joblib 缓存，下次启动将加速")
    except Exception as e:
        logger.warning(f"保存 joblib 缓存失败: {e}")

    logger.info(f"✅ 加载 chunks: {len(_CACHED_CHUNKS)}, tfidf_vectors 稀疏形状: {_CACHED_TFIDF_VECTORS.shape}")
    return _CACHED_CHUNKS, _CACHED_TFIDF_VECTORS

def get_search_tfidf_model():
    """加载检索用 TF-IDF 模型（全局缓存，避免每次检索都重新加载）"""
    global _TFIDF_MODEL_CACHE
    if _TFIDF_MODEL_CACHE is not None:
        return _TFIDF_MODEL_CACHE
    model_path = os.path.join(VECTOR_DB_DIR, "tfidf_model.pkl")
    if not os.path.exists(model_path):
        with open(os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl"), "rb") as f:
            data_full = pickle.load(f)
            tfidf_model = data_full["tfidf_model"]
            import joblib
            joblib.dump(tfidf_model, model_path)
    else:
        import joblib
        tfidf_model = joblib.load(model_path)
    _TFIDF_MODEL_CACHE = tfidf_model
    return tfidf_model


def normalize_scores(scores):
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-6:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)

def keyword_fuzzy_match(query, chunks, cand_indices=None):
    """关键词模糊加分：只对混合分数最高的候选子集计算（全量 6.5 万条循环太慢）"""
    query_terms = set(re.findall(r"[\u4e00-\u9fa5]{2,}", query))
    if not query_terms:
        query_terms = set(query.lower().split())
    if not query_terms:
        return np.zeros(len(chunks))

    scores = np.zeros(len(chunks))
    if cand_indices is None:
        cand_indices = range(len(chunks))
    for i in cand_indices:
        text = chunks[i].page_content.lower()
        scores[i] = sum(1 for term in query_terms if term in text)
    return scores

def search_knowledge(query, top_k=None):
    top_k = top_k or RETRIEVE_TOP_K
    final_k = RETRIEVE_FINAL_K

    chunks, tfidf_vectors = load_vector_data()
    if chunks is None:
        return "知识库不存在，请先构建向量库"

    # 查询扩展
    queries = [query]
    if QUERY_EXPAND_ENABLED:
        queries = expand_medical_query(query)
        if len(queries) > 1:
            logger.info(f"查询扩展: {queries}")

    # 加载 TF-IDF 模型（全局缓存）
    tfidf_model = get_search_tfidf_model()

    all_tfidf_scores = np.zeros(len(chunks))
    for q in queries:
        q_vec = tfidf_model.embed_query(q)
        scores = tfidf_vectors.dot(q_vec).flatten()
        all_tfidf_scores = np.maximum(all_tfidf_scores, scores)

    # BGE（只对原查询编码，查询扩展仅用于 TF-IDF，减少 4 倍编码/检索开销）
    bge = get_cached_bge()
    q_vec = bge.embed_query(query).reshape(1, -1).astype(np.float32)

    faiss_index = load_faiss_index()
    bge_scores = np.zeros(len(chunks))
    if faiss_index is not None:
        scores, indices = faiss_index.search(q_vec, top_k * 2)
        for idx, sc in zip(indices[0], scores[0]):
            if sc > bge_scores[idx]:
                bge_scores[idx] = sc
    else:
        logger.warning("FAISS 索引不可用，仅使用 TF-IDF 检索")
        bge_scores = np.zeros(len(chunks))

    # 归一化
    norm_tfidf = normalize_scores(all_tfidf_scores)
    norm_bge = normalize_scores(bge_scores)
    hybrid_scores = TFIDF_WEIGHT * norm_tfidf + SEMANTIC_WEIGHT * norm_bge

    # 模糊匹配只对混合分数最高的候选子集做（全量 6.5 万条逐条匹配太慢）
    cand_k = max(top_k * 4, 200)
    cand_indices = np.argsort(hybrid_scores)[-cand_k:].tolist()
    fuzzy_scores = keyword_fuzzy_match(query, chunks, cand_indices)
    fuzzy_scores = normalize_scores(fuzzy_scores)
    final_scores = 0.85 * hybrid_scores + 0.15 * fuzzy_scores

    top_indices = np.argsort(final_scores)[-top_k:][::-1]

    seen_texts = set()
    unique_indices = []
    for idx in top_indices:
        text = chunks[idx].page_content[:100]
        if text not in seen_texts:
            seen_texts.add(text)
            unique_indices.append(idx)
        if len(unique_indices) >= final_k:
            break

    ctx_parts = []
    for idx in unique_indices:
        score = final_scores[idx]
        content = chunks[idx].page_content
        ctx_parts.append(f"[相关度{score:.3f}]\n{content}")

    ctx = "\n\n".join(ctx_parts)
    if len(ctx) > MAX_CONTEXT_LENGTH:
        ctx = ctx[:MAX_CONTEXT_LENGTH] + "..."

    return ctx.strip()
