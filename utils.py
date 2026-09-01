#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import pickle
import re
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from config import VECTOR_DB_DIR, LOG_FILE, MODEL_DIR, EMBED_DEVICE

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==================== 医学同义词扩展表 ====================
MEDICAL_SYNONYMS = {
    "心梗": ["心肌梗死", "心肌梗塞", "急性冠脉综合征"],
    "心肌梗死": ["心梗", "心肌梗塞"],
    "高血压": ["血压高", "原发性高血压", "继发性高血压"],
    "心衰": ["心力衰竭", "心功能不全"],
    "房颤": ["心房颤动", "心房纤颤"],
    "冠心病": ["冠状动脉粥样硬化性心脏病", "缺血性心脏病"],
    "慢阻肺": ["慢性阻塞性肺疾病", "COPD", "慢性支气管炎", "肺气肿"],
    "COPD": ["慢性阻塞性肺疾病", "慢阻肺"],
    "哮喘": ["支气管哮喘"],
    "肺炎": ["肺部感染", "肺实质炎症"],
    "胃溃疡": ["消化性溃疡", "胃十二指肠溃疡"],
    "肝硬化": ["肝纤维化", "门脉高压"],
    "胰腺炎": ["急性胰腺炎", "慢性胰腺炎"],
    "糖尿病": ["高血糖症", "2型糖尿病", "1型糖尿病"],
    "甲亢": ["甲状腺功能亢进", "甲状腺毒症"],
    "甲减": ["甲状腺功能减退"],
    "脑梗": ["脑梗死", "脑梗塞", "缺血性脑卒中"],
    "脑卒中": ["中风", "脑血管意外", "脑梗死", "脑出血"],
    "中风": ["脑卒中", "脑血管意外"],
    "癫痫": ["羊角风", "抽搐发作"],
    "癌症": ["恶性肿瘤", "癌", "肉瘤"],
    "肿瘤": ["新生物", "肿块", "占位性病变"],
    "乙肝": ["乙型肝炎", "HBV感染"],
    "艾滋病": ["AIDS", "获得性免疫缺陷综合征", "HIV感染"],
    "结核": ["肺结核", "结核病", "TB"],
    "骨折": ["骨裂", "骨断", "骨损伤"],
    "贫血": ["低血红蛋白", "红细胞减少"],
    "发烧": ["发热", "体温升高"],
    "拉肚子": ["腹泻", "肠炎", "痢疾"],
    "肚子疼": ["腹痛", "腹部疼痛", "腹绞痛"],
    "头疼": ["头痛", "偏头痛", "头部疼痛"],
    "腰疼": ["腰痛", "腰部疼痛", "腰肌劳损"],
    "高血压危象": ["高血压急症", "恶性高血压"],
    "休克": ["循环衰竭", "低血压状态"],
}


def expand_medical_query(query: str) -> list:
    expanded = [query]
    query_lower = query.lower()
    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term in query_lower:
            for syn in synonyms:
                new_query = query_lower.replace(term, syn)
                if new_query not in expanded:
                    expanded.append(new_query)
    return expanded


# ==================== TF-IDF 向量器 ====================
class LocalTFIDFEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed_documents(self, texts):
        if not self.fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts)

    def embed_query(self, text):
        if not self.fitted:
            raise RuntimeError("需要先拟合数据")
        return self.vectorizer.transform([text]).toarray()[0]


# ==================== ONNX BGE 高性能推理封装（长度分桶版）====================
class ONNXBGEEmbeddings:
    def __init__(self, model_path, threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        onnx_path = os.path.join(model_path, "model.onnx")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX 模型不存在: {onnx_path}，请先运行 export_onnx_native.py"
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        if threads is None:
            import multiprocessing
            threads = max(1, multiprocessing.cpu_count() // 2)
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1

        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        self.input_names = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.hidden_size = self.session.get_outputs()[0].shape[-1] or 512

        logger.info(
            f"🚀 ONNX Runtime 加载成功 | 线程: {threads} | 维度: {self.hidden_size}"
        )

    def _run_inference(self, texts, max_length=512):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )

        ort_inputs = {}
        for name in self.input_names:
            if name in inputs:
                ort_inputs[name] = inputs[name].astype(np.int64)
            elif name == 'token_type_ids':
                ort_inputs[name] = np.zeros_like(inputs['input_ids'], dtype=np.int64)

        outputs = self.session.run(self.output_names, ort_inputs)
        return outputs[0], inputs['attention_mask']

    def _mean_pooling(self, last_hidden, attention_mask):
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        masked = last_hidden * mask
        return masked.sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

    def _normalize(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    ):
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.hidden_size)

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm.auto import tqdm
            total = (len(texts) + batch_size - 1) // batch_size
            iterator = tqdm(iterator, total=total, desc="ONNX编码")

        for i in iterator:
            batch = texts[i:i + batch_size]
            last_hidden, attention_mask = self._run_inference(batch)
            pooled = self._mean_pooling(last_hidden, attention_mask)
            if normalize_embeddings:
                pooled = self._normalize(pooled)
            all_embeddings.append(pooled.astype(np.float32))

        return np.vstack(all_embeddings)


# ==================== fastembed 官方 ONNX BGE（低内存轻量推理） ====================
FASTEMBED_CACHE_DIR = os.path.join(MODEL_DIR, "fastembed_cache")


class FastEmbedBGE:
    """fastembed + ONNX Runtime：官方 ONNX 模型（BAAI/bge-small-zh-v1.5，512维）
    相比 PyTorch：内存 -60~70%，CPU 编码快数倍，无需 torch 常驻"""

    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    def __init__(self, cache_dir=None):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # hf-mirror 不支持 xet，禁用
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像
        self.cache_dir = cache_dir or FASTEMBED_CACHE_DIR
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=self.MODEL_NAME, cache_dir=self.cache_dir)
        self.dim = 512

    def encode(self, texts, batch_size=64):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self.model.embed(list(texts), batch_size=batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text):
        q = f"为这个句子生成表示以用于检索相关文章: {text}"
        return self.encode([q], batch_size=1)[0]


# ==================== BGE 语义向量器（fastembed 优先，ONNX/PyTorch 回退） ====================
class BGEEmbeddings:
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or MODEL_DIR
        self.device = device or EMBED_DEVICE
        self.model = None
        self.onnx_embedder = None
        self.fast_embedder = None  # ⭐ fastembed 官方 ONNX（首选，低内存）

        # CPU 模式优先 fastembed（轻量），失败再走本地 ONNX / PyTorch
        if self.device == "cpu":
            self._try_load_fastembed()
            if self.fast_embedder is None:
                self._try_load_onnx()

        # 全部失败则加载 PyTorch
        if self.fast_embedder is None and self.onnx_embedder is None:
            self._load_pytorch_model()

        # batch_size 按后端设置
        if self.device == "cuda":
            self.batch_size = 256
        elif self.fast_embedder:
            self.batch_size = 128
        else:
            self.batch_size = 512 if self.onnx_embedder else 64

    def _try_load_fastembed(self):
        try:
            self.fast_embedder = FastEmbedBGE()
            logger.info("✅ 已启用 fastembed ONNX 推理（低内存）")
        except Exception as e:
            logger.warning(f"⚠️ fastembed 加载失败，回退本地 ONNX/PyTorch: {e}")
            self.fast_embedder = None

    def _try_load_onnx(self):
        try:
            self.onnx_embedder = ONNXBGEEmbeddings(self.model_path)
            logger.info("✅ 已启用 ONNX Runtime 加速（绕过 PyTorch）")
        except Exception as e:
            logger.warning(f"⚠️ ONNX 加载失败，回退 PyTorch: {e}")
            self.onnx_embedder = None

    def _load_pytorch_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            if self.device == "cpu" and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("✅ 检测到 CUDA，自动切换到 GPU")

            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs = {'torch_dtype': torch.float16}
                logger.info("🚀 GPU 模式：FP16 加速")
            else:
                logger.info("💻 CPU 模式：PyTorch 推理")

            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
                model_kwargs=model_kwargs
            )

            if self.device == "cpu" and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("⚡ torch.compile 已启用")
                except Exception:
                    pass

            logger.info(f"✅ PyTorch BGE 加载成功: {self.model_path}")

        except ImportError:
            logger.error("❌ 未安装 sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"❌ BGE 模型加载失败: {e}")
            raise

    def embed_documents(self, texts, show_progress_bar=True, **kwargs):
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 512)
        if self.fast_embedder:
            return self.fast_embedder.encode(texts, batch_size=self.batch_size)
        if self.onnx_embedder:
            return self.onnx_embedder.encode(
                texts, batch_size=self.batch_size, show_progress_bar=show_progress_bar
            )
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            **kwargs
        )

    def embed_query(self, text):
        query_text = f"为这个句子生成表示以用于检索相关文章: {text}"
        if self.fast_embedder:
            return self.fast_embedder.encode_query(text)
        if self.onnx_embedder:
            result = self.onnx_embedder.encode([query_text], batch_size=1)
            return result[0]
        return self.model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def embed_queries(self, texts, **kwargs):
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 512)
        queries = [f"为这个句子生成表示以用于检索相关文章: {t}" for t in texts]
        if self.fast_embedder:
            return self.fast_embedder.encode(queries, batch_size=self.batch_size)
        if self.onnx_embedder:
            return self.onnx_embedder.encode(
                queries, batch_size=self.batch_size, show_progress_bar=False
            )
        return self.model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            **kwargs
        )


# ==================== 单例管理 ====================
_tfidf_instance = None
_bge_instance = None
_vector_db_cache = None


def get_tfidf_model():
    global _tfidf_instance
    if _tfidf_instance is None:
        logger.info("初始化 TF-IDF 向量器（内存优化版）")
        _tfidf_instance = LocalTFIDFEmbeddings()
    return _tfidf_instance


def get_bge_model():
    global _bge_instance
    if _bge_instance is None:
        logger.info("初始化 BGE 语义向量器（ONNX 优先）")
        _bge_instance = BGEEmbeddings()
    return _bge_instance


def get_embedding_model():
    return get_tfidf_model()


# ==================== 加载向量库（强制 joblib 优先） ====================
def load_vector_db():
    global _vector_db_cache
    if _vector_db_cache is not None:
        return _vector_db_cache

    joblib_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.joblib")
    index_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.pkl")
    legacy_path = os.path.join(VECTOR_DB_DIR, "tfidf_index.pkl")

    if os.path.exists(joblib_path):
        try:
            logger.info(f"📂 加载 joblib 缓存: {joblib_path}")
            _vector_db_cache = joblib.load(joblib_path)
            logger.info("✅ 向量库已加载 (joblib 加速)")
            return _vector_db_cache
        except Exception as e:
            logger.warning(f"joblib 加载失败: {e}，尝试 pickle")

    if not os.path.exists(index_path):
        if os.path.exists(legacy_path):
            with open(legacy_path, "rb") as f:
                _vector_db_cache = pickle.load(f)
            return _vector_db_cache
        return None

    logger.info("📂 加载 pickle 缓存...")
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    if "tfidf_vectors" in data:
        vecs = data["tfidf_vectors"]
        if not isinstance(vecs, csr_matrix):
            try:
                data["tfidf_vectors"] = csr_matrix(vecs)
                logger.info("✅ TF-IDF 向量已转换为稀疏格式")
            except Exception as e:
                logger.warning(f"无法转换 tfidf_vectors 为稀疏: {e}")

    _vector_db_cache = data

    try:
        joblib.dump(data, joblib_path)
        logger.info("✅ 已生成 joblib 缓存，下次启动将加速")
    except Exception as e:
        logger.warning(f"保存 joblib 缓存失败: {e}")

    logger.info("✅ 向量库已加载到内存（缓存）")
    return _vector_db_cache


def clear_vector_db_cache():
    global _vector_db_cache
    _vector_db_cache = None
    joblib_path = os.path.join(VECTOR_DB_DIR, "hybrid_index.joblib")
    if os.path.exists(joblib_path):
        try:
            os.remove(joblib_path)
            logger.info("🗑️ 已删除 joblib 缓存")
        except Exception:
            pass
    logger.info("🗑️ 向量库缓存已清除")


def retry_request(func, retry_times, *args, **kwargs):
    for i in range(retry_times + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i == retry_times:
                raise e
            logger.warning(f"请求失败，第{i+1}次重试...")
            time.sleep(1)


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"创建文件夹: {dir_path}")
