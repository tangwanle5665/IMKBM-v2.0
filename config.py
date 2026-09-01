#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置文件（完整修复版）
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 教材解析后的文本目录
PARSED_DIR = os.path.join(BASE_DIR, "parsed_texts")
os.makedirs(PARSED_DIR, exist_ok=True)

# ⭐ 药物说明书目录（新增）
DRUG_LABEL_DIR = os.path.join(BASE_DIR, "drug_labels")
os.makedirs(DRUG_LABEL_DIR, exist_ok=True)

# ⭐ PDF/Word 教材源文件目录
PDF_DIR = os.path.join(BASE_DIR, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# ⭐ PDF/Word 解析后的文本输出目录
TXT_DIR = os.path.join(BASE_DIR, "parsed_texts")
os.makedirs(TXT_DIR, exist_ok=True)

# ⭐ PPT 源文件目录
PPT_DIR = os.path.join(BASE_DIR, "ppt")
os.makedirs(PPT_DIR, exist_ok=True)

# ⭐ PPT 解析后的文本输出目录
PPT_TXT_DIR = os.path.join(BASE_DIR, "parsed_texts", "ppt")
os.makedirs(PPT_TXT_DIR, exist_ok=True)

# 向量数据库目录
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# 题库目录
QUESTION_BANK_DIR = os.path.join(BASE_DIR, "question_bank")
os.makedirs(QUESTION_BANK_DIR, exist_ok=True)

# ⭐ 题库文件路径（CSV 格式）
QUESTION_BANK_PATH = os.path.join(BASE_DIR, "题库.docx.csv")

# 图片临时目录
IMAGE_DIR = os.path.join(BASE_DIR, "images", "temp")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==================== 分块参数 ====================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ⭐ 说明书检索权重系数（新增）
# <1.0 降低说明书在混合检索中的权重，避免淹没教材
DRUG_LABEL_WEIGHT = 0.7

# ==================== TF-IDF 参数 ====================
TFIDF_MAX_FEATURES = 10000

# ==================== BGE 模型配置 ====================
BGE_MODEL_NAME = "BAAI/bge-small-zh"
BGE_DEVICE = "cpu"
BGE_BATCH_SIZE = 32

# ==================== DeepSeek API 配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-3ed13018815f4e3d9dec1afbfbd111e8")#输入你的API密钥
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

API_TIMEOUT = 60
API_RETRY_TIMES = 3

# ==================== 检索参数 ====================
QUERY_EXPAND_ENABLED = True
QUESTION_TOP_K = 5
KNOWLEDGE_TOP_K = 8

# ⭐ 混合检索参数（retrieval_test.py 使用）
RETRIEVE_TOP_K = 15          # 初步检索 Top-K
RETRIEVE_FINAL_K = 8         # 去重后最终保留数量
MAX_CONTEXT_LENGTH = 6000    # 上下文最大长度
TFIDF_WEIGHT = 0.4           # TF-IDF 混合权重
SEMANTIC_WEIGHT = 0.6        # BGE 语义混合权重
SPLIT_SEPARATORS = ["\n\n", "\n", "。", "；", " ", ""]  # 文本分割符

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"

# ==================== 模型路径配置 ====================
# BGE 模型本地目录（与 download_model.py 下载路径一致）
MODEL_DIR = os.path.join(BASE_DIR, "models", "bge-small-zh")

# 嵌入设备（与 BGE 设备配置保持一致）
EMBED_DEVICE = BGE_DEVICE

# ==================== 日志文件路径 ====================
LOG_FILE = os.path.join(BASE_DIR, "app.log")
