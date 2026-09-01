#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import requests
import hashlib
import time
import traceback
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties, findfont
import platform
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval_test import search_knowledge
from question_bank import search_similar_question
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
    API_TIMEOUT, API_RETRY_TIMES, BASE_DIR
)
from utils import logger, retry_request, get_bge_model, get_tfidf_model

# ==================== 模块化 RAG + 记忆 + 安全 + Agent 导入 ====================
from rag_modules import ModularRAGPipeline, get_memory_manager, reset_memory_manager, SecurityGuard

# ==================== 中文字体设置 ====================
def get_chinese_font():
    font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'STHeiti', 'STSong']
    for font_name in font_names:
        try:
            fp = FontProperties(family=font_name)
            if findfont(fp, fallback_to_default=False) is not None:
                return font_name
        except:
            continue
    if platform.system() == 'Windows':
        try:
            windows_font_paths = [
                r'C:\Windows\Fonts\simhei.ttf',
                r'C:\Windows\Fonts\msyh.ttc',
                r'C:\Windows\Fonts\simsun.ttc',
                r'C:\Windows\Fonts\STSONG.TTF',
            ]
            for path in windows_font_paths:
                if os.path.exists(path):
                    return 'SimHei'
        except:
            pass
    return 'SimHei'

CHINESE_FONT_NAME = get_chinese_font()
if CHINESE_FONT_NAME:
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT_NAME]
    plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置路径 ====================
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
TEMP_IMAGE_DIR = os.path.join(BASE_DIR, "images", "temp")
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# ==================== 代谢途径绘图配置 ====================
C_NODE = "#E3F2FD"
C_EDGE = "#1565C0"
C_ARROW = "#37474F"
C_ENZ = "#BF360C"
C_ENZ_HOT = "#C62828"
C_ENERGY = "#00695C"
C_TITLE = "#1A237E"
C_MITO = "#FFF8E1"
C_CYTO = "#E8F5E9"

TCA_POS = {
    "柠檬酸": (0, 2), "异柠檬酸": (1.4, 1.4), "α-酮戊二酸": (2, 0),
    "琥珀酰CoA": (1.4, -1.4), "琥珀酸": (0, -2), "延胡索酸": (-1.4, -1.4),
    "苹果酸": (-2, 0), "草酰乙酸": (-1.4, 1.4),
}
TCA_EDGES = [
    ("柠檬酸","异柠檬酸"), ("异柠檬酸","α-酮戊二酸"), ("α-酮戊二酸","琥珀酰CoA"),
    ("琥珀酰CoA","琥珀酸"), ("琥珀酸","延胡索酸"), ("延胡索酸","苹果酸"),
    ("苹果酸","草酰乙酸"), ("草酰乙酸","柠檬酸")
]
TCA_ENZ = ["柠檬酸合酶", "异柠檬酸脱氢酶", "α-酮戊二酸脱氢酶复合体",
           "琥珀酰CoA合成酶", "琥珀酸脱氢酶", "延胡索酸酶", "苹果酸脱氢酶", "柠檬酸合酶"]

GLY_POS = {
    "葡萄糖": (0, 5), "葡萄糖-6-磷酸": (0, 4), "果糖-6-磷酸": (0, 3),
    "果糖-1,6-二磷酸": (0, 2), "磷酸二羟丙酮": (-1.5, 1), "3-磷酸甘油醛": (1.5, 1),
    "1,3-二磷酸甘油酸": (0, 0), "3-磷酸甘油酸": (0, -1),
    "2-磷酸甘油酸": (0, -2), "磷酸烯醇式丙酮酸": (0, -3), "丙酮酸": (0, -4),
}
GLY_EDGES = [
    ("葡萄糖","葡萄糖-6-磷酸"), ("葡萄糖-6-磷酸","果糖-6-磷酸"), ("果糖-6-磷酸","果糖-1,6-二磷酸"),
    ("果糖-1,6-二磷酸","磷酸二羟丙酮"), ("果糖-1,6-二磷酸","3-磷酸甘油醛"),
    ("磷酸二羟丙酮","3-磷酸甘油醛"), ("3-磷酸甘油醛","1,3-二磷酸甘油酸"),
    ("1,3-二磷酸甘油酸","3-磷酸甘油酸"), ("3-磷酸甘油酸","2-磷酸甘油酸"),
    ("2-磷酸甘油酸","磷酸烯醇式丙酮酸"), ("磷酸烯醇式丙酮酸","丙酮酸")
]
GLY_ENZ = ["己糖激酶", "磷酸葡萄糖异构酶", "磷酸果糖激酶-1", "醛缩酶", "醛缩酶",
           "丙糖磷酸异构酶", "GAPDH", "磷酸甘油酸激酶", "磷酸甘油酸变位酶",
           "烯醇化酶", "丙酮酸激酶"]

GLU_POS = {
    "丙酮酸": (0, -4), "草酰乙酸": (2.2, -2), "磷酸烯醇式丙酮酸": (0, -2),
    "2-磷酸甘油酸": (0, -1), "3-磷酸甘油酸": (0, 0),
    "1,3-二磷酸甘油酸": (0, 1), "3-磷酸甘油醛": (0, 2),
    "磷酸二羟丙酮": (-1.5, 2), "果糖-1,6-二磷酸": (0, 3),
    "果糖-6-磷酸": (0, 4), "葡萄糖-6-磷酸": (0, 5), "葡萄糖": (0, 6),
}
GLU_EDGES = [
    ("丙酮酸","草酰乙酸"), ("草酰乙酸","磷酸烯醇式丙酮酸"),
    ("磷酸烯醇式丙酮酸","2-磷酸甘油酸"), ("2-磷酸甘油酸","3-磷酸甘油酸"),
    ("3-磷酸甘油酸","1,3-二磷酸甘油酸"), ("1,3-二磷酸甘油酸","3-磷酸甘油醛"),
    ("3-磷酸甘油醛","磷酸二羟丙酮"), ("3-磷酸甘油醛","果糖-1,6-二磷酸"),
    ("磷酸二羟丙酮","果糖-1,6-二磷酸"), ("果糖-1,6-二磷酸","果糖-6-磷酸"),
    ("果糖-6-磷酸","葡萄糖-6-磷酸"), ("葡萄糖-6-磷酸","葡萄糖")
]
GLU_ENZ = ["丙酮酸羧化酶", "PEPCK", "烯醇化酶(逆)", "磷酸甘油酸变位酶(逆)",
           "GAPDH(逆)", "GAPDH(逆)", "丙糖磷酸异构酶", "醛缩酶(逆)", "醛缩酶(逆)",
           "果糖二磷酸酶-1", "葡萄糖-6-磷酸酶", "己糖激酶(绕过)"]

UREA_POS = {
    "NH₃+CO₂": (0, 2), "氨基甲酰磷酸": (0, 1), "瓜氨酸": (0, 0),
    "天冬氨酸": (2, 0), "精氨酸代琥珀酸": (2, -1.5),
    "延胡索酸": (4, -1.5), "精氨酸": (2, -3),
    "尿素": (0, -3), "鸟氨酸": (0, -1.5),
}
UREA_EDGES = [
    ("NH₃+CO₂","氨基甲酰磷酸"), ("氨基甲酰磷酸","瓜氨酸"), ("瓜氨酸","精氨酸代琥珀酸"),
    ("天冬氨酸","精氨酸代琥珀酸"), ("精氨酸代琥珀酸","延胡索酸"),
    ("精氨酸代琥珀酸","精氨酸"), ("精氨酸","尿素"), ("精氨酸","鸟氨酸"),
    ("鸟氨酸","瓜氨酸")
]
UREA_ENZ = ["CPS-I", "鸟氨酸氨基甲酰转移酶", "精氨酸代琥珀酸合成酶",
            "精氨酸代琥珀酸合成酶", "精氨酸代琥珀酸裂解酶",
            "精氨酸代琥珀酸裂解酶", "精氨酸酶", "精氨酸酶", "鸟氨酸转位酶"]

BETA_POS = {
    "脂酰CoA": (0, 2), "烯脂酰CoA": (0, 1), "L-羟脂酰CoA": (0, 0),
    "β-酮脂酰CoA": (0, -1), "乙酰CoA": (0, -2),
}
BETA_EDGES = [
    ("脂酰CoA","烯脂酰CoA"), ("烯脂酰CoA","L-羟脂酰CoA"),
    ("L-羟脂酰CoA","β-酮脂酰CoA"), ("β-酮脂酰CoA","乙酰CoA")
]
BETA_ENZ = ["脂酰CoA脱氢酶", "烯酰CoA水化酶", "羟脂酰CoA脱氢酶", "β-酮硫解酶"]

PPP_POS = {
    "葡萄糖-6-磷酸": (0, 3), "6-磷酸葡萄糖酸-δ-内酯": (1.5, 3),
    "6-磷酸葡萄糖酸": (3, 3), "核酮糖-5-磷酸": (4.5, 3),
    "核糖-5-磷酸": (6, 3), "木酮糖-5-磷酸": (4.5, 1.5),
    "景天糖-7-磷酸": (3, 1.5), "赤藓糖-4-磷酸": (1.5, 1.5),
    "果糖-6-磷酸": (0, 1.5), "3-磷酸甘油醛": (0, 0),
}
PPP_EDGES = [
    ("葡萄糖-6-磷酸","6-磷酸葡萄糖酸-δ-内酯"), ("6-磷酸葡萄糖酸-δ-内酯","6-磷酸葡萄糖酸"),
    ("6-磷酸葡萄糖酸","核酮糖-5-磷酸"), ("核酮糖-5-磷酸","核糖-5-磷酸"),
    ("核酮糖-5-磷酸","木酮糖-5-磷酸"), ("木酮糖-5-磷酸","景天糖-7-磷酸"),
    ("景天糖-7-磷酸","赤藓糖-4-磷酸"), ("赤藓糖-4-磷酸","果糖-6-磷酸"),
    ("木酮糖-5-磷酸","果糖-6-磷酸"), ("果糖-6-磷酸","葡萄糖-6-磷酸"),
    ("3-磷酸甘油醛","果糖-6-磷酸")
]
PPP_ENZ = ["G6PD", "内酯酶", "6PGD", "磷酸戊糖异构酶", "磷酸戊糖差向异构酶",
           "转酮醇酶", "转醛醇酶", "转醛醇酶", "转酮醇酶", "异构酶", ""]

CHOL_POS = {
    "乙酰CoA": (0, 4), "乙酰乙酰CoA": (0, 3), "HMG-CoA": (0, 2),
    "甲羟戊酸": (0, 1), "鲨烯": (0, 0), "羊毛固醇": (0, -1), "胆固醇": (0, -2),
}
CHOL_EDGES = [
    ("乙酰CoA","乙酰乙酰CoA"), ("乙酰乙酰CoA","HMG-CoA"), ("HMG-CoA","甲羟戊酸"),
    ("甲羟戊酸","鲨烯"), ("鲨烯","羊毛固醇"), ("羊毛固醇","胆固醇")
]
CHOL_ENZ = ["硫解酶", "HMG-CoA合酶", "HMG-CoA还原酶", "多步酶促", "鲨烯单加氧酶", "多步酶促"]

GLYCOGEN_POS = {
    "葡萄糖": (0, 3), "葡萄糖-6-磷酸": (0, 2), "葡萄糖-1-磷酸": (0, 1),
    "UDP-葡萄糖": (0, 0), "糖原(n)": (0, -1), "糖原(n+1)": (0, -2),
    "乳酸": (2, 2), "丙酮酸": (2, 1),
}
GLYCOGEN_EDGES = [
    ("葡萄糖","葡萄糖-6-磷酸"), ("葡萄糖-6-磷酸","葡萄糖-1-磷酸"),
    ("葡萄糖-1-磷酸","UDP-葡萄糖"), ("UDP-葡萄糖","糖原(n+1)"),
    ("糖原(n+1)","糖原(n)"), ("糖原(n)","葡萄糖-1-磷酸"),
    ("葡萄糖-6-磷酸","乳酸"), ("乳酸","丙酮酸")
]
GLYCOGEN_ENZ = ["己糖激酶/葡萄糖激酶", "磷酸葡萄糖变位酶", "UDP-葡萄糖焦磷酸化酶",
                "糖原合酶", "糖原磷酸化酶", "糖原磷酸化酶", "无氧酵解", "乳酸脱氢酶"]

PATHWAY_CONFIG = {
    "三羧酸循环": {
        "nodes": list(TCA_POS.keys()), "pos": TCA_POS, "edges": TCA_EDGES,
        "enzymes": TCA_ENZ, "compartment": "线粒体", "title": "三羧酸循环 (TCA Cycle)"
    },
    "糖酵解": {
        "nodes": list(GLY_POS.keys()), "pos": GLY_POS, "edges": GLY_EDGES,
        "enzymes": GLY_ENZ, "compartment": "胞质", "title": "糖酵解途径 (Glycolysis)"
    },
    "糖异生": {
        "nodes": list(GLU_POS.keys()), "pos": GLU_POS, "edges": GLU_EDGES,
        "enzymes": GLU_ENZ, "compartment": "胞质/线粒体", "title": "糖异生 (Gluconeogenesis)"
    },
    "尿素循环": {
        "nodes": list(UREA_POS.keys()), "pos": UREA_POS, "edges": UREA_EDGES,
        "enzymes": UREA_ENZ, "compartment": "线粒体/胞质", "title": "尿素循环 (Urea Cycle)"
    },
    "脂肪酸β氧化": {
        "nodes": list(BETA_POS.keys()), "pos": BETA_POS, "edges": BETA_EDGES,
        "enzymes": BETA_ENZ, "compartment": "线粒体", "title": "脂肪酸β-氧化 (β-Oxidation)"
    },
    "磷酸戊糖途径": {
        "nodes": list(PPP_POS.keys()), "pos": PPP_POS, "edges": PPP_EDGES,
        "enzymes": PPP_ENZ, "compartment": "胞质", "title": "磷酸戊糖途径 (PPP)"
    },
    "胆固醇合成": {
        "nodes": list(CHOL_POS.keys()), "pos": CHOL_POS, "edges": CHOL_EDGES,
        "enzymes": CHOL_ENZ, "compartment": "胞质/内质网", "title": "胆固醇合成"
    },
    "糖原": {
        "nodes": list(GLYCOGEN_POS.keys()), "pos": GLYCOGEN_POS, "edges": GLYCOGEN_EDGES,
        "enzymes": GLYCOGEN_ENZ, "compartment": "胞质", "title": "糖原代谢 (合成与分解)"
    },
}


# ==================== 通路增强元数据（净产物/关键酶/能量标注/可逆边/记忆口诀） ====================
_PATHWAY_META = {
    "三羧酸循环": {
        "net_product": "每轮：3 NADH + 1 FADH2 + 1 GTP(ATP) + 2 CO2",
        "key_enzymes": ["柠檬酸合酶", "异柠檬酸脱氢酶", "α-酮戊二酸脱氢酶复合体"],
        "energy": {
            ("异柠檬酸", "α-酮戊二酸"): "NADH+CO2",
            ("α-酮戊二酸", "琥珀酰CoA"): "NADH+CO2",
            ("琥珀酰CoA", "琥珀酸"): "GTP",
            ("琥珀酸", "延胡索酸"): "FADH2",
            ("苹果酸", "草酰乙酸"): "NADH",
        },
        "rev_edges": [("苹果酸", "草酰乙酸"), ("琥珀酸", "延胡索酸")],
        "mnemonic": "柠异酮琥、琥延苹草转一圈；每圈 3NADH、1FADH2、1GTP、2CO2",
    },
    "糖酵解": {
        "net_product": "1 葡萄糖 → 2 丙酮酸 ＋ 2 ATP ＋ 2 NADH",
        "key_enzymes": ["己糖激酶", "磷酸果糖激酶-1", "丙酮酸激酶"],
        "energy": {
            ("葡萄糖", "葡萄糖-6-磷酸"): "-ATP",
            ("果糖-6-磷酸", "果糖-1,6-二磷酸"): "-ATP",
            ("3-磷酸甘油醛", "1,3-二磷酸甘油酸"): "NADHx2",
            ("1,3-二磷酸甘油酸", "3-磷酸甘油酸"): "+ATPx2",
            ("磷酸烯醇式丙酮酸", "丙酮酸"): "+ATPx2",
        },
        "rev_edges": [],
        "mnemonic": "两丙酮酸两氢两ATP；三大关键酶：己糖激酶、PFK-1、丙酮酸激酶",
    },
    "糖异生": {
        "net_product": "2 丙酮酸 → 1 葡萄糖，消耗 4 ATP + 2 GTP + 2 NADH",
        "key_enzymes": ["丙酮酸羧化酶", "PEPCK", "果糖二磷酸酶-1", "葡萄糖-6-磷酸酶"],
        "energy": {
            ("丙酮酸", "草酰乙酸"): "ATP",
            ("草酰乙酸", "磷酸烯醇式丙酮酸"): "GTP",
            ("3-磷酸甘油酸", "1,3-二磷酸甘油酸"): "-ATP",
            ("1,3-二磷酸甘油酸", "3-磷酸甘油醛"): "-NADH",
        },
        "rev_edges": [("磷酸烯醇式丙酮酸", "2-磷酸甘油酸"), ("3-磷酸甘油酸", "1,3-二磷酸甘油酸")],
        "mnemonic": "四ATP两GTP两NADH造糖；四大酶绕过不可逆：丙酮酸羧化酶、PEPCK、果糖二磷酸酶、G-6-Pase",
    },
    "尿素循环": {
        "net_product": "每轮生成 1 尿素，消耗 3 ATP + 1 天冬氨酸",
        "key_enzymes": ["CPS-I", "鸟氨酸氨基甲酰转移酶", "精氨酸代琥珀酸合成酶"],
        "energy": {
            ("NH3+CO2", "氨基甲酰磷酸"): "-2ATP",
            ("瓜氨酸", "精氨酸代琥珀酸"): "-ATP",
        },
        "rev_edges": [("精氨酸", "鸟氨酸")],
        "mnemonic": "氨碳线粒体变瓜氨酸，胞质加天冬氨酸，裂解出尿素回鸟氨酸；耗3ATP",
    },
    "脂肪酸β氧化": {
        "net_product": "每轮：1 FADH2 + 1 NADH + 1 乙酰CoA（软脂酸共 7 轮）",
        "key_enzymes": ["脂酰CoA脱氢酶", "β-酮硫解酶"],
        "energy": {
            ("脂酰CoA", "烯脂酰CoA"): "FADH2",
            ("L-羟脂酰CoA", "β-酮脂酰CoA"): "NADH",
        },
        "rev_edges": [],
        "mnemonic": "脱氢、加水、再脱氢、硫解一刀两半；每轮1FADH2+1NADH+1乙酰CoA",
    },
    "磷酸戊糖途径": {
        "net_product": "产生 NADPH（合成代谢供氢）与核糖-5-磷酸",
        "key_enzymes": ["G6PD", "6PGD"],
        "energy": {
            ("葡萄糖-6-磷酸", "6-磷酸葡萄糖酸-δ-内酯"): "NADPH",
            ("6-磷酸葡萄糖酸", "核酮糖-5-磷酸"): "NADPH+CO2",
        },
        "rev_edges": [("果糖-6-磷酸", "葡萄糖-6-磷酸")],
        "mnemonic": "磷酸戊糖旁路：氧化段产NADPH和CO2；G6PD、6PGD两个脱氢酶是关键",
    },
    "胆固醇合成": {
        "net_product": "乙酰CoA → 胆固醇；限速酶 HMG-CoA 还原酶（他汀靶点）",
        "key_enzymes": ["HMG-CoA还原酶"],
        "energy": {
            ("HMG-CoA", "甲羟戊酸"): "限速步",
        },
        "rev_edges": [],
        "mnemonic": "乙酰CoA→HMG-CoA→甲羟戊酸→鲨烯→羊毛固醇→胆固醇；HMG-CoA还原酶限速，他汀抑制",
    },
    "糖原": {
        "net_product": "糖原合酶（合成）↔ 糖原磷酸化酶（分解）",
        "key_enzymes": ["糖原合酶", "糖原磷酸化酶"],
        "energy": {
            ("葡萄糖-1-磷酸", "UDP-葡萄糖"): "-UTP",
        },
        "rev_edges": [("糖原(n+1)", "糖原(n)")],
        "mnemonic": "合成靠糖原合酶(G1P→UDPG→糖原)，分解靠糖原磷酸化酶(糖原→G1P)",
    },
}
for _k, _v in _PATHWAY_META.items():
    PATHWAY_CONFIG[_k].update(_v)


def draw_metabolic_pathway(pathway_name):
    """绘制代谢途径图（优化版：关键酶高亮 + 能量标注 + 可逆箭头 + 分区）"""
    if pathway_name not in PATHWAY_CONFIG:
        return None

    config = PATHWAY_CONFIG[pathway_name]
    nodes = config["nodes"]
    edges = config["edges"]
    pos = config["pos"]
    enzymes = config.get("enzymes", [])
    compartment = config.get("compartment", "")
    title = config.get("title", pathway_name)
    net_product = (config.get("net_product", "")
                   .replace("₂", "2").replace("₃", "3").replace("₄", "4"))
    key_enzymes = set(config.get("key_enzymes", []))
    energy = {k: v.replace("₂", "2").replace("₃", "3").replace("₄", "4")
              for k, v in config.get("energy", {}).items()}
    rev_edges = set(config.get("rev_edges", []))

    for node in nodes:
        if node not in pos:
            logger.warning(f"节点 {node} 缺少位置定义")
            return None

    fig, ax = plt.subplots(figsize=(12, 9))
    pos_dict = {node: pos[node] for node in nodes}

    # 亚细胞定位背景（分区着色 + 标签）
    if compartment == "线粒体":
        ax.set_facecolor(C_MITO)
        bg_label = "线粒体"
    elif compartment == "胞质":
        ax.set_facecolor(C_CYTO)
        bg_label = "胞质"
    elif compartment in ("胞质/线粒体", "线粒体/胞质"):
        ax.set_facecolor("#F3E5F5")
        bg_label = "胞质 / 线粒体"
    else:
        ax.set_facecolor("#FAFAFA")
        bg_label = ""

    # 边（箭头 + 酶 + 能量）
    for i, (u, v) in enumerate(edges):
        if u not in pos_dict or v not in pos_dict:
            continue
        x1, y1 = pos_dict[u]
        x2, y2 = pos_dict[v]
        rev = (u, v) in rev_edges
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="<->" if rev else "->",
                        color="#90A4AE" if rev else C_ARROW,
                        lw=2.0 if rev else 1.6,
                        connectionstyle="arc3,rad=0.05"))
        enz = enzymes[i] if i < len(enzymes) else ""
        if enz:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2) ** 0.5 or 1
            off_x, off_y = -dy / length * 0.28, dx / length * 0.28
            hot = enz in key_enzymes
            enz_disp = enz.replace("₂", "2").replace("₃", "3")
            label = ("★" + enz_disp) if hot else enz_disp
            ax.text(mx + off_x, my + off_y, label,
                    fontsize=8, color=C_ENZ_HOT if hot else C_ENZ,
                    style='italic' if not hot else 'normal',
                    fontweight='bold' if hot else 'normal',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF8E1' if hot else 'white',
                              alpha=0.92, edgecolor='#EF9A9A' if hot else 'none',
                              linewidth=0.8 if hot else 0))
        if (u, v) in energy:
            ex, ey = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(ex - off_x * 0.6, ey - off_y * 0.6, energy[(u, v)],
                    fontsize=7.5, color=C_ENERGY, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='#E0F2F1',
                              alpha=0.9, edgecolor='none'))

    # 节点（代谢物）
    for node, (x, y) in pos_dict.items():
        box = FancyBboxPatch(
            (x - 0.48, y - 0.19), 0.96, 0.38,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=C_NODE, edgecolor=C_EDGE, linewidth=1.8, zorder=3
        )
        ax.add_patch(box)
        node_disp = node.replace("₂", "2").replace("₃", "3")
        ax.text(x, y, node_disp, ha='center', va='center',
                fontsize=10, fontweight='bold', color="#1A237E", zorder=4)

    # 标题（净产物放图底部，避免与标题重叠/裁剪）
    ax.set_title(title, fontsize=15, fontweight='bold', color=C_TITLE, pad=14)
    # 底部信息：净产物 + 记忆口诀
    mnemonic = config.get("mnemonic", "")
    info_y = 0.055 if mnemonic else 0.012
    if net_product:
        fig.text(0.5, info_y, f"净产物: {net_product}", ha='center', va='bottom',
                 fontsize=9.5, color="#00695C",
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#E0F2F1',
                           alpha=0.85, edgecolor='#80CBC4'))
    if mnemonic:
        fig.text(0.5, 0.012, f"口诀: {mnemonic}", ha='center', va='bottom',
                 fontsize=9, color="#4A148C",
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E5F5',
                           alpha=0.88, edgecolor='#CE93D8'))
    if bg_label:
        ax.text(0.985, 0.97, f"区域: {bg_label}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color="#5D4037",
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.8, edgecolor='#BCAAA4'))

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=C_NODE, edgecolor=C_EDGE, label='代谢物'),
        mpatches.FancyArrowPatch((0, 0), (1, 0), color=C_ARROW, arrowstyle='->',
                                  mutation_scale=20, label='反应方向'),
        mpatches.FancyArrowPatch((0, 0), (1, 0), color="#90A4AE", arrowstyle='<->',
                                  mutation_scale=20, label='可逆反应'),
        mpatches.Patch(facecolor='white', edgecolor='none', label='普通酶(斜体)'),
        mpatches.Patch(facecolor='#FFF8E1', edgecolor='#EF9A9A', label='★ 关键/限速酶'),
        mpatches.Patch(facecolor='#E0F2F1', edgecolor='none', label='能量/产物'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5,
              framealpha=0.92, edgecolor='gray')

    ax.set_xlim(min(p[0] for p in pos_dict.values()) - 1.1,
                max(p[0] for p in pos_dict.values()) + 1.1)
    ax.set_ylim(min(p[1] for p in pos_dict.values()) - 1.2,
                max(p[1] for p in pos_dict.values()) + 1.3)
    ax.axis('off')
    plt.tight_layout()

    img_filename = f"{pathway_name.replace(' ', '_').replace('/', '_')}.png"
    img_path = os.path.join(TEMP_IMAGE_DIR, img_filename)

    try:
        fig.savefig(img_path, bbox_inches='tight', dpi=200, facecolor='white')
    finally:
        plt.close(fig)
        plt.clf()
        gc.collect()

    logger.info(f"✅ 生成代谢图: {img_path} ({pathway_name})")
    return img_path


def get_image_tag(question):
    """检测问题中的代谢途径关键词"""
    keyword_map = {
        "三羧酸循环": ["三羧酸循环", "TCA循环", "TCA", "柠檬酸循环", "Krebs循环"],
        "糖酵解": ["糖酵解", "糖解", "EMP途径"],
        "糖异生": ["糖异生", "糖质新生", "gluconeogenesis"],
        "尿素循环": ["尿素循环", "鸟氨酸循环", "urea cycle"],
        "脂肪酸β氧化": ["脂肪酸氧化", "β氧化", "β-氧化", "脂酰CoA脱氢"],
        "磷酸戊糖途径": ["磷酸戊糖", "PPP", "戊糖磷酸", "NADPH合成"],
        "胆固醇合成": ["胆固醇合成", "胆固醇代谢", "甲羟戊酸途径", "HMG-CoA还原酶"],
        "糖原": ["糖原合成", "糖原分解", "糖原代谢", "糖原贮积"],
    }
    for pathway, keywords in keyword_map.items():
        if any(kw in question for kw in keywords):
            img_path = draw_metabolic_pathway(pathway)
            if img_path:
                return f"[IMAGE:{img_path}]"
    return ""

# ==================== 历史管理 ====================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ==================== 语义缓存 ====================
class SemanticCache:
    def __init__(self, threshold=0.92, max_size=30):
        self.cache = []
        self.threshold = threshold
        self.max_size = max_size
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_bge_model()
        return self._encoder

    def get(self, question):
        if not self.cache:
            return None, None
        try:
            emb = self._get_encoder().embed_query(question)
            emb = emb.reshape(1, -1)
            for cached_q, cached_ans, cached_img, cached_emb in self.cache:
                sim = cosine_similarity(emb, cached_emb.reshape(1, -1))[0][0]
                if sim > self.threshold:
                    logger.info(f"✅ 语义缓存命中: {question[:20]}... (相似度: {sim:.3f})")
                    return cached_ans, cached_img
        except Exception as e:
            logger.debug(f"语义缓存查询失败: {e}")
        return None, None

    def put(self, question, answer, image_path=None):
        try:
            emb = self._get_encoder().embed_query(question)
            self.cache.append((question, answer, image_path, emb))
            if len(self.cache) > self.max_size:
                self.cache.pop(0)
        except Exception as e:
            logger.debug(f"语义缓存保存失败: {e}")

_semantic_cache = SemanticCache()

# ==================== 精确缓存（支持持久化） ====================
_answer_cache = {}
CACHE_MAX_SIZE = 128
CACHE_TTL = 3600
CACHE_FILE = os.path.join(BASE_DIR, "answer_cache.json")

def _load_cache_file():
    """启动时从磁盘加载缓存（高频问题重启后秒回）"""
    global _answer_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            _answer_cache = {
                k: v for k, v in data.items()
                if now - v.get("timestamp", 0) < CACHE_TTL
            }
            if _answer_cache:
                logger.info(f"📂 已加载持久化缓存 {len(_answer_cache)} 条")
    except Exception as e:
        logger.debug(f"加载缓存文件失败: {e}")

def _save_cache_file():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_answer_cache, f, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"保存缓存文件失败: {e}")

_load_cache_file()

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode('utf-8')).hexdigest()

def get_cached_answer(question: str):
    key = get_cache_key(question)
    if key in _answer_cache:
        entry = _answer_cache[key]
        if time.time() - entry.get("timestamp", 0) < CACHE_TTL:
            logger.info(f"✅ 精确缓存命中: {question[:30]}...")
            return entry.get("answer"), entry.get("image_path")
        else:
            del _answer_cache[key]
    return None, None

def save_cached_answer(question: str, answer: str, image_path: str = None):
    key = get_cache_key(question)
    if len(_answer_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_answer_cache.keys(), key=lambda k: _answer_cache[k].get("timestamp", 0))
        del _answer_cache[oldest_key]
    _answer_cache[key] = {"answer": answer, "image_path": image_path, "timestamp": time.time()}
    _save_cache_file()
    logger.info(f"💾 已缓存答案: {question[:30]}...")

def clear_cache():
    global _answer_cache
    _answer_cache.clear()
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
    except Exception:
        pass
    logger.info("🗑️ 已清空所有缓存")

# ==================== 知识点历史（复习串联） ====================
KNOW_HISTORY_FILE = os.path.join(BASE_DIR, "knowledge_history.json")
_know_history = []
_know_emb_cache = {}   # topic -> embedding


def _load_know_history():
    global _know_history
    try:
        if os.path.exists(KNOW_HISTORY_FILE):
            with open(KNOW_HISTORY_FILE, "r", encoding="utf-8") as f:
                _know_history = json.load(f)
    except Exception:
        _know_history = []


def _save_know_history():
    try:
        with open(KNOW_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_know_history, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _topic_embed(topic):
    if topic in _know_emb_cache:
        return _know_emb_cache[topic]
    try:
        e = get_bge_model().embed_query(topic).reshape(1, -1)
        _know_emb_cache[topic] = e
        return e
    except Exception:
        return None


def _record_knowledge(question):
    """把问题记入知识点历史（语义聚类：相似则并入主题）"""
    try:
        q_emb = get_bge_model().embed_query(question).reshape(1, -1)
        best, best_sim = None, 0.62   # 归入阈值
        for h in _know_history:
            t_emb = _topic_embed(h.get("topic", ""))
            if t_emb is None:
                continue
            sim = float(cosine_similarity(q_emb, t_emb)[0][0])
            if sim > best_sim:
                best, best_sim = h, sim
        if best:
            qs = best.setdefault("questions", [])
            if question not in qs:
                qs.append(question)
            best["questions"] = qs[-8:]
        else:
            _know_history.append({
                "topic": question[:30],
                "questions": [question],
                "ts": time.time(),
            })
        _save_know_history()
    except Exception as e:
        logger.debug(f"记录知识点历史失败: {e}")


def _find_related(question, top=3):
    """查找与当前问题相关（但非重复）的历史主题，用于复习串联"""
    try:
        q_emb = get_bge_model().embed_query(question).reshape(1, -1)
        scored = []
        for h in _know_history:
            t_emb = _topic_embed(h.get("topic", ""))
            if t_emb is None:
                continue
            sim = float(cosine_similarity(q_emb, t_emb)[0][0])
            # 0.45~0.88：相关但不算同一问题（避免对刚问过的重复提示）
            if 0.45 <= sim <= 0.88:
                scored.append((sim, h))
        scored.sort(key=lambda x: -x[0])
        return [(round(s, 2), h) for s, h in scored[:top]]
    except Exception:
        return []


def build_review_context(question):
    """生成复习串联上下文文本"""
    related = _find_related(question)
    if not related:
        return ""
    parts = []
    for sim, h in related:
        qs = "；".join(h.get("questions", [])[-3:])
        parts.append(f"[{h.get('topic', '')}](相关度{sim}) 你之前问过: {qs}")
    return "\n".join(parts)


_load_know_history()

# ==================== 复习填空题收集（考研六大板块分类 · docx 输出） ====================
REVIEW_DATA_FILE = os.path.join(BASE_DIR, "复习填空题_data.json")
REVIEW_DOCX_FILE = os.path.join(BASE_DIR, "复习填空题.docx")
REVIEW_SECTIONS = ["生理", "生化", "病理", "内科", "外科", "医学人文素养"]

# 学科分类关键词（考研模式）
_SUBJECT_RULES = [
    ("生理", ["静息电位", "动作电位", "心动周期", "血压调节", "血压形成", "动脉血压", "收缩压",
              "舒张压", "肾小球", "肾小管", "肺通气", "呼吸膜", "呼吸调节", "气体运输",
              "胃酸", "突触", "激素调节", "血糖调节", "血型", "离子", "跨膜", "心输出量",
              "循环调节", "能量代谢", "体温调节", "神经调节", "兴奋", "肌肉收缩",
              "内环境", "稳态",
              # 血液生理（凝血/止血机制归生理）
              "凝血", "凝血因子", "血液凝固", "血小板", "纤溶", "止血", "血细胞",
              "血红蛋白", "造血", "生理性止血", "内源性凝血", "外源性凝血",
              # 细胞物质转运（跨膜转运）
              "物质转运", "跨膜转运", "单纯扩散", "易化扩散", "主动转运", "原发性主动转运",
              "继发性主动转运", "钠泵", "钠钾泵", "胞吞", "胞吐", "载体转运", "通道转运",
              # 生物电 / 电活动
              "生物电", "膜电位", "阈电位", "阈刺激", "局部电位", "去极化", "复极化",
              "超极化", "不应期", "钠通道", "钾通道", "离子通道", "兴奋性", "全或无",
              "电紧张", "反极化", "锋电位",
              # 肌肉收缩（等长/等张收缩等）
              "等长收缩", "等张收缩", "前负荷", "后负荷", "兴奋收缩耦联", "兴奋-收缩耦联",
              "肌丝滑行", "横桥", "强直收缩", "单收缩", "初长度", "肌张力",
              # 神经活动（全部归生理）
              "反射", "反射弧", "神经递质", "突触传递", "突触后电位", "突触前抑制",
              "神经-肌接头", "终板电位", "兴奋性突触", "抑制性突触", "传入神经",
              "传出神经", "自主神经", "交感神经", "副交感神经", "迷走神经", "感受器",
              "感觉", "痛觉", "视觉", "听觉", "前庭", "嗅觉", "味觉", "牵张反射",
              "肌梭", "腱器官", "运动单位", "条件反射", "脑电", "觉醒", "睡眠周期",
              "睡眠时相", "学习记忆", "乙酰胆碱", "去甲肾上腺素", "多巴胺", "5-羟色胺",
              "γ-氨基丁酸", "谷氨酸", "神经活动", "突触间隙", "神经冲动",
              # 呼吸生理专有名词（肺通气/呼吸功能指标）
              "潮气量", "肺活量", "功能残气量", "残气量", "肺泡通气量", "无效腔",
              "肺顺应性", "表面活性物质", "胸内压", "气道阻力", "氧解离曲线",
              "肺牵张反射", "呼吸中枢", "化学感受器", "呼吸频率", "呼吸深度",
              # 女性激素（生殖内分泌归生理）
              "雌激素", "孕激素", "雌二醇", "孕酮", "卵泡刺激素", "黄体生成素",
              "促性腺激素", "卵泡期", "黄体期", "月经周期", "排卵", "卵巢周期",
              "绝经", "更年期", "催乳素", "HCG", "人绒毛膜促性腺激素",
              # 胰岛内分泌（保护生理：胰岛素分泌调节归生理）
              "胰岛素分泌", "胰岛素的分泌", "胰岛素释放", "胰岛素的释放",
              "胰岛素的作用", "胰岛素的合成", "胰岛素的调节", "胰岛素对血糖",
              "胰岛β细胞", "胰岛α细胞", "胰高血糖素"]),
    ("生化", ["糖代谢", "脂代谢", "氨基酸代谢", "物质代谢", "代谢途径", "代谢紊乱",
              "代谢过程", "代谢产物", "代谢物", "中间代谢", "代谢调节", "代谢障碍",
              "糖酵解", "三羧酸", "柠檬酸", "氧化磷酸化", "呼吸链", "电子传递链",
              "电子传递", "ATP合酶", "细胞色素", "泛醌", "酶促", "蛋白质",
              "核酸", "DNA", "RNA", "转录", "翻译", "脂肪酸", "胆固醇", "氨基酸",
              "尿素", "磷酸戊糖", "糖原", "生物化学", "ATP", "NADH", "生化",
              # 基因相关（细化长词，避免裸"基因"抢走病理的癌基因）
              "基因表达", "基因突变", "基因工程", "基因治疗", "基因诊断", "基因重组",
              "基因扩增", "基因组", "基因型", "基因调控", "基因多态性",
              # 维生素/辅酶（生化营养代谢）
              "维生素", "叶酸", "辅酶", "辅基", "核黄素", "硫胺素", "烟酸", "泛酸",
              "钴胺素", "生物素", "生物氧化", "一碳单位", "酮体", "脂蛋白", "糖异生",
              "DNA复制", "逆转录", "PCR", "密码子", "米氏常数", "同工酶", "变构调节"]),
    # 外科放在病理之前：消化/呼吸系统肿瘤等具体肿瘤归外科，基础病理词（炎症/坏死等）仍归病理
    ("外科", ["手术", "骨折", "创伤", "烧伤", "麻醉", "休克", "阑尾", "疝", "胆道", "乳腺",
              "食管", "肠梗阻", "外科", "无菌", "输血", "围术期", "脱水", "电解质",
              "气胸", "血胸", "感染伤口",
              # 消化系统肿瘤（外科学重点）
              "胃癌", "胃肿瘤", "肝癌", "肝肿瘤", "结肠癌", "结直肠癌", "直肠癌", "大肠癌",
              "胰腺癌", "胆囊癌", "胆管癌", "食管癌", "胃肠道间质瘤", "胃间质瘤",
              # 呼吸系统肿瘤（胸外科重点）
              "肺癌", "肺肿瘤", "支气管肺癌", "肺结节",
              # 甲状腺肿瘤/囊肿/结节（外科）；甲亢甲减等内分泌疾病归内科
              "甲状腺癌", "甲状腺腺瘤", "甲状腺肿瘤", "甲状腺囊肿", "甲状腺结节",
              "甲状腺肿", "甲状腺次全切除", "甲状腺手术",
              # 脑部疾病（神经外科）
              "脑肿瘤", "脑膜瘤", "胶质瘤", "垂体瘤", "听神经瘤", "脑出血", "颅脑损伤",
              "脑外伤", "硬膜外血肿", "硬膜下血肿", "脑挫裂伤", "颅内压增高", "脑疝",
              "脑积水", "脑脓肿", "颅内动脉瘤", "脑血管畸形", "脊髓肿瘤", "椎管内肿瘤",
              "开颅", "颅底",
              # 围手术期处理（术前准备/术后处理）
              "术前准备", "术后处理", "术前", "术后", "切口", "引流管", "拆线",
              "禁食水", "伤口愈合", "备皮",
              # 病生部分（外科总论：水电解质/酸碱平衡/休克/感染/多器官）
              "酸碱平衡", "代谢性酸中毒", "呼吸性酸中毒", "代谢性碱中毒", "呼吸性碱中毒",
              "低钾血症", "高钾血症", "低钠血症", "高钠血症", "低钙血症",
              "MODS", "多器官功能", "SIRS", "全身炎症反应", "脓毒症", "败血症", "菌血症",
              "感染性休克", "低血容量性休克", "失血性休克", "过敏性休克", "创伤性休克",
              "挤压综合征", "脂肪栓塞", "筋膜间隔综合征", "破伤风", "气性坏疽",
              # 其他外科肿瘤
              "肾癌", "膀胱癌", "前列腺癌", "乳腺癌", "骨肿瘤", "骨肉瘤",
              "软组织肉瘤", "神经母细胞瘤", "肾母细胞瘤", "畸胎瘤"]),
    ("病理", ["炎症", "肿瘤", "癌", "肉瘤", "坏死", "凋亡", "血栓", "栓塞", "梗死", "水肿",
              "肉芽肿", "损伤", "化生", "增生", "萎缩", "变性", "转移", "充血", "淤血",
              "病理", "病变", "纤维化", "异型性", "原位癌", "浸润", "癌前病变",
              # 细胞损伤与适应
              "玻璃样变", "脂肪变性", "凝固性坏死", "液化性坏死", "坏疽", "纤维素样坏死",
              "肥大", "再生", "瘢痕", "肉芽组织", "机化",
              # 局部血液循环障碍（裸"出血"不加，防止"消化道出血"误伤）
              "侧支循环", "漏出性出血", "破裂性出血",
              # 炎症
              "渗出", "变质", "炎症介质", "白细胞渗出", "化脓性", "浆液性", "纤维素性", "假膜",
              # 肿瘤相关（病理专章）
              "癌基因", "抑癌基因", "分化程度", "分级", "分期", "良恶性",
              # 传染病 / 免疫病理（结核形态词归病理；结核疾病与抗结核治疗归内科）
              "结核结节", "结核瘤", "干酪样坏死", "粟粒性结核",
              "伤寒", "痢疾", "梅毒", "麻风", "血吸虫", "包虫",
              "超敏反应", "自身免疫", "移植排斥", "免疫缺陷",
              # 动脉粥样硬化（病理专章；冠心病仍归内科）
              "动脉粥样硬化"]),
    ("内科", ["肺炎", "心衰", "高血压", "糖尿病", "冠心病", "肾病", "肝硬化", "白血病", "哮喘",
              "COPD", "慢阻肺", "胃炎", "溃疡", "胰腺炎", "贫血", "肾衰",
              "感染", "发热", "心律失常", "瓣膜", "风湿", "狼疮", "内科", "内分泌疾病",
              # 心肌疾病词（细化，避免裸"心肌"抢走生理的心肌电活动题）
              "心肌炎", "心肌病", "心肌梗死", "心肌肥厚", "心肌缺血",
              # 甲状腺内分泌疾病（甲亢/甲减等归内科；肿瘤/囊肿归外科）
              "甲亢", "甲减", "甲状腺功能亢进", "甲状腺功能减退", "甲状腺炎",
              "Graves", "格雷夫斯", "甲状腺危象",
              # 病毒性肝炎归内科（病理词不加肝硬化/肝炎，避免抢）
              "病毒性肝炎", "乙型肝炎", "丙型肝炎", "乙肝", "丙肝", "肝炎",
              # 结核疾病与治疗归内科（结核形态词归病理）
              "结核", "肺结核", "结核病", "抗结核", "肠结核", "骨结核", "结核性",
              # 药理学知识（归内科）
              "药理学", "药效学", "药动学", "药代动力学", "药物代谢", "血药浓度",
              "生物利用度", "首过消除", "肝药酶", "药物相互作用", "不良反应", "副作用",
              "毒性反应", "停药反应", "耐受性", "依赖性", "耐药性", "抗菌谱", "抗生素",
              "抗菌药物", "抗菌药", "抗真菌", "抗病毒药", "化疗", "抗肿瘤药",
              "受体激动剂", "受体拮抗剂", "受体阻滞剂", "激动剂", "拮抗剂", "半衰期",
              "给药途径", "量效关系", "治疗窗", "糖皮质激素", "他汀", "贝特类",
              "硝酸酯", "硝酸甘油", "强心苷", "地高辛", "肝素", "华法林", "氯吡格雷",
              "阿司匹林", "青霉素", "头孢", "氨基糖苷", "大环内酯", "喹诺酮", "磺胺",
              "甲硝唑", "抗凝药", "抗血小板药", "降糖药", "降压药", "降脂药", "平喘药",
              "利尿剂", "利尿药", "抑酸药", "质子泵抑制剂", "H2受体拮抗剂", "镇痛药",
              "解热镇痛", "镇静催眠", "抗癫痫", "抗精神病药", "抗抑郁药", "免疫抑制剂",
              "抗疟药", "驱虫药", "磺脲", "胰岛素抵抗", "胰岛素治疗",
              "胰岛素的不良反应", "胰岛素的用法"]),
    ("医学人文素养", ["伦理", "法律", "医患", "沟通", "心理", "人文", "卫生法", "导论", "道德",
                       "知情同意", "隐私", "素养", "医德", "职业精神", "关怀", "素养"]),
]


def classify_subjects(question: str) -> list:
    """多标签分类：返回所有命中的学科（含学科间有争议的问题会同时命中多个）
    无命中时默认 ["内科"]；命中多个时按规则顺序返回，如 ["内科", "外科"]"""
    q = question or ""
    hits = []
    for name, kws in _SUBJECT_RULES:
        if any(k in q for k in kws) and name not in hits:
            hits.append(name)
    return hits or ["内科"]


def classify_subject(question: str) -> str:
    """主学科分类（兼容旧接口）：多标签时取第一个命中"""
    return classify_subjects(question)[0]


def _parse_fills(answer):
    """解析三档填空题 -> (fills_dict, answers_dict)"""
    fills, answers = {}, {}
    if not answer:
        return fills, answers
    blocks = re.split(r"【(基础填空|进阶填空|挑战填空)】", answer)
    for i in range(1, len(blocks) - 1, 2):
        lvl = blocks[i]
        body = blocks[i + 1]
        m = re.search(r"参考答案[:：]\s*(.*)", body, re.S)
        q_text = body[: m.start()].strip() if m else body.strip()
        a_text = m.group(1).strip() if m else ""
        fills[lvl] = q_text
        answers[lvl] = a_text
    return fills, answers


def _load_review_data():
    """加载复习数据：返回 questions 列表（新贴标签格式）
    兼容旧格式（{"内科": [...]} 分组结构）"""
    try:
        if os.path.exists(REVIEW_DATA_FILE):
            with open(REVIEW_DATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            if isinstance(data, dict):   # 旧格式：按板块分组的 dict
                items = []
                for arr in data.values():
                    items.extend(arr)
                return items
    except Exception:
        pass
    return []


def _save_review_data(items):
    """以贴标签格式保存：扁平 questions 列表（version=2）"""
    with open(REVIEW_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "questions": items}, f, ensure_ascii=False, indent=1)


def _subject_tag(it):
    """多学科标记：条目命中多个学科时生成【内科·外科】式标记；单一学科不标"""
    subs = it.get("subjects") or []
    if len(subs) >= 2:
        return "【" + "·".join(subs) + "】"
    return ""


def _render_review_docx(items):
    """从贴标签格式的 questions 列表生成 docx：
    题目区 = 基础/中等/挑战 三大难度区 × 六板块（按 subjects 标签归入所有命中板块，
    多学科题会出现在多个板块并带标签标记）；题号全局唯一，答案区按题号对照。
    字体：楷体（正文 7.5pt，标题保持层级但同为楷体）"""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.oxml.ns import qn
    except Exception:
        logger.warning("python-docx 未安装，无法生成复习 docx")
        return

    def _p(doc, text, size=7.5, bold=False):
        """添加楷体段落（中文字体需同时设置 eastAsia）"""
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "楷体"
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')
        return p

    doc = Document()
    _p(doc, "考研填空题复习册（基础 · 中等 · 挑战）", size=14, bold=True)
    _p(doc, "使用方法：先看题目自测，答案统一在文件最后，按题号对照查找。", size=7.5)

    # ===== 收集所有填空小题（扁平 + 标签） =====
    small = []
    for it in items or []:
        subjects = it.get("subjects") or classify_subjects(it.get("question", ""))
        fills = it.get("fills") or {}
        answers = it.get("answers") or {}
        for section, key in [("基础", "基础填空"), ("中等", "进阶填空"), ("挑战", "挑战填空")]:
            q_text = fills.get(key)
            if q_text:
                small.append({
                    "section": section,
                    "subjects": list(subjects),
                    "question": q_text,
                    "answer": answers.get(key, ""),
                })

    # 排序：基础 → 中等 → 挑战；然后全局编号（题号唯一，答案区对应）
    sec_order = {"基础": 0, "中等": 1, "挑战": 2}
    small.sort(key=lambda x: sec_order.get(x["section"], 9))
    for i, it in enumerate(small, 1):
        it["num"] = i

    # ===== 题目区：三大难度区 × 六板块（按标签归入所有命中板块） =====
    section_titles = {"基础": "一、基础填空（掌握基本概念）",
                      "中等": "二、中等填空（理解机制）",
                      "挑战": "三、挑战填空（临床/机制推理）"}
    for section in ["基础", "中等", "挑战"]:
        sec_items = [it for it in small if it["section"] == section]
        _p(doc, section_titles[section], size=12, bold=True)
        for subj in REVIEW_SECTIONS:
            sub_items = [it for it in sec_items if subj in it["subjects"]]
            if not sub_items:
                continue
            _p(doc, f"{section} · {subj}", size=10, bold=True)
            for it in sub_items:
                tag = _subject_tag(it)
                _p(doc, f"【{it['num']}】{tag}{it['question']}", size=7.5)

    # ===== 答案区（文件最后，按题号查找） =====
    doc.add_page_break()
    _p(doc, "参考答案（按题号对照，请做完后再看！）", size=12, bold=True)
    for it in small:
        ans = it["answer"] or "（未提供）"
        _p(doc, f"{it['num']}. {ans}", size=7.5)

    doc.save(REVIEW_DOCX_FILE)
    logger.info(f"📄 已生成复习 docx: {REVIEW_DOCX_FILE}（楷体 7.5pt，共 {len(small)} 道小题）")


def _save_review_fill(question, answer):
    """把本次填空题以贴标签格式存入扁平列表，并重新生成 docx"""
    try:
        fills, answers = _parse_fills(answer)
        if not fills:
            return
        subjects = classify_subjects(question)
        entry = {
            "question": question,
            "ts": time.strftime("%Y-%m-%d %H:%M"),
            "fills": fills,
            "answers": answers,
            "subjects": subjects,          # 多学科标签（如 ["内科","外科"]）
        }
        items = _load_review_data()
        # 同 question 去重更新（避免重复提问产生重复条目）
        for i, it in enumerate(items):
            if isinstance(it, dict) and it.get("question") == question:
                items[i] = entry
                break
        else:
            items.append(entry)
        _save_review_data(items)
        _render_review_docx(items)
        logger.info(f"📝 已保存填空题到复习文件（标签: {'/'.join(subjects)}）")
    except Exception as e:
        logger.debug(f"保存复习填空题失败: {e}")

# ==================== API 调用 ====================
def call_deepseek(messages, max_tokens=1000):
    if not DEEPSEEK_API_KEY:
        return "请先配置API密钥"

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    try:
        res = retry_request(
            lambda: requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=API_TIMEOUT),
            API_RETRY_TIMES
        )
        if res.status_code != 200:
            logger.error(f"API 请求失败: {res.status_code} - {res.text[:200]}")
            return f"❌ API 请求失败: {res.status_code}"
        data = res.json()
        usage = data.get('usage', {})
        if usage:
            logger.info(f"Token 消耗: 输入 {usage.get('prompt_tokens', 0)}, 输出 {usage.get('completion_tokens', 0)}")
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {traceback.format_exc()}")
        return None

# ==================== 混合 ReAct Pipeline 单例 ====================
_rag_pipeline = None

def get_rag_pipeline(user_id: str = "default"):
    """获取或创建混合 ReAct Pipeline"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = ModularRAGPipeline(llm_callback=call_deepseek, user_id=user_id)
        logger.info("✅ 混合 ReAct Pipeline 已初始化")
    return _rag_pipeline

# ==================== 用户反馈接口 ====================

def record_feedback(question: str, answer: str, feedback_type: str, comment: str = ""):
    try:
        memory = get_memory_manager()
        from rag_modules.router import QuestionRouter
        router = QuestionRouter()
        route_type = router.route(question)

        memory.record_feedback(
            question=question,
            answer=answer,
            feedback=feedback_type,
            comment=comment,
            route_type=route_type,
            sources=["dense", "sparse"]
        )

        stats = memory.get_feedback_stats()
        msg = f"✅ 反馈已记录！累计反馈: 👍{stats.get('thumbs_up',0)} 👎{stats.get('thumbs_down',0)} 📝{stats.get('correction',0)}"
        return True, msg
    except Exception as e:
        logger.error(f"记录反馈失败: {e}")
        return False, f"❌ 反馈记录失败: {e}"


def get_user_memory_summary() -> dict:
    try:
        memory = get_memory_manager()
        pref = memory.preference
        stats = memory.get_feedback_stats()
        return {
            "preferred_length": pref.preferred_length,
            "likes_images": pref.likes_images,
            "top_subject": max(pref.subject_weights.items(), key=lambda x: x[1])[0] if pref.subject_weights else "无",
            "subject_count": len(pref.subject_weights),
            "route_history": dict(pref.route_type_history),
            "feedback_stats": stats,
            "adaptive_weights": memory.get_adaptive_weights("general")
        }
    except Exception as e:
        logger.warning(f"获取记忆摘要失败: {e}")
        return {}


def get_security_stats() -> dict:
    try:
        guard = SecurityGuard()
        return guard.get_stats()
    except Exception as e:
        logger.warning(f"获取安全统计失败: {e}")
        return {"total_events": 0, "blocked": 0, "warnings": 0}


# ==================== 流式问答（混合 ReAct 主入口） ====================
def med_exam_chat_stream(question, session_history, brief=False):
    logger.info(f"提问: {question}")

    try:
        # 1. 精确缓存
        cached_answer, cached_image = get_cached_answer(question)
        if cached_answer is not None:
            if cached_image and os.path.exists(cached_image):
                yield cached_answer + "\n\n" + f"[IMAGE:{cached_image}]"
            else:
                yield cached_answer
            return

        # 2. 语义缓存
        semantic_ans, semantic_img = _semantic_cache.get(question)
        if semantic_ans is not None:
            if semantic_img and os.path.exists(semantic_img):
                yield semantic_ans + "\n\n" + f"[IMAGE:{semantic_img}]"
            else:
                yield semantic_ans
            return

        # 3. 混合 ReAct Pipeline（带复习串联上下文 + 多学科标签）
        review_ctx = build_review_context(question)
        pipeline = get_rag_pipeline()
        subject_tags = classify_subjects(question)
        result = pipeline.run(question, session_history, review_context=review_ctx,
                              brief=brief, subject_tags=subject_tags)

        # 记录本次知识点到历史（复习用）
        try:
            _record_knowledge(question)
        except Exception:
            pass

        # 安全拦截处理
        if result.blocked:
            sec = result.security.get("input_guard")
            if sec:
                yield f"🛡️ [安全系统拦截]\n"
                yield f"风险等级: {sec.risk_level}\n"
                yield f"原因: {sec.reason}\n\n"
            else:
                yield f"🛡️ [安全系统拦截]\n\n"
            yield result.answer
            return

        # ⭐ 显示运行模式
        if result.used_react:
            yield f"🤖 [Agent 模式] 复杂问题多步推理\n"
            yield f"📊 LLM 调用: {result.llm_calls} 次\n"
            # 显示 ReAct 轨迹
            if result.agent_steps:
                yield f"\n🧠 推理轨迹：\n"
                for step in result.agent_steps:
                    yield f"  Step {step['step']}: [{step['action']}] {step['action_input'][:30]}...\n"
                    yield f"    💭 {step['thought'][:60]}...\n"
            yield "\n"
        else:
            yield f"🚀 [快速模式] 直接检索生成\n\n"

        # 若结合上下文理解了问题，向用户展示；多学科时标注学科视角
        effective_q = getattr(result, "effective_question", "") or ""
        q_header = f"【问题】{question}"
        if len(subject_tags) >= 2:
            q_header += f"\n📚 学科视角：{'、'.join(subject_tags)}（各科标准可能不同，已分别标注）"
        if effective_q and effective_q != question:
            q_header += f"\n🔁 结合上下文，我理解为：{effective_q}"
        yield f"{'='*40}\n{q_header}\n【回答】\n"
        # 组装最终答案
        answer = result.answer

        # 代谢图检测
        img_tag = get_image_tag(question)
        if img_tag:
            answer = answer + "\n\n" + img_tag

        # 低置信度提示
        if result.validation.get("confidence") == "low":
            answer += "\n\n⚠️ [系统提示] 当前参考资料置信度较低，答案仅供参考，建议核实教材。"

        # 缓存
        clean_answer = answer.replace(img_tag, "").strip() if img_tag else answer
        image_path = None
        if img_tag:
            match = re.search(r'\[IMAGE:(.*?)\]', img_tag)
            if match and os.path.exists(match.group(1)):
                image_path = match.group(1)
        save_cached_answer(question, clean_answer, image_path)
        _semantic_cache.put(question, answer, image_path)

        # 记忆更新
        try:
            memory = get_memory_manager()
            memory.update_from_question(
                question=question,
                route_type=result.route_type,
                used_image=bool(img_tag)
            )
        except Exception as e:
            logger.debug(f"记忆更新失败: {e}")

        # 📝 把填空题保存到复习文件
        try:
            _save_review_fill(question, answer)
        except Exception as e:
            logger.debug(f"保存填空题失败: {e}")

        yield answer

    except Exception as e:
        logger.error(f"问答异常: {traceback.format_exc()}")
        yield f"⚠️ 系统错误: {e}"

# ==================== 非流式版本 ====================
def med_exam_chat(question, session_history):
    full_answer = ""
    for chunk in med_exam_chat_stream(question, session_history):
        full_answer += chunk
    return full_answer

if __name__ == "__main__":
    pass
