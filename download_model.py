#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 BGE-small-zh 语义模型到本地（重建版）
用法: python download_model.py
模型将保存到 models/bge-small-zh，供 utils.BGEEmbeddings 使用。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "BAAI/bge-small-zh"
TARGET_DIR = os.path.join(BASE_DIR, "models", "bge-small-zh")


def main():
    os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
    print(f"正在下载模型 {MODEL_NAME} -> {TARGET_DIR} ...")

    def p(msg):
        # 避免 emoji 在 GBK 控制台报错
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("gbk", "ignore").decode("gbk"))

    try:
        from sentence_transformers import SentenceTransformer
        # 方式一：通过 sentence-transformers 下载并保存
        model = SentenceTransformer(MODEL_NAME)
        model.save(TARGET_DIR)
        p("模型已保存到 " + TARGET_DIR)
    except Exception as e:
        print(f"sentence-transformers 方式失败: {e}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=TARGET_DIR,
            )
            p("模型已通过 huggingface_hub 下载到 " + TARGET_DIR)
        except Exception as e2:
            print(f"下载失败: {e2}")
            print("请手动下载 BAAI/bge-small-zh 并放到 models/bge-small-zh")
            sys.exit(1)

    # 提示 ONNX 导出（可选加速）
    print("提示：如需 ONNX 加速，可运行 export 脚本将模型导出为 model.onnx")


if __name__ == "__main__":
    main()
