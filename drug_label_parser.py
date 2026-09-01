#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
药物说明书解析（重建版）
- 读取 drug_labels 目录下的 PDF 说明书
- 提取纯文本并按医学章节分块（适应症/禁忌/用法用量/不良反应等）
提供接口：
    parse_drug_label_dir(dir)          -> List[Document]（含 source_type=drug_label）
    parse_drug_label_file(path)        -> List[Document]（单文件按章节分块）
    split_drug_label(text, source)     -> List[Document]（按章节关键字分块）
"""

import os
import re
from langchain_core.documents import Document

# 说明书章节关键字（用于按章节分块）
SECTION_KEYWORDS = [
    "药品名称", "成份", "性状", "适应症", "功能主治", "规格",
    "用法用量", "不良反应", "禁忌", "注意事项", "孕妇及哺乳期妇女用药",
    "儿童用药", "老年用药", "药物相互作用", "药物过量", "药理毒理",
    "药代动力学", "贮藏", "包装", "有效期", "执行标准", "批准文号",
    "生产企业", "说明书修订日期",
]


def _extract_pdf_text(pdf_path):
    """用 PyMuPDF 提取 PDF 文本"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        parts = [page.get_text() for page in doc if page.get_text()]
        doc.close()
        return "\n".join(parts)
    except Exception:
        try:
            from pymupdf import open as fitz_open
            doc = fitz_open(pdf_path)
            parts = [page.get_text() for page in doc if page.get_text()]
            doc.close()
            return "\n".join(parts)
        except Exception:
            return ""


def split_drug_label(text, source="unknown"):
    """按章节关键字把说明书文本分块"""
    if not text or not text.strip():
        return []

    # 构建章节标题匹配：行首或独立行的 "【适应症】/适应症：" 等
    pattern = re.compile(
        r"(?:^|\n)\s*[【\[]?\s*(" + "|".join(SECTION_KEYWORDS) + r")\s*[】\]]?\s*[:：]?",
        re.MULTILINE
    )
    matches = list(pattern.finditer(text))
    chunks = []

    if not matches:
        # 无章节标题，整体作为一个块（截断超长）
        body = text.strip()
        if len(body) > 4000:
            body = body[:4000]
        chunks.append(Document(
            page_content=body,
            metadata={"source": source, "source_type": "drug_label"}
        ))
        return chunks

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = m.group(1).strip()
        content = text[start:end].strip()
        # 清理章节标题行本身
        content = re.sub(r"^.*?[】\]]?\s*[:：]?\s*", "", content, count=1)
        if len(content) < 10:
            continue
        if len(content) > 2000:
            content = content[:2000]
        chunks.append(Document(
            page_content=f"【{section}】\n{content}",
            metadata={"source": source, "source_type": "drug_label"}
        ))

    return chunks


def parse_drug_label_file(pdf_path):
    """解析单个说明书 PDF 为按章节分块的 Document 列表"""
    text = _extract_pdf_text(pdf_path)
    if not text.strip():
        return []
    return split_drug_label(text, source=pdf_path)


def parse_drug_label_dir(dir_path):
    """解析目录下全部说明书 PDF，返回 Document 列表"""
    docs = []
    if not os.path.exists(dir_path):
        return docs
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(dir_path, fname)
        try:
            text = _extract_pdf_text(fpath)
            if text.strip():
                docs.extend(split_drug_label(text, source=fpath))
        except Exception:
            continue
    return docs


if __name__ == "__main__":
    import sys
    from config import DRUG_LABEL_DIR
    if os.path.exists(DRUG_LABEL_DIR):
        ds = parse_drug_label_dir(DRUG_LABEL_DIR)
        print(f"说明书分块数: {len(ds)}")
        for d in ds[:5]:
            print(" -", d.metadata.get("source"), "|", d.page_content[:40])
    else:
        print("说明书目录不存在")
