#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
题库更新模块（重建版）
从源文件（.doc/.docx/.pdf）重新生成 CSV，并重新加载到内存。
提供接口：
    update_all() -> (doc_cnt, pdf_cnt, err_msg)
生成文件：题库.docx.csv（doc 源）与 题库.pdf.csv（pdf 源）
"""

import os
import re
import csv
import glob
import time
from pathlib import Path

from utils import logger

# ===== 导入转换所需模块 =====
try:
    import comtypes.client
except ImportError:
    comtypes = None
    logger.warning("comtypes 未安装，无法转换 .doc 文件")

try:
    from docx import Document
except ImportError:
    Document = None
    logger.warning("python-docx 未安装，无法读取 .docx 文件")

try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        from pymupdf import open as fitz_open
        fitz = None
    except ImportError:
        fitz = None
        logger.warning("PyMuPDF 未安装，无法读取 PDF 文件")

# ===== 配置路径 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_SOURCE_DIR = os.path.join(BASE_DIR, "题库doc")
DOC_OUTPUT_DIR = os.path.join(DOC_SOURCE_DIR, "docx")  # .docx 输出目录
DOC_CSV = os.path.join(BASE_DIR, "题库.docx.csv")
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "题库pdf")
PDF_CSV = os.path.join(BASE_DIR, "题库.pdf.csv")

HEADERS = ["question", "options", "answer", "analysis", "know_point"]


# ===== 辅助函数 =====
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_text(t):
    return re.sub(r"\s+", " ", t or "").strip()


# ===== 1. .doc → .docx 转换 =====
def convert_doc_to_docx():
    """使用 comtypes 将 .doc 转为 .docx"""
    if comtypes is None:
        raise RuntimeError("comtypes 未安装，无法转换 .doc 文件")
    if not os.path.exists(DOC_SOURCE_DIR):
        logger.warning(f"源目录不存在：{DOC_SOURCE_DIR}")
        return 0

    doc_files = glob.glob(os.path.join(DOC_SOURCE_DIR, "*.doc"))
    if not doc_files:
        logger.info("未找到 .doc 文件，跳过")
        return 0

    ensure_dir(DOC_OUTPUT_DIR)
    success_count = 0
    for doc_path in doc_files:
        base_name = os.path.basename(doc_path)
        docx_name = os.path.splitext(base_name)[0] + ".docx"
        docx_path = os.path.join(DOC_OUTPUT_DIR, docx_name)
        if os.path.exists(docx_path):
            logger.info(f"跳过已存在的 {docx_name}")
            continue
        try:
            word = comtypes.client.CreateObject("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(doc_path)
            doc.SaveAs2(docx_path, FileFormat=16)
            doc.Close()
            word.Quit()
            logger.info(f"✅ 转换成功：{base_name}")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ 转换失败 {base_name}: {e}")
    return success_count


# ===== 2. .docx → CSV =====
def parse_question_block(block):
    """解析一个题目块为 [question, options, answer, analysis, know_point]"""
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if not lines:
        return None
    first_line = lines[0]
    q_num_match = re.match(r'^(\d+)[.、)）]\s*', first_line)
    if not q_num_match:
        return None
    q_text = first_line[q_num_match.end():].strip()
    if not q_text and len(lines) > 1:
        q_text = lines[1].strip()
        lines = lines[2:]
    else:
        lines = lines[1:]

    options = {}
    option_pattern = re.compile(r'^([A-E])[.、)）]\s*(.*)')
    rest_lines = []
    for line in lines:
        opt_match = option_pattern.match(line)
        if opt_match:
            options[opt_match.group(1)] = opt_match.group(2).strip()
        else:
            rest_lines.append(line)

    # 从剩余行提取答案/解析/知识点
    answer = ""
    analysis = ""
    know_point = ""
    for line in rest_lines:
        if re.match(r'^(答\s*案|参考答案|正确答案)\s*[:：]?\s*[A-Ea-e]', line):
            m = re.match(r'^(答\s*案|参考答案|正确答案)\s*[:：]?\s*([A-Ea-e])', line)
            if m:
                answer = m.group(2).upper()
            continue
        if re.match(r'^(解\s*析|答案解\s*析|注释)', line):
            analysis = clean_text(line)
            continue
        if re.match(r'^(知识\s*点|考点)', line):
            know_point = clean_text(line)
            continue
        if line and not analysis:
            analysis = clean_text(line)

    option_text = "；".join(f"{k}.{v}" for k, v in sorted(options.items()))
    return [q_text, option_text, answer, analysis, know_point]


def convert_docx_to_csv():
    """从 .docx 提取题目生成 CSV"""
    if Document is None:
        raise RuntimeError("python-docx 未安装")
    if not os.path.exists(DOC_OUTPUT_DIR):
        logger.warning(f".docx 目录不存在：{DOC_OUTPUT_DIR}")
        return 0

    rows = [HEADERS]
    total = 0
    for fname in sorted(os.listdir(DOC_OUTPUT_DIR)):
        if not fname.endswith(".docx"):
            continue
        fpath = os.path.join(DOC_OUTPUT_DIR, fname)
        try:
            doc = Document(fpath)
            text_blocks = []
            cur = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if not t:
                    if cur:
                        text_blocks.append("\n".join(cur))
                        cur = []
                    continue
                cur.append(t)
            if cur:
                text_blocks.append("\n".join(cur))

            for block in text_blocks:
                item = parse_question_block(block)
                if item:
                    rows.append(item)
                    total += 1
        except Exception as e:
            logger.error(f"❌ 解析失败 {fname}: {e}")

    with open(DOC_CSV, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    return total


# ===== 3. 真题 PDF → CSV（考研西医综合真题专用解析器） =====
# 支持三种真题版式：
#   ① 老格式(1988-2022)：题号 "1. 题干 A.xx B.xx C.xx D.xx"（选项同行，无答案）
#   ② 2024 格式：题号 "2024N1A."，选项独立行，答案 "答案：X"
#   ③ 2025 格式：题号 "1、"，选项独立行，答案 "参考答案：X"
# 扫描版 PDF（无文本层，如 2019/2020/2023）自动跳过并统计。
_EXAM_Q_RE = re.compile(r'^\s*(\d{1,3})[.、)）]\s*(.*)$')          # ①② 通用题号
_EXAM_NEW_RE = re.compile(r'\d{4}N\d+[ABX]?[.、]')                  # 2024N1A.（不锚定开头，兼容 BOM/前导字符）
_EXAM_OPT_RE = re.compile(r'^([A-D])[.、)）]\s*(.*)$')               # 独立行选项
_EXAM_ANS_RE = re.compile(r'^(?:答案|参考答案|正确答案)\s*[:：]?\s*([A-Ea-e])\s*$')
_EXAM_INLINE_OPT_RE = re.compile(r'\s+([A-D])[.、)）]\s*')           # 同行选项分割


def _split_options_block(block_lines):
    """独立行选项：['题干','A.xx','B.xx','答案：C'] -> (question, options_dict, answer)"""
    question, options, answer = "", {}, ""
    for line in block_lines:
        line = line.strip()
        if not line:
            continue
        om = _EXAM_OPT_RE.match(line)
        if om:
            options[om.group(1)] = om.group(2).strip()
            continue
        am = _EXAM_ANS_RE.match(line)
        if am:
            answer = am.group(1).upper()
            continue
        if not question:
            question = line
    return question, options, answer


def _split_options_inline(text):
    """同行选项：'题干 A.xx B.xx C.xx D.xx' -> (question, options_dict)"""
    m = _EXAM_INLINE_OPT_RE.search(text)
    if not m:
        return text.strip(), {}
    question = text[:m.start()].strip()
    parts = _EXAM_INLINE_OPT_RE.split(text[m.start():])
    # parts: [前导, 'A', 'xx', 'B', 'yy', ...]
    options = {}
    for i in range(1, len(parts) - 1, 2):
        options[parts[i]] = parts[i + 1].strip()
    return question, options


def convert_exam_pdfs_to_csv():
    """解析 题库pdf 目录中的考研真题 -> 题库.pdf.csv（可重复运行，覆盖更新）
    返回 (total, skipped_scans, failed_list)"""
    import fitz
    if not os.path.exists(PDF_SOURCE_DIR):
        logger.warning(f"真题目录不存在：{PDF_SOURCE_DIR}")
        return 0, [], []

    rows = [HEADERS]
    total = 0
    skipped_scans = []   # 扫描版（无文本层）文件清单
    failed = []          # 解析失败清单

    for fname in sorted(os.listdir(PDF_SOURCE_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(PDF_SOURCE_DIR, fname)
        try:
            doc = fitz.open(fpath)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(text.strip()) < 200:
                skipped_scans.append(fname)   # 扫描版无文本层，跳过
                continue

            # 补前导换行：保证第一题也能按题号切分（blocks[0] 为题前说明，忽略）
            text = "\n" + text

            file_total = 0
            if _EXAM_NEW_RE.search(text):
                # 2024 格式：按 "2024N1A." 切分
                blocks = re.split(r'\n\s*(\d{4}N\d+[ABX]?)[.、]?\s*', text)
                for i in range(1, len(blocks) - 1, 2):
                    q, opts, ans = _split_options_block(blocks[i + 1].split('\n'))
                    if q and opts:
                        opt_text = "；".join(f"{k}.{v}" for k, v in sorted(opts.items()))
                        rows.append([q, opt_text, ans, "", ""])
                        file_total += 1
            else:
                # 老格式 / 2025 格式：按 "1." / "1、" 切分
                blocks = re.split(r'\n\s*(\d{1,3})[.、)）]\s*', text)
                for i in range(1, len(blocks) - 1, 2):
                    body = blocks[i + 1].strip()
                    q, opts, ans = _split_options_block(body.split('\n'))
                    if not opts:
                        q, opts = _split_options_inline(body)
                    if q and opts:
                        opt_text = "；".join(f"{k}.{v}" for k, v in sorted(opts.items()))
                        rows.append([q, opt_text, ans, "", ""])
                        file_total += 1

            logger.info(f"✅ 真题解析 {fname}: {file_total} 题")
            total += file_total
        except Exception as e:
            failed.append(f"{fname}: {str(e)[:80]}")
            logger.error(f"❌ 真题解析失败 {fname}: {e}")

    with open(PDF_CSV, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    logger.info(f"真题库更新完成：共 {total} 题；扫描版跳过 {len(skipped_scans)} 份；失败 {len(failed)} 份")
    return total, skipped_scans, failed


# ===== 总入口 =====
def update_all():
    """一键更新题库：返回 (doc_cnt, pdf_cnt, skipped_scans, failed_list, err)"""
    err = ""
    doc_cnt, pdf_cnt = 0, 0
    skipped_scans, failed = [], []
    try:
        doc_cnt = convert_doc_to_docx()
        doc_cnt += convert_docx_to_csv()
    except Exception as e:
        err += f"doc 处理失败: {e}; "
        logger.error(err)
    try:
        pdf_cnt, skipped_scans, failed = convert_exam_pdfs_to_csv()
    except Exception as e:
        err += f"pdf 处理失败: {e}; "
        logger.error(err)
    logger.info(f"题库更新完成：docx {doc_cnt} 题，真题pdf {pdf_cnt} 题")
    return doc_cnt, pdf_cnt, skipped_scans, failed, err


if __name__ == "__main__":
    d, p, s, f, e = update_all()
    print(f"doc: {d}, pdf: {p}, 扫描版跳过: {s}, 失败: {f}, err: {e}")
