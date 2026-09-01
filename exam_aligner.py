#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真题导入对齐模块（重建版）
将一份试卷（PDF/Word/TXT）与可选答案文件对齐，解析为题目并合并进题库 CSV。
提供接口：
    import_exam_paper(paper_path, answer_path=None) -> (success: bool, msg: str)
"""

import os
import re
import csv

from utils import logger

HEADERS = ["question", "options", "answer", "analysis", "know_point"]
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "题库.docx.csv")


def _read_text(path):
    """读取 PDF/Word/TXT 为纯文本"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(path)
            txt = "\n".join(page.get_text() for page in doc if page.get_text())
            doc.close()
            return txt
        except Exception:
            try:
                from pymupdf import open as fitz_open
                doc = fitz_open(path)
                txt = "\n".join(page.get_text() for page in doc if page.get_text())
                doc.close()
                return txt
            except Exception as e:
                raise RuntimeError(f"PDF 读取失败: {e}")
    elif ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            return f.read()
    else:
        raise RuntimeError(f"不支持的文件类型: {ext}")


def _parse_paper(text):
    """把试卷文本解析为题目列表"""
    questions = []
    # 按题号分块
    blocks = re.split(r'\n\s*(\d+)[.、)）]\s*', text)
    for i in range(1, len(blocks) - 1, 2):
        content = blocks[i + 1] if i + 1 < len(blocks) else ""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            continue
        q_text = lines[0]
        options = {}
        rest = []
        opt_pattern = re.compile(r'^([A-E])[.、)）]\s*(.*)')
        for line in lines[1:]:
            m = opt_pattern.match(line)
            if m:
                options[m.group(1)] = m.group(2).strip()
            else:
                rest.append(line)
        option_text = "；".join(f"{k}.{v}" for k, v in sorted(options.items()))
        questions.append([q_text, option_text, "", "", ""])
    return questions


def _extract_answers(text, count):
    """从答案文本中提取答案列表（每行一个或连续的字母序列）"""
    answers = []
    if not text:
        return answers
    # 尝试 "1.B 2.C ..." 形式
    pairs = re.findall(r'(\d+)\s*[.、)）]?\s*([A-Ea-e])', text)
    if pairs:
        for _, a in pairs:
            answers.append(a.upper())
    else:
        # 连续字母序列
        seq = re.findall(r'[A-Ea-e]', text)
        answers = [a.upper() for a in seq]
    return answers[:count]


def import_exam_paper(paper_path, answer_path=None):
    """导入一份试卷（+可选答案），合并进题库 CSV"""
    try:
        if not paper_path or not os.path.exists(paper_path):
            return False, "试卷文件不存在"

        paper_text = _read_text(paper_path)
        questions = _parse_paper(paper_text)
        if not questions:
            return False, "未能从试卷中解析出题目"

        # 答案文件
        answers = []
        if answer_path and os.path.exists(answer_path):
            ans_text = _read_text(answer_path)
            answers = _extract_answers(ans_text, len(questions))

        for i, q in enumerate(questions):
            if i < len(answers) and answers[i]:
                q[2] = answers[i]

        # 合并进题库 CSV
        csv_path = DEFAULT_CSV
        rows = [HEADERS]
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
                if rows and rows[0] == HEADERS:
                    pass
                else:
                    rows = [HEADERS] + rows
        rows.extend(questions)

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(rows)

        with_answer = sum(1 for q in questions if q[2])
        logger.info(f"真题导入完成：{len(questions)} 题（含答案 {with_answer}）")
        return True, f"导入成功：{len(questions)} 题（含答案 {with_answer} 题）"
    except Exception as e:
        logger.error(f"真题导入失败: {e}")
        return False, f"导入失败: {e}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ok, msg = import_exam_paper(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(msg)
    else:
        print("用法: python exam_aligner.py 试卷文件 [答案文件]")
