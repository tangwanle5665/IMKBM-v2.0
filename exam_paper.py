#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从考试题库数据库按难度分池随机抽取填空题生成试卷
- 数据库目录：考试题库/数据库/*.json（六板块 -> 题目条目 -> 三档填空+答案）
- 试卷构成：基础 40 题 + 中等 40 题 + 挑战 20 题 = 共 100 题
- 抽取规则：各区独立随机抽取（同区内不重复）；同一知识点可跨区重复出现
- 某区题目不足时给出缺口提示，暂不生成
- 试卷输出：考试题库/试卷/试卷_日期时间.docx（题目区在前，答案按题号在最后）
"""

import os
import json
import random
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "考试题库", "数据库")
PAPER_DIR = os.path.join(BASE_DIR, "考试题库", "试卷")

# 各区抽取数量：基础 / 中等 / 挑战
SECTION_PLAN = [
    ("基础", "基础填空", 40),
    ("中等", "进阶填空", 40),
    ("挑战", "挑战填空", 20),
]


def load_all_questions(db_dir=DB_DIR):
    """汇总所有 json 数据库的题目条目（贴标签扁平格式 version2）
    兼容旧的分组格式；每条附上主学科 _subject 与标签 subjects"""
    items = []
    if not os.path.isdir(db_dir):
        return items
    for fname in sorted(os.listdir(db_dir)):
        if not fname.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(db_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "questions" in data:
                # 新格式：扁平列表 + subjects 标签
                for it in data["questions"]:
                    if isinstance(it, dict) and it.get("fills"):
                        it = dict(it)
                        subs = it.get("subjects") or ["综合"]
                        it["_subject"] = subs[0] if subs else "综合"
                        items.append(it)
            elif isinstance(data, dict):
                # 旧格式：{"内科": [...]} 分组
                for subj, arr in data.items():
                    if not isinstance(arr, list):
                        continue
                    for it in arr:
                        if isinstance(it, dict) and it.get("fills"):
                            it = dict(it)
                            it["_subject"] = subj
                            it["subjects"] = it.get("subjects") or [subj]
                            items.append(it)
            elif isinstance(data, list):
                for it in data:
                    if isinstance(it, dict) and it.get("fills"):
                        it = dict(it)
                        subs = it.get("subjects") or ["综合"]
                        it["_subject"] = subs[0] if subs else "综合"
                        items.append(it)
        except Exception as e:
            print(f"⚠️ 数据库加载失败 {fname}: {e}")
    # 跨文件去重（自动同步文件与手动快照可能重叠）
    seen, unique = set(), []
    for it in items:
        q = it.get("question", "")
        if q in seen:
            continue
        seen.add(q)
        unique.append(it)
    return unique


def _pool_questions(items, key):
    """按档位收集小题池：返回 [(question, answer, subject)]"""
    pool = []
    for it in items:
        fills = it.get("fills") or {}
        answers = it.get("answers") or {}
        qs = fills.get(key)
        ans = answers.get(key)
        q_list = qs if isinstance(qs, list) else ([qs] if qs else [])
        a_list = ans if isinstance(ans, list) else ([ans] if ans else [])
        for qi, q in enumerate(q_list):
            pool.append({
                "question": q,
                "answer": a_list[qi] if qi < len(a_list) else "",
                "subject": it.get("_subject", ""),
                "source": it.get("question", ""),
            })
    return pool


def _p(doc, text, size=10.5, bold=False):
    """楷体段落"""
    from docx.shared import Pt
    from docx.oxml.ns import qn
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "楷体"
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')
    return p


def generate_exam_paper(db_dir=DB_DIR, paper_dir=PAPER_DIR):
    """按 基础40 + 中等40 + 挑战20 抽取生成试卷 docx
    返回 dict: {success, msg, path, total, plan}"""
    try:
        from docx import Document
    except Exception:
        return {"success": False, "msg": "python-docx 未安装，无法生成试卷"}

    items = load_all_questions(db_dir)
    if not items:
        return {"success": False, "msg": f"数据库目录为空或没有题目：\n{db_dir}"}

    # ===== 按档位分池并检查数量 =====
    pools = {}
    for section, key, need in SECTION_PLAN:
        pools[section] = _pool_questions(items, key)

    shortage = []
    for section, key, need in SECTION_PLAN:
        have = len(pools[section])
        if have < need:
            shortage.append(f"{section}填空：{have} 题（需 {need} 题，差 {need - have} 题）")
    if shortage:
        return {
            "success": False,
            "msg": "当前题库不足以生成试卷（100题 = 基础40 + 中等40 + 挑战20）：\n\n"
                   + "\n".join(shortage)
                   + "\n\n请继续提问积累相应难度的填空题后再试。",
            "total": len(items),
        }

    # ===== 各区独立随机抽取（同区不重复，知识点可跨区重复） =====
    random.seed()
    chosen = []          # [(section, question, answer)]
    for section, key, need in SECTION_PLAN:
        picked = random.sample(pools[section], need)
        chosen.extend((section, p["question"], p["answer"]) for p in picked)

    # 混合打乱 100 道题的顺序，再全局编号 1-100（模拟真实试卷，不按难度分区）
    random.shuffle(chosen)

    os.makedirs(paper_dir, exist_ok=True)
    now = datetime.now()
    fname = f"试卷_{now.strftime('%Y-%m-%d_%H-%M')}.docx"
    fpath = os.path.join(paper_dir, fname)

    doc = Document()
    # ===== 卷头 =====
    _p(doc, "医学知识库 · 考研填空题试卷", size=16, bold=True)
    _p(doc, f"共 100 题（基础 40 + 中等 40 + 挑战 20，随机抽取、混合排序）　|　生成时间：{now.strftime('%Y-%m-%d %H:%M')}", size=10.5)
    _p(doc, "考生须知：请先完成题目，答案统一在文末，按题号对照。", size=10.5)
    doc.add_page_break()

    # ===== 题目区：100 题混合编号 1-100 =====
    for idx, (sec, q, a) in enumerate(chosen, 1):
        _p(doc, f"{idx}. {q}", size=10.5)

    # ===== 答案区（文末，按题号） =====
    doc.add_page_break()
    _p(doc, "参考答案（请做完后再看！）", size=14, bold=True)
    for idx, (sec, q, a) in enumerate(chosen, 1):
        _p(doc, f"{idx}. {a or '（未提供）'}", size=10.5)

    doc.save(fpath)
    return {
        "success": True,
        "msg": (f"✅ 试卷生成成功！\n\n"
                f"共 100 题：基础 40 + 中等 40 + 挑战 20\n"
                f"保存位置：{fpath}"),
        "path": fpath,
        "total": len(items),
        "plan": {s: n for s, _, n in SECTION_PLAN},
    }


if __name__ == "__main__":
    result = generate_exam_paper()
    print(result["msg"])
