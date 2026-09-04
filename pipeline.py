# -*- coding: utf-8 -*-
"""
混合 RAG Pipeline（重建版：快速模式实现）
简单问题直接检索 + 生成；保留 result 字段与 med_qa 的接口兼容。
支持：复习串联上下文 + 三档难度填空题 + 对话上下文问题改写（指代消解）。
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGResult:
    answer: str = ""
    blocked: bool = False
    security: dict = field(default_factory=dict)
    used_react: bool = False
    llm_calls: int = 1
    agent_steps: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    route_type: str = "general"
    # 结合对话历史理解后的完整问题（若与原问题不同，说明发生了指代消解）
    effective_question: str = ""


class ModularRAGPipeline:
    """最小可用的 RAG Pipeline：检索教材 + 相似题 + LLM 生成"""

    def __init__(self, llm_callback=None, user_id: str = "default"):
        self.llm_callback = llm_callback
        self.user_id = user_id

    def _retrieve(self, question: str) -> str:
        try:
            from retrieval_test import search_knowledge
            return search_knowledge(question, top_k=5) or ""
        except Exception:
            return ""

    def _similar_questions(self, question: str) -> list:
        try:
            from question_bank import search_similar_question
            return search_similar_question(question, top_k=3) or []
        except Exception:
            return []

    # ---------- 对话上下文：问题理解（指代消解） ----------
    # 命中这些词说明问题可能依赖上文（"各种药物"、"上述" 等）
    _REFER_RE = re.compile(
        r"(这些|那些|各种|上述|以上|它们|他们|该病|该药|这种|那种|"
        r"这个|那个|其中|分别|它(?:们)?的|其(?:他|它)?的|下列|如下)")

    def _history_text(self, session_history, max_rounds=4, max_chars=600):
        """把会话历史转成简短文本（兼容 {"question","answer"} 与 {"role","content"}）"""
        if not session_history:
            return ""
        lines = []
        for item in list(session_history)[-max_rounds:]:
            if not isinstance(item, dict):
                continue
            if "question" in item:
                lines.append(f"用户：{item.get('question', '')}")
                a = str(item.get("answer", "") or "")
                if a:
                    lines.append(f"助手：{a[:150]}")
            elif "role" in item:
                role = "用户" if item.get("role") == "user" else "助手"
                lines.append(f"{role}：{str(item.get('content', ''))[:150]}")
        text = "\n".join(lines)
        return text[-max_chars:]

    def _needs_rewrite(self, question, session_history) -> bool:
        """有对话历史，且问题含指代词或极短（几乎必然依赖上文）时，才结合上下文理解"""
        if not question or not session_history:
            return False
        if len(question) <= 8:          # 极短问题必然依赖上文
            return True
        return bool(self._REFER_RE.search(question))

    def _rewrite_question(self, question, history_text) -> str:
        """让 LLM 结合历史把问题改写为独立完整的问题；失败则原样返回"""
        sys_p = (
            "你是医学知识库的『问题理解』模块。根据对话历史，把用户最新问题改写成一个"
            "独立完整、不依赖上下文即可检索的医学问题。\n"
            "规则：1) 问题本身已完整则原样输出；2) 若含指代（这些/各种/上述/该病/它等），"
            "用历史中明确提到的具体对象（疾病、药物、概念等）补全；"
            "3) 只输出改写后的问题本身，不要任何解释、引号或多余标点。")
        usr_p = f"对话历史：\n{history_text}\n\n最新问题：{question}"
        try:
            raw = self.llm_callback([
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ]) if self.llm_callback else None
            if raw:
                q2 = re.sub(r'^["\'「『“]|["\'」』”]$', '', raw.strip()).strip()
                q2 = re.sub(r'\s+', ' ', q2)
                if 2 < len(q2) <= 80 and q2 != question:
                    return q2
        except Exception:
            pass
        return question

    def run(self, question: str, session_history: Optional[list] = None,
            review_context: str = "", brief: bool = False, subject_tags: Optional[list] = None,
            enable_answer: bool = True, enable_fills: bool = True, enable_review: bool = True):
        """执行一次问答：理解上下文 -> 检索 -> 组装 -> LLM 生成
        session_history: 本次会话之前的问答记录（用于指代消解）
        review_context: 之前相关提问的复习串联上下文（可为空）
        brief: 快速模式（手机版）——回答精简
        subject_tags: 命中的多学科标签
        enable_answer: 是否输出回答正文（关闭=只出填空/串联，如"明知故问"）
        enable_fills: 是否生成填空题
        enable_review: 是否生成知识点串联复习（关闭时填空题增至每档 3 道）"""
        # 1. 结合对话历史理解问题（指代消解）
        effective_q = question
        if self._needs_rewrite(question, session_history):
            hist_text = self._history_text(session_history)
            effective_q = self._rewrite_question(question, hist_text)

        # 2. 用理解后的问题检索（TF-IDF + 相似题）
        ctx = self._retrieve(effective_q)
        sims = self._similar_questions(effective_q)

        sim_text = "\n".join(
            f"Q: {s.get('question', '')}\nA: {s.get('answer', '')}"
            for s in sims[:3]
        )

        # 3. 多学科冲突提示：同时命中多个学科时，让 LLM 分别给出各科标准并标记
        multi_subjects = [s for s in (subject_tags or []) if s]
        if len(multi_subjects) >= 2:
            conflict_note = (
                "\n\n另外：该问题涉及多个学科视角（"
                + "、".join(multi_subjects)
                + "）。若各学科对同一问题的标准/观点/治疗原则存在差异，"
                "请分别列出各学科的标准，并在每条前面标注学科名，如"
                "【内科】……【外科】……，方便用户对比学习。"
            )
        else:
            conflict_note = ""

        # 3. 组装 system prompt（三个独立开关：出回答 / 出填空题 / 出知识点串联）
        #    关闭知识点串联时，填空题增加为每档 3 道（共 9 道）
        fills_per_level = 3 if (enable_fills and not enable_review) else 1
        fills_note = (
            f"\n\n生成三档难度的填空题，每档 {fills_per_level} 道"
            f"（共 {fills_per_level * 3} 道，先基础后进阶再挑战）。"
            "填空题的考察角度、挖空考点与表述风格，请对齐【相似真题参考】中的考研真题："
            "优先选择与当前知识点同类的真题考点，把关键概念/机制/数值处挖空，"
            "用真题式的设问方式改写为填空，而不是简单抄题干。\n"
            "每道题严格按以下格式输出（同一难度多道就重复该难度标记），用空行分隔：\n"
            "【基础填空】<题干，填空处用 ____ 表示>\n参考答案：<答案>\n"
            "【进阶填空】<跨知识点或多空填空，考查机制理解>\n参考答案：<答案>\n"
            "【挑战填空】<结合机制/临床/病例的推理型填空，难度最高>\n参考答案：<答案>"
        )
        review_note = (
            "\n\n另外：由于用户之前问过相关知识点，请输出一段「知识点串联复习」，"
            "把本题知识点与【你之前的相关提问】联系起来，"
            "说明二者关联或异同，帮助用户复习巩固，格式：\n"
            "📌 知识点串联：<联系说明>"
        )

        if brief:
            system_prompt = (
                "你是一名专业的医学知识库助手。请严格基于【参考资料】回答用户问题，"
                "语言简洁准确、重点突出，适当分点，总长度控制在200字以内；"
                "参考资料不足时请明确说明。" + conflict_note
            )
        else:
            base = ("你是一名专业的医学知识库助手。请严格基于【参考资料】作答，"
                    "语言简洁准确；参考资料不足时请明确说明。")
            if enable_answer:
                base += "请先给出对该问题的完整回答正文。"
            else:
                base += "本次**不需要回答正文**，直接输出下方要求的内容即可。"
            system_prompt = base + conflict_note
            if enable_fills:
                system_prompt += fills_note
        if review_context and not brief and enable_review:
            system_prompt += review_note

        user_content = f"【参考资料】\n{ctx or '（无）'}\n"
        if sim_text:
            user_content += f"\n【相似真题参考】（填空题请参考这些真题的考点与风格）\n{sim_text}\n"
        if review_context and enable_review:
            user_content += f"\n【你之前的相关提问】\n{review_context}\n"
        if effective_q != question:
            user_content += f"\n【问题（已结合上下文理解）】\n{effective_q}\n【用户原话】\n{question}"
        else:
            user_content += f"\n【问题】\n{question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        answer = ""
        if self.llm_callback:
            if brief:
                # 快速模式：限制输出长度；回调不支持 max_tokens 参数时降级为默认调用
                try:
                    answer = self.llm_callback(messages, max_tokens=400) or "（生成失败，请检查 API 配置）"
                except TypeError:
                    answer = self.llm_callback(messages) or "（生成失败，请检查 API 配置）"
            else:
                answer = self.llm_callback(messages) or "（生成失败，请检查 API 配置）"
        else:
            answer = "（未配置 LLM 回调）"

        from .router import QuestionRouter
        route = QuestionRouter().route(question)

        return RAGResult(
            answer=answer,
            used_react=False,
            llm_calls=1,
            route_type=route,
            effective_question=effective_q,
            validation={"confidence": "medium"},
        )
