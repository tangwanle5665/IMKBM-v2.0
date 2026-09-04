# -*- coding: utf-8 -*-
"""问题路由（重建版：关键词分类）"""


class QuestionRouter:
    """把问题按类型路由：definition / mechanism / treatment / comparison / general"""

    def route(self, question: str) -> str:
        q = question or ""
        if any(k in q for k in ("是什么", "定义", "何为", "含义", "概念")):
            return "definition"
        if any(k in q for k in ("机制", "原理", "为什么", "过程", "途径")):
            return "mechanism"
        if any(k in q for k in ("治疗", "药物", "用药", "方案", "处理")):
            return "treatment"
        if any(k in q for k in ("区别", "比较", "对比", "vs", "还是")):
            return "comparison"
        return "general"
