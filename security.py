# -*- coding: utf-8 -*-
"""安全护栏（重建版：最小实现，返回空统计）"""


class SecurityGuard:
    """四层安全护栏的最小占位实现"""

    def __init__(self):
        self.total_events = 0
        self.blocked = 0
        self.warnings = 0

    def check(self, text: str):
        return None

    def get_stats(self) -> dict:
        return {
            "total_events": self.total_events,
            "blocked": self.blocked,
            "warnings": self.warnings,
        }
