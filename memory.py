# -*- coding: utf-8 -*-
"""三层记忆架构（重建版：内存级最小实现）"""

import threading


class _Preference:
    """用户偏好（内存版）"""

    def __init__(self):
        self.preferred_length = "medium"
        self.likes_images = False
        self.subject_weights = {}
        self.route_type_history = {}

    def to_dict(self):
        return {
            "preferred_length": self.preferred_length,
            "likes_images": self.likes_images,
            "subject_weights": self.subject_weights,
            "route_type_history": self.route_type_history,
        }


class MemoryManager:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.preference = _Preference()
        self._feedback = {"thumbs_up": 0, "thumbs_down": 0, "correction": 0}
        self._lock = threading.Lock()

    def get_feedback_stats(self) -> dict:
        with self._lock:
            return dict(self._feedback)

    def record_feedback(self, question="", answer="", feedback="",
                        comment="", route_type="general", sources=None):
        with self._lock:
            key = {"thumbs_up": "thumbs_up", "thumbs_down": "thumbs_down",
                   "correction": "correction"}.get(feedback)
            if key:
                self._feedback[key] = self._feedback.get(key, 0) + 1
            if route_type:
                self.preference.route_type_history[route_type] = \
                    self.preference.route_type_history.get(route_type, 0) + 1
        return True

    def update_from_question(self, question="", route_type="general", used_image=False):
        with self._lock:
            self.preference.route_type_history[route_type] = \
                self.preference.route_type_history.get(route_type, 0) + 1
            if used_image:
                self.preference.likes_images = True
            # 简单学科偏好统计（按关键词粗分）
            for kw, subj in [("心","心血管"), ("肺","呼吸"), ("肝","消化"),
                             ("肾","泌尿"), ("血糖|糖尿","内分泌"), ("免疫","风湿免疫")]:
                import re
                if re.search(kw, question or ""):
                    self.preference.subject_weights[subj] = \
                        self.preference.subject_weights.get(subj, 0) + 1

    def get_adaptive_weights(self, route_type="general") -> dict:
        return {"dense": 0.6, "sparse": 0.4, "keyword": 0.15, "bank": 0.2}


_instance = None
_lock = threading.Lock()


def get_memory_manager(user_id: str = "default") -> MemoryManager:
    global _instance
    with _lock:
        if _instance is None:
            _instance = MemoryManager(user_id)
        return _instance


def reset_memory_manager():
    global _instance
    with _lock:
        _instance = None
