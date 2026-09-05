#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import os
import json
import queue
import re
import traceback
from PIL import Image, ImageTk
from pdf_parser import parse_files_to_txt
from ppt_parser import parse_all_ppt
from med_qa import med_exam_chat_stream, clear_cache, record_feedback, daily_snapshot
from config import BASE_DIR
from utils import logger, get_bge_model, clear_vector_db_cache
from update_question_bank import update_all
from question_bank import init_bank
from exam_paper import generate_exam_paper
from full_update import full_update

# ==================== 医学主题配色（现代医疗风） ====================
BG_COLOR = "#F0F4F8"          # 页面背景 浅灰蓝
CARD_COLOR = "#FFFFFF"        # 卡片白
ACCENT_COLOR = "#2563EB"      # 主蓝
ACCENT_DARK = "#1E40AF"       # 标题栏深蓝
ACCENT_LIGHT = "#DBEAFE"      # 浅蓝（气泡/选中）
TEXT_COLOR = "#1F2937"        # 主文字
SUB_TEXT = "#64748B"          # 次要文字
LINE_COLOR = "#E2E8F0"        # 分隔线
SUCCESS_COLOR = "#059669"     # 医疗绿
WARN_COLOR = "#D97706"        # 橙
RED_COLOR = "#DC2626"         # 红
PURPLE_COLOR = "#7C3AED"      # 紫
BROWN_COLOR = "#92400E"       # 棕
BTN_FONT = ("微软雅黑", 9)
TITLE_FONT = ("微软雅黑", 14, "bold")

SAVE_DIR = os.path.join(BASE_DIR, "saved_sessions")
os.makedirs(SAVE_DIR, exist_ok=True)


def make_btn(parent, text, command, bg, fg="white", font=BTN_FONT,
             padx=12, pady=5, width=None, hover=None, state=None):
    """统一样式按钮：扁平 + 手型 + 悬停变色"""
    b = tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        relief=tk.FLAT, bd=0, font=font, cursor="hand2",
        padx=padx, pady=pady, width=width, state=state,
        activebackground=hover or bg, activeforeground=fg,
        disabledforeground="#94A3B8",
    )
    b.bind("<Enter>", lambda e: b.config(bg=hover or bg))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


def parse_all_docs():
    """解析教材(PDF/Word)与教学PPT为文本（解析文档按钮）"""
    parse_files_to_txt()
    parse_all_ppt()


class ImageViewerWindow:
    """独立图片查看窗口（内存优化版）"""
    def __init__(self, parent, img_path):
        self.parent = parent
        self.img_path = img_path
        self.window = tk.Toplevel(parent)
        self.window.title("图片查看器")
        self.window.geometry("800x600")
        self.window.minsize(200, 150)
        self.current_img = None
        self.canvas = tk.Canvas(self.window, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.window.bind("<Configure>", self.on_resize)
        self.max_load_size = (1200, 900)
        self.load_image()

    def load_image(self):
        try:
            if not os.path.exists(self.img_path):
                self.canvas.create_text(400, 300, text="图片文件不存在",
                                        font=("微软雅黑", 16), fill="red")
                return
            with Image.open(self.img_path) as img:
                if img.width > self.max_load_size[0] or img.height > self.max_load_size[1]:
                    img.thumbnail(self.max_load_size, Image.Resampling.LANCZOS)
                self.img = img.copy()
            self.update_image()
        except Exception as e:
            self.canvas.create_text(400, 300, text=f"加载失败: {e}",
                                    font=("微软雅黑", 14), fill="red")

    def on_resize(self, event=None):
        self.update_image()

    def update_image(self):
        if not hasattr(self, 'img') or self.img is None:
            return
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        if width < 10 or height < 10:
            return
        img_copy = self.img.copy()
        img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.current_img = ImageTk.PhotoImage(img_copy)
        self.canvas.delete("all")
        self.canvas.create_image(width//2, height//2, anchor=tk.CENTER, image=self.current_img)

    def close(self):
        self.img = None
        self.current_img = None
        self.window.destroy()


class SessionTab(ttk.Frame):
    def __init__(self, parent, notebook, session_id, title, app=None):
        super().__init__(parent)
        self.notebook = notebook
        self.session_id = session_id
        self.title = title
        self.app = app            # MedApp 引用（用于刷新侧栏）
        self.history = []
        self.load_history()
        self.current_image_path = None
        self.image_window = None
        self.streaming = False
        self.stream_answer = ""
        self.current_question = ""
        self.create_widgets()
        self.result_queue = queue.Queue()
        self.after(100, self.check_queue)

    def create_widgets(self):
        # ===== 底部输入区（白色圆角卡片） =====
        input_card = tk.Frame(self, bg=CARD_COLOR, highlightthickness=1,
                              highlightbackground=LINE_COLOR)
        input_card.pack(side="bottom", fill="x", padx=14, pady=(4, 12))

        tk.Label(input_card, text="✚  输入问题", bg=CARD_COLOR, fg=SUB_TEXT,
                 font=("微软雅黑", 8)).pack(anchor="w", padx=10, pady=(8, 2))
        self.entry = tk.Entry(input_card, font=("微软雅黑", 11), relief="flat",
                              bg="#F8FAFC", fg=TEXT_COLOR, insertbackground=ACCENT_COLOR,
                              highlightthickness=1, highlightbackground=LINE_COLOR,
                              highlightcolor=ACCENT_COLOR)
        self.entry.pack(fill="x", padx=10, ipady=7)
        self.entry.bind("<Return>", self.ask)

        # 输出开关：回答正文 / 填空题 / 知识点串联（关串联时填空增至每档3道）
        opt_row = tk.Frame(input_card, bg=CARD_COLOR)
        opt_row.pack(fill="x", padx=10, pady=(6, 0))
        self.var_answer = tk.BooleanVar(value=True)
        self.var_fills = tk.BooleanVar(value=True)
        self.var_review = tk.BooleanVar(value=True)
        cb_style = {"bg": CARD_COLOR, "fg": SUB_TEXT, "selectcolor": "#FFFFFF",
                    "activebackground": CARD_COLOR, "font": ("微软雅黑", 8),
                    "highlightthickness": 0, "bd": 0, "cursor": "hand2"}
        tk.Checkbutton(opt_row, text="出回答正文", variable=self.var_answer,
                       command=self._on_option_change, **cb_style).pack(side="left")
        tk.Checkbutton(opt_row, text="出填空题", variable=self.var_fills,
                       command=self._on_option_change, **cb_style).pack(side="left", padx=(12, 0))
        tk.Checkbutton(opt_row, text="出知识点串联", variable=self.var_review,
                       command=self._on_option_change, **cb_style).pack(side="left", padx=(12, 0))
        self.opt_hint = tk.Label(opt_row, text="", bg=CARD_COLOR, fg=WARN_COLOR,
                                 font=("微软雅黑", 8))
        self.opt_hint.pack(side="left", padx=(12, 0))

        btn_row = tk.Frame(input_card, bg=CARD_COLOR)
        btn_row.pack(fill="x", padx=10, pady=(8, 9))
        self.ask_btn = make_btn(
            btn_row, "提问 ➤", self.ask, ACCENT_COLOR,
            hover=ACCENT_DARK, padx=22, pady=5
        )
        self.ask_btn.pack(side="left")
        self.view_image_btn = make_btn(
            btn_row, "🖼 查看大图", self.open_image_window, SUCCESS_COLOR,
            padx=10, pady=5, hover="#047857", state=tk.DISABLED
        )
        self.view_image_btn.pack(side="left", padx=(8, 0))
        make_btn(btn_row, "清空历史", self.clear_history, RED_COLOR,
                 padx=10, pady=5, hover="#B91C1C").pack(side="left", padx=(8, 0))
        make_btn(btn_row, "👍", lambda ft="thumbs_up": self._send_feedback(ft),
                 SUCCESS_COLOR, padx=8, pady=5).pack(side="left", padx=(8, 0))
        make_btn(btn_row, "👎", lambda ft="thumbs_down": self._send_feedback(ft),
                 RED_COLOR, padx=8, pady=5).pack(side="left", padx=(8, 0))
        self.status_label = tk.Label(
            btn_row, text="● 就绪", bg=CARD_COLOR, fg=SUCCESS_COLOR,
            font=("微软雅黑", 9)
        )
        self.status_label.pack(side="right")

        # ===== 对话区（聊天气泡） =====
        content_frame = tk.Frame(self, bg=BG_COLOR)
        content_frame.pack(fill="both", expand=True, padx=14, pady=(6, 4))

        self.image_frame = tk.Frame(content_frame, bg=CARD_COLOR,
                                    highlightthickness=1, highlightbackground=LINE_COLOR)
        self.image_label = tk.Label(self.image_frame, bg=CARD_COLOR)
        self.image_label.pack(padx=6, pady=6)

        self.out = scrolledtext.ScrolledText(
            content_frame, font=("微软雅黑", 10), wrap="word",
            bg=BG_COLOR, relief=tk.FLAT, bd=0, padx=6, pady=6,
            highlightthickness=0, spacing1=1, spacing2=1
        )
        self.out.pack(fill="both", expand=True)

        # 聊天气泡样式
        self.out.tag_configure("q", background=ACCENT_COLOR, foreground="#FFFFFF",
                               justify="right", lmargin1=90, lmargin2=90, rmargin=4,
                               spacing1=6, spacing2=6, font=("微软雅黑", 10, "bold"))
        self.out.tag_configure("a", background=CARD_COLOR, foreground=TEXT_COLOR,
                               lmargin1=8, lmargin2=8, rmargin=26,
                               spacing1=4, spacing2=4, font=("微软雅黑", 10))
        self.out.tag_configure("meta", foreground=SUB_TEXT,
                               font=("微软雅黑", 8), lmargin1=10, lmargin2=10)
        self.out.tag_configure("err", foreground=RED_COLOR,
                               font=("微软雅黑", 10), lmargin1=8, lmargin2=8)

        for item in self.history:
            self._display_plain_answer(item["question"], item["answer"])

    def _display_plain_answer(self, question, answer):
        if answer is None:
            answer = "⚠️ 回答为空"
        # 问题气泡（右对齐蓝底）
        self.out.insert(tk.END, f"{question}", "q")
        # 回答卡片（左对齐白底）
        clean = re.sub(r'\[IMAGE:.*?\]', '', answer)
        self.out.insert(tk.END, f"\n{clean}\n", "a")
        img_tag = "[IMAGE:"
        if img_tag in answer:
            start = answer.find(img_tag) + len(img_tag)
            end = answer.find("]", start)
            if end != -1:
                img_path = answer[start:end]
                self.show_image_preview(img_path)
        self.out.see(tk.END)

    def show_image_preview(self, img_path):
        try:
            if not os.path.exists(img_path):
                self.hide_image_preview()
                self.view_image_btn.config(state=tk.DISABLED)
                return
            img = Image.open(img_path)
            max_width = 280
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.current_preview_img = ImageTk.PhotoImage(img)
            img.close()
            self.image_label.config(image=self.current_preview_img)
            self.image_frame.pack(fill="x", pady=(0, 5))
            self.current_image_path = img_path
            self.view_image_btn.config(state=tk.NORMAL)
        except Exception as e:
            logger.error(f"加载预览图片失败: {traceback.format_exc()}")
            self.hide_image_preview()

    def hide_image_preview(self):
        self.image_frame.pack_forget()
        self.current_image_path = None
        self.view_image_btn.config(state=tk.DISABLED)

    def open_image_window(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showwarning("提示", "没有可显示的图片")
            return
        if self.image_window is not None:
            try:
                self.image_window.window.lift()
                self.image_window.window.focus_force()
                return
            except:
                self.image_window = None
        self.image_window = ImageViewerWindow(self, self.current_image_path)
        self.image_window.window.protocol("WM_DELETE_WINDOW", self.on_image_window_close)

    def on_image_window_close(self):
        if self.image_window:
            self.image_window.close()
            self.image_window = None

    def _on_option_change(self):
        """开关联动提示：关回答=只出填空/串联；关串联时填空增至每档 3 道"""
        a, f, r = (self.var_answer.get(), self.var_fills.get(), self.var_review.get())
        if not a and not f:
            self.opt_hint.config(text="⚠ 回答和填空都关了，将没有输出")
        elif not a and f:
            self.opt_hint.config(text="只出填空模式" + ("（每档3道）" if not r else ""))
        elif not f:
            self.opt_hint.config(text="当前不出填空题")
        elif not r:
            self.opt_hint.config(text="知识点关闭 → 每档出 3 道填空")
        else:
            self.opt_hint.config(text="")

    def ask(self, event=None):
        q = self.entry.get().strip()
        if not q:
            messagebox.showwarning("提示", "请输入问题")
            return
        if self.streaming:
            return
        self.ask_btn.config(state="disabled")
        self.status_label.config(text="正在生成回答...", fg=ACCENT_COLOR)
        self.hide_image_preview()
        self.entry.delete(0, tk.END)
        self.current_question = q
        thread = threading.Thread(target=self._do_ask_stream, args=(q,), daemon=True)
        thread.start()

    def _do_ask_stream(self, question):
        try:
            generator = med_exam_chat_stream(
                question, self.history,
                enable_answer=self.var_answer.get(),
                enable_fills=self.var_fills.get(),
                enable_review=self.var_review.get(),
            )
            self.result_queue.put({"type": "stream_start", "question": question})
            full_answer = ""
            for chunk in generator:
                full_answer += chunk
                self.result_queue.put({"type": "stream_chunk", "chunk": chunk})
            self.result_queue.put({
                "type": "stream_end",
                "full_answer": full_answer,
                "question": question
            })
        except Exception as e:
            logger.error(f"问答线程异常: {traceback.format_exc()}")
            self.result_queue.put({"type": "error", "error": str(e), "question": question})

    def check_queue(self):
        try:
            while True:
                result = self.result_queue.get_nowait()
                if result["type"] == "stream_start":
                    self.streaming = True
                    self.stream_answer = ""
                    question = result["question"]
                    self.out.insert(tk.END, f"{question}", "q")
                    self.out.insert(tk.END, "\n", "a")
                    self.out.see(tk.END)
                elif result["type"] == "stream_chunk":
                    chunk = result["chunk"]
                    self.out.insert(tk.END, chunk, "a")
                    self.stream_answer += chunk
                    self.out.see(tk.END)
                    self.update_idletasks()
                elif result["type"] == "stream_end":
                    full_answer = result["full_answer"]
                    question = result.get("question", self.current_question)
                    img_tag = ""
                    img_path = None
                    if "[IMAGE:" in full_answer:
                        start = full_answer.find("[IMAGE:")
                        img_tag = full_answer[start:]
                        clean_answer = full_answer[:start]
                        self.out.delete(f"end-{len(full_answer)+2}c", "end")
                        self.out.insert(tk.END, f"{clean_answer}\n", "a")
                        img_match = re.search(r'\[IMAGE:(.*?)\]', img_tag)
                        if img_match:
                            img_path = img_match.group(1)
                            if os.path.exists(img_path):
                                self.show_image_preview(img_path)
                    else:
                        self.out.insert(tk.END, "\n")
                    self.out.see(tk.END)
                    clean_for_history = full_answer.replace(img_tag, "").strip()
                    self.history.append({"question": question, "answer": clean_for_history})
                    self.save_history()
                    if self.app is not None:
                        self.app.refresh_sidebar()
                    self.streaming = False
                    self.ask_btn.config(state="normal")
                    self.status_label.config(text="● 就绪", fg=SUCCESS_COLOR)
                elif result["type"] == "error":
                    error_msg = result.get("error", "未知错误")
                    question = result.get("question", self.current_question)
                    self.out.insert(tk.END, f"❌ 出错：{error_msg}\n", "err")
                    self.out.see(tk.END)
                    self.streaming = False
                    self.ask_btn.config(state="normal")
                    self.status_label.config(text="● 就绪", fg=SUCCESS_COLOR)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"队列处理异常: {traceback.format_exc()}")
            self.streaming = False
            self.ask_btn.config(state="normal")
            self.status_label.config(text="就绪", fg="#666666")
        finally:
            self.after(100, self.check_queue)

    def clear_history(self):
        self.history = []
        self.out.delete(1.0, tk.END)
        self.hide_image_preview()
        self.save_history()

    def clear_cache(self):
        clear_cache()
        messagebox.showinfo("提示", "问答缓存已清空")

    def _send_feedback(self, feedback_type):
        if not self.history:
            messagebox.showwarning("提示", "当前会话无历史记录，无法反馈")
            return
        last = self.history[-1]
        try:
            success, msg = record_feedback(
                question=last["question"],
                answer=last["answer"],
                feedback_type=feedback_type
            )
            if success:
                messagebox.showinfo("反馈已记录", msg)
            else:
                messagebox.showerror("反馈失败", msg)
        except Exception as e:
            messagebox.showerror("错误", f"反馈提交失败: {e}")

    def load_history(self):
        file_path = os.path.join(SAVE_DIR, f"session_{self.session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)

    def save_history(self):
        file_path = os.path.join(SAVE_DIR, f"session_{self.session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


class MedApp:
    def __init__(self, root):
        self.root = root
        root.title("医学知识库智能问答系统V1.0")
        root.geometry("1120x720")
        root.minsize(900, 580)
        root.configure(bg=BG_COLOR)

        # ============ 左侧导航配色（深色对话风） ============
        self.SB_BG = "#111827"        # 侧栏深底
        self.SB_BG2 = "#1F2937"       # 分组底
        self.SB_FG = "#E5E7EB"        # 主文字
        self.SB_SUB = "#94A3B8"       # 小字
        self.SB_ACT = "#2563EB"       # 选中
        self.SB_HOV = "#334155"

        # ============ 主框架：左侧导航 + 右侧聊天 ============
        main = tk.Frame(root, bg=BG_COLOR)
        main.pack(fill="both", expand=True)

        # ---- 左导航 ----
        self.sidebar = tk.Frame(main, bg=self.SB_BG, width=240)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # ---- 右侧容器 ----
        right = tk.Frame(main, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True)

        # 会话容器：Notebook（隐藏标签页，由左侧列表切换）
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(10, 4))
        try:
            style = ttk.Style(root)
            style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
            style.configure("TNotebook.Tab", font=("微软雅黑", 9), padding=(6, 2))
        except Exception:
            pass

        # ===== 右区底部状态行（浅色细条） =====
        status_bar = tk.Frame(right, bg=CARD_COLOR, highlightthickness=1,
                              highlightbackground=LINE_COLOR)
        status_bar.pack(fill="x", padx=16, pady=(2, 8))
        self.status = tk.Label(
            status_bar,
            text="●  混合 RAG 检索 · DeepSeek 生成 · 支持上下文/三档填空/每日归档",
            bg=CARD_COLOR, fg=SUB_TEXT, pady=4, font=("微软雅黑", 8)
        )
        self.status.pack(fill="x")

        self.session_counter = 0
        self._active_sid = None
        self.load_all_sessions()

        if len(self.notebook.tabs()) == 0:
            self.new_session()

        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        threading.Thread(target=self._preload_bge, daemon=True).start()
        threading.Thread(target=self._load_bank_info, daemon=True).start()
        logger.info("🔄 BGE模型正在后台加载...")

    # ==================== 侧栏构建 ====================
    def _nav_btn(self, parent, text, cmd, sub=False):
        """侧栏扁平按钮（深底文字，hover 提亮）"""
        b = tk.Label(parent, text=text, bg=self.SB_BG, fg=self.SB_SUB if sub else self.SB_FG,
                     font=("微软雅黑", 9), anchor="w", padx=14, pady=5, cursor="hand2")
        b.pack(fill="x")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.config(bg=self.SB_HOV))
        b.bind("<Leave>", lambda e: b.config(bg=self.SB_BG))
        return b

    def _group_label(self, parent, text):
        tk.Label(parent, text=text, bg=self.SB_BG, fg=self.SB_SUB,
                 font=("微软雅黑", 7), anchor="w", padx=14, pady=(10, 2)
                 ).pack(fill="x")

    def _build_sidebar(self):
        # 品牌
        brand = tk.Frame(self.sidebar, bg=self.SB_BG)
        brand.pack(fill="x", pady=(12, 6))
        tk.Label(brand, text="✚  医学知识库", bg=self.SB_BG, fg="white",
                 font=("微软雅黑", 13, "bold")).pack(anchor="w", padx=14)
        tk.Label(brand, text="考研医学 AI 助手 · 教材/题库/PPT",
                 bg=self.SB_BG, fg=self.SB_SUB, font=("微软雅黑", 7)
                 ).pack(anchor="w", padx=14, pady=(1, 0))

        # 新建会话
        new_btn = tk.Label(self.sidebar, text="＋  新建会话", bg=self.SB_ACT, fg="white",
                           font=("微软雅黑", 9, "bold"), anchor="center", pady=7,
                           cursor="hand2")
        new_btn.pack(fill="x", padx=10, pady=(8, 4))
        new_btn.bind("<Button-1>", lambda e: self.new_session())
        new_btn.bind("<Enter>", lambda e: new_btn.config(bg="#1D4ED8"))
        new_btn.bind("<Leave>", lambda e: new_btn.config(bg=self.SB_ACT))

        # 会话列表（滚动）
        self._group_label(self.sidebar, "会  话")
        list_holder = tk.Frame(self.sidebar, bg=self.SB_BG)
        list_holder.pack(fill="both", expand=True, padx=(0, 0))
        self.sess_canvas = tk.Canvas(list_holder, bg=self.SB_BG,
                                     highlightthickness=0, bd=0)
        self.sess_inner = tk.Frame(self.sess_canvas, bg=self.SB_BG)
        self._sess_win = self.sess_canvas.create_window((0, 0), window=self.sess_inner,
                                                        anchor="nw")
        self.sess_canvas.pack(side="left", fill="both", expand=True)
        self.sess_inner.bind("<Configure>",
                             lambda e: self.sess_canvas.configure(
                                 scrollregion=self.sess_canvas.bbox("all")))
        self.sess_canvas.bind("<MouseWheel>",
                              lambda e: self.sess_canvas.yview_scroll(
                                  int(-e.delta / 120), "units"))

        # 知识库工具（固定底部）
        tools = tk.Frame(self.sidebar, bg=self.SB_BG)
        tools.pack(side="bottom", fill="x")
        self._group_label(tools, "知识库工具")
        self._nav_btn(tools, "📄  解析文档", lambda: self.run_task(parse_all_docs))
        self._nav_btn(tools, "📚  更新题库", self.update_question_bank)
        self._nav_btn(tools, "📝  导入真题", self.import_exam_paper)
        self._nav_btn(tools, "📋  生成试卷", self.generate_exam_paper)
        self._nav_btn(tools, "🔄  一键更新", self.full_update_knowledge_base)
        tk.Frame(tools, bg=self.SB_BG2, height=1).pack(fill="x", pady=4)
        self._nav_btn(tools, "🧹  清空缓存", self.clear_all_cache)
        # 侧栏底部小字（题库规模）
        self.side_info = tk.Label(tools, text="加载中…", bg=self.SB_BG, fg=self.SB_SUB,
                                  font=("微软雅黑", 7), anchor="w", padx=14, pady=6)
        self.side_info.pack(fill="x")

    def _session_title(self, sid):
        """会话标题：优先用第一条问题摘要"""
        fp = os.path.join(SAVE_DIR, f"session_{sid}.json")
        try:
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    hist = json.load(f)
                if hist and hist[0].get("question"):
                    q = hist[0]["question"].strip().replace("\n", " ")
                    return q[:16] + ("…" if len(q) > 16 else "")
        except Exception:
            pass
        return f"会话 {sid[:6]}"

    def _refresh_session_list(self, select_sid=None):
        """重建左侧会话列表"""
        for w in self.sess_inner.winfo_children():
            w.destroy()
        for cid in self.notebook.tabs():
            w = self.notebook.nametowidget(cid)
            sid = getattr(w, "session_id", "")
            title = self._session_title(sid)
            active = (sid == self._active_sid)
            row = tk.Label(self.sess_inner, text=title,
                           bg=self.SB_ACT if active else self.SB_BG,
                           fg="white" if active else "#CBD5E1",
                           font=("微软雅黑", 9), anchor="w", padx=12, pady=6,
                           cursor="hand2")
            row.pack(fill="x")
            row.bind("<Button-1>", lambda e, c=cid: self.select_session(c))
            row.bind("<Enter>", lambda e, r=row, a=active: r.config(
                bg=self.SB_ACT if a else self.SB_HOV))
            row.bind("<Leave>", lambda e, r=row, a=active: r.config(
                bg=self.SB_ACT if a else self.SB_BG))

    def select_session(self, child):
        """点击侧栏会话 -> 切换右侧聊天区"""
        w = self.notebook.nametowidget(child)
        self._active_sid = getattr(w, "session_id", "")
        self.notebook.select(child)
        self._refresh_session_list()
        self.status.config(text=f"📁 当前会话：{self._session_title(self._active_sid)}",
                           fg=SUB_TEXT)

    def _hide_all_tabs(self):
        """隐藏 notebook 标签页头部（会话切换交给侧栏）"""
        try:
            for i in range(len(self.notebook.tabs())):
                self.notebook.tab(i, state="hidden")
        except Exception:
            pass

    def clear_all_cache(self):
        clear_cache()
        messagebox.showinfo("提示", "问答缓存已清空")
        self.status.config(text="🧹 问答缓存已清空", fg=SUB_TEXT)

    def _load_bank_info(self):
        """后台统计题库规模并显示在状态栏与侧栏（只读 CSV，不编码）"""
        try:
            from question_bank import bank_stats
            st = bank_stats()
            self.status.config(
                text=f"●  混合 RAG 检索 · DeepSeek 生成 · 题库共 {st['file_total']} 题（含考研真题）· 就绪",
                fg=SUB_TEXT)
            try:
                self.side_info.config(text=f"题库：{st['file_total']} 题 · 会话：{len(self.notebook.tabs())}")
            except Exception:
                pass
        except Exception:
            pass

    def _preload_bge(self):
        try:
            get_bge_model()
            logger.info("✅ BGE模型已预加载完成")
        except Exception as e:
            logger.error(f"BGE模型预加载失败: {traceback.format_exc()}")

    def load_all_sessions(self):
        files = sorted(
            [f for f in os.listdir(SAVE_DIR)
             if f.startswith("session_") and f.endswith(".json")],
            key=lambda f: int(f[len("session_"):-len(".json")])
        )
        for f in files:
            self.session_counter += 1
            session_id = f[len("session_"):-len(".json")]
            self.add_tab(session_id, load_history=False)

    def new_session(self):
        self.session_counter += 1
        session_id = str(int(time.time()))
        self.add_tab(session_id)

    def add_tab(self, session_id, title=None, load_history=True):
        tab = SessionTab(self.notebook, self.notebook,
                         session_id, title or f"会话 {self.session_counter}", app=self)
        self.notebook.add(tab, text=tab.title)
        self._hide_all_tabs()
        self._active_sid = session_id
        self.notebook.select(tab)
        self._refresh_session_list()

    def refresh_sidebar(self):
        """会话有新问答后刷新侧栏标题/状态（由 SessionTab 触发）"""
        try:
            self._refresh_session_list()
            self.side_info.config(text=f"会话数：{len(self.notebook.tabs())}")
        except Exception:
            pass

    def pin_current(self):
        current = self.notebook.select()
        if current:
            tab = self.notebook.nametowidget(current)
            self.notebook.insert(0, child=tab)
            self.notebook.select(0)

    def run_task(self, func):
        threading.Thread(target=func, daemon=True).start()
        messagebox.showinfo("提示", "后台任务已启动")

    def generate_exam_paper(self):
        """从考试题库数据库随机抽取 100 题生成试卷（满 100 题才生成）"""
        self.status.config(text="正在检查题库并生成试卷，请稍候...", fg="#0E7490")

        def _do():
            try:
                result = generate_exam_paper()
                if result.get("success"):
                    messagebox.showinfo("试卷已生成", result["msg"])
                else:
                    messagebox.showwarning("暂无法生成试卷", result.get("msg", "未知错误"))
            except Exception as e:
                messagebox.showerror("生成失败", f"生成试卷出错：{e}")
            finally:
                self.status.config(text="●  混合 RAG 检索 · DeepSeek 生成 · 就绪", fg=SUB_TEXT)

        threading.Thread(target=_do, daemon=True).start()

    def update_question_bank(self):
        self.status.config(text="正在更新题库，请稍候...", fg="#FF8C00")
        current_tab = self.notebook.select()
        if current_tab:
            tab = self.notebook.nametowidget(current_tab)
            if hasattr(tab, 'status_label'):
                tab.status_label.config(text="更新中...", fg="#FF8C00")

        def _update():
            try:
                doc_cnt, pdf_cnt, skipped, failed, err = update_all()
                msg = (f"✅ 题库更新完成！\n\n"
                       f"📘 教材题库（docx）：{doc_cnt} 题\n"
                       f"📕 考研真题（pdf）：{pdf_cnt} 题\n"
                       f"⏭️ 扫描版跳过：{len(skipped)} 份\n"
                       f"❌ 解析失败：{len(failed)} 份")
                if failed:
                    msg += "\n\n失败清单：\n" + "\n".join(failed[:5])
                if skipped:
                    msg += ("\n\n提示：扫描版真题（2019/2020/2023 等）没有文字层，"
                            "无法自动解析，需要 OCR 才能纳入题库。")
                if err:
                    messagebox.showwarning("更新完成但有问题", msg + f"\n\n{err}")
                else:
                    messagebox.showinfo("更新完成", msg)
                init_bank()
                logger.info("✅ 题库已重新加载")
            except Exception as e:
                messagebox.showerror("更新失败", str(e))
            finally:
                self.status.config(text="●  混合 RAG 检索 · DeepSeek 生成 · 就绪", fg=SUB_TEXT)
                if current_tab:
                    tab = self.notebook.nametowidget(current_tab)
                    if hasattr(tab, 'status_label'):
                        tab.status_label.config(text="就绪", fg="#666666")

        threading.Thread(target=_update, daemon=True).start()

    def import_exam_paper(self):
        paper = filedialog.askopenfilename(
            title="选择试卷文件",
            filetypes=[("PDF", "*.pdf"), ("Word", "*.docx"), ("文本", "*.txt")]
        )
        if not paper:
            return
        answer = filedialog.askopenfilename(
            title="选择答案文件（可选，没有则跳过）",
            filetypes=[("PDF", "*.pdf"), ("Word", "*.docx"), ("文本", "*.txt")]
        )
        self.status.config(text="正在对齐导入真题...", fg="#9C27B0")

        def _do_import():
            try:
                from exam_aligner import import_exam_paper as do_import
                from question_bank import init_bank
                success, msg = do_import(paper, answer or None)
                if success:
                    init_bank()
                    messagebox.showinfo("真题导入完成", msg)
                else:
                    messagebox.showerror("导入失败", msg)
            except Exception as e:
                messagebox.showerror("导入异常", str(e))
            finally:
                self.status.config(text="●  混合 RAG 检索 · DeepSeek 生成 · 就绪", fg=SUB_TEXT)

        threading.Thread(target=_do_import, daemon=True).start()

    def full_update_knowledge_base(self):
        self.status.config(text="正在一键更新知识库，请稍候...", fg="#8B4513")
        current_tab = self.notebook.select()
        if current_tab:
            tab = self.notebook.nametowidget(current_tab)
            if hasattr(tab, 'status_label'):
                tab.status_label.config(text="更新中...", fg="#8B4513")

        def _do_update():
            try:
                success, msg = full_update()
                if success:
                    messagebox.showinfo("一键更新完成", msg)
                    clear_vector_db_cache()
                    logger.info("✅ 向量库缓存已清除")
                else:
                    messagebox.showwarning("一键更新失败", msg)
            except Exception as e:
                messagebox.showerror("一键更新异常", str(e))
            finally:
                self.status.config(text="●  混合 RAG 检索 · DeepSeek 生成 · 就绪", fg=SUB_TEXT)
                if current_tab:
                    tab = self.notebook.nametowidget(current_tab)
                    if hasattr(tab, 'status_label'):
                        tab.status_label.config(text="就绪", fg="#666666")

        threading.Thread(target=_do_update, daemon=True).start()

    def on_closing(self):
        # 关闭前自动做每日归档（当天数据存为 数据库_YYYY-MM-DD.json）
        try:
            path = daily_snapshot()
            if path:
                self.status.config(text=f"🗂️ 已归档: {os.path.basename(path)}", fg="#0E7490")
                self.root.update_idletasks()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MedApp(root)
    root.mainloop()
