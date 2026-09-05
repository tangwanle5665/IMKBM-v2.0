#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手机版 RAG 问答服务（直接利用 med_qa 缓存）
界面：现代移动聊天风格（气泡对话 + 渐变顶栏 + 底部输入栏）
"""
import re
from flask import Flask, request, jsonify, render_template_string

from med_qa import med_exam_chat_stream, get_cached_answer, _semantic_cache, clear_cache
from utils import logger

app = Flask(__name__)

# ---------- 会话历史（支持上下文关联提问） ----------
SESSIONS = {}          # session_id -> [{question, answer}, ...]
MAX_HISTORY = 30       # 每个会话最多保留 30 轮


def get_session_history(sid: str) -> list:
    return SESSIONS.setdefault(sid, [])


def push_session(sid: str, question: str, answer: str):
    hist = get_session_history(sid)
    hist.append({"question": question, "answer": answer[:2000]})
    if len(hist) > MAX_HISTORY:
        del hist[:len(hist) - MAX_HISTORY]


# ---------- HTML 页面（现代聊天界面） ----------
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="theme-color" content="#1E40AF">
<title>医学知识库 · 智能问答</title>
<style>
  :root {
    --bg:#EDF2F9; --card:#FFFFFF; --accent:#2563EB; --accent-dark:#1E40AF;
    --text:#1F2937; --sub:#64748B; --line:#E2E8F0;
    --ai-shadow:0 1px 4px rgba(15,23,42,.08);
    --user-shadow:0 2px 6px rgba(37,99,235,.28);
  }
  * { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html,body { height:100%; }
  body {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    background:var(--bg); display:flex; flex-direction:column; overflow:hidden;
  }
  /* ===== 顶部渐变标题栏 ===== */
  header {
    background:linear-gradient(135deg,#1E40AF 0%,#2563EB 55%,#3B82F6 100%);
    color:#fff; padding:16px 16px 14px; padding-top:calc(16px + env(safe-area-inset-top));
    box-shadow:0 2px 14px rgba(30,64,175,.28); position:relative; z-index:5;
  }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand .logo {
    width:38px; height:38px; border-radius:12px; background:rgba(255,255,255,.18);
    display:flex; align-items:center; justify-content:center; font-size:20px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.22);
  }
  .brand h1 { font-size:18px; font-weight:700; letter-spacing:.5px; }
  .brand p { font-size:11px; opacity:.85; margin-top:2px; letter-spacing:.3px; }
  .online {
    margin-left:auto; display:flex; align-items:center; gap:5px;
    font-size:11px; background:rgba(255,255,255,.16); padding:4px 10px;
    border-radius:20px; font-weight:600;
  }
  .online::before { content:''; width:7px; height:7px; border-radius:50%; background:#4ADE80; box-shadow:0 0 6px #4ADE80; }
  /* ===== 消息区 ===== */
  main { flex:1; overflow-y:auto; padding:16px 12px 10px; -webkit-overflow-scrolling:touch; }
  .welcome {
    text-align:center; color:var(--sub); font-size:12px; margin:6px auto 16px;
    background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:10px 14px; max-width:86%; line-height:1.7; box-shadow:var(--ai-shadow);
  }
  .msg { display:flex; margin-bottom:14px; animation:pop .22s ease; }
  .msg.user { justify-content:flex-end; }
  .msg.ai { justify-content:flex-start; }
  .avatar {
    width:32px; height:32px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:16px;
  }
  .msg.user .avatar { order:2; margin-left:8px; background:#DBEAFE; }
  .msg.ai  .avatar { margin-right:8px; background:linear-gradient(135deg,#1E40AF,#3B82F6); color:#fff; box-shadow:0 1px 4px rgba(37,99,235,.3); }
  .bubble {
    max-width:78%; padding:10px 14px; border-radius:16px; font-size:15px;
    line-height:1.7; white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;
  }
  .msg.user .bubble {
    background:var(--accent); color:#fff; border-bottom-right-radius:4px;
    box-shadow:var(--user-shadow);
  }
  .msg.ai .bubble {
    background:var(--card); color:var(--text); border-bottom-left-radius:4px;
    box-shadow:var(--ai-shadow); border:1px solid var(--line);
  }
  /* ===== 打字动画 ===== */
  .typing .bubble { display:flex; align-items:center; gap:4px; padding:14px 16px; }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); animation:blink 1.2s infinite; }
  .dot:nth-child(2) { animation-delay:.18s; }
  .dot:nth-child(3) { animation-delay:.36s; }
  @keyframes blink { 0%,80%,100% { opacity:.22; transform:translateY(0);} 40% { opacity:1; transform:translateY(-3px);} }
  @keyframes pop { from { opacity:0; transform:translateY(8px);} to { opacity:1; transform:none;} }
  /* ===== 底部输入栏 ===== */
  footer {
    background:var(--card); border-top:1px solid var(--line);
    padding:10px 12px; padding-bottom:calc(10px + env(safe-area-inset-bottom));
    display:flex; align-items:flex-end; gap:8px;
    box-shadow:0 -2px 10px rgba(15,23,42,.05);
  }
  #input {
    flex:1; border:1.5px solid var(--line); border-radius:22px; background:#F8FAFC;
    padding:10px 16px; font-size:15px; line-height:1.4; outline:none; resize:none;
    max-height:110px; font-family:inherit; color:var(--text);
    transition:border-color .2s, box-shadow .2s;
  }
  #input::placeholder { color:#94A3B8; }
  #input:focus { border-color:var(--accent); background:#fff; box-shadow:0 0 0 3px rgba(37,99,235,.13); }
  #send {
    width:42px; height:42px; border-radius:50%; border:none; cursor:pointer;
    background:linear-gradient(135deg,#2563EB,#1E40AF); color:#fff; font-size:17px;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 2px 8px rgba(37,99,235,.35); transition:transform .15s, opacity .2s;
  }
  #send:active { transform:scale(.9); }
  #send:disabled { opacity:.45; transform:none; }
  .hint { text-align:center; font-size:10px; color:#94A3B8; padding:2px 0 4px; background:var(--card); }
  /* ===== 回答内表格（体系性知识） ===== */
  .bubble table { border-collapse: collapse; margin:6px 0; width:100%; font-size:13px; }
  .bubble th, .bubble td { border:1px solid #CBD5E1; padding:4px 8px; text-align:left; }
  .bubble th { background:#F1F5F9; font-weight:600; }
  .bubble tr:nth-child(even) td { background:#FAFCFF; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo">🩺</div>
    <div>
      <h1>医学知识库</h1>
      <p>基于教材 · 题库 · 教学PPT 的 AI 问答</p>
    </div>
    <span class="online">在线</span>
  </div>
</header>

<main id="list"></main>

<footer>
  <textarea id="input" rows="1" placeholder="输入医学问题，回车发送…"></textarea>
  <button id="send" title="发送">➤</button>
</footer>
<div class="hint" style="display:flex;justify-content:center;align-items:center;gap:12px;">
  <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" id="optAnswer" checked> 出回答正文</label>
  <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" id="optFills" checked> 出填空题</label>
  <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" id="optReview" checked> 出知识点串联</label>
  <span id="optTip" style="color:#D97706;"></span>
</div>

<script>
(function(){
  // 兼容隐私模式：localStorage 可能被禁用
  var sid = '';
  try { sid = localStorage.getItem('med_sid') || ''; } catch(e) {}
  if(!sid){
    sid = 's' + Date.now().toString(36) + Math.random().toString(36).slice(2,8);
    try { localStorage.setItem('med_sid', sid); } catch(e) {}
  }
  var list = document.getElementById('list');
  var input = document.getElementById('input');
  var send = document.getElementById('send');
  var optAnswer = document.getElementById('optAnswer');
  var optFills = document.getElementById('optFills');
  var optReview = document.getElementById('optReview');
  var optTip = document.getElementById('optTip');

  // 开关状态持久化 + 联动提示
  try {
    optAnswer.checked = localStorage.getItem('med_optAnswer') !== '0';
    optFills.checked = localStorage.getItem('med_optFills') !== '0';
    optReview.checked = localStorage.getItem('med_optReview') !== '0';
  } catch(e) {}
  function syncOpt(){
    try {
      localStorage.setItem('med_optAnswer', optAnswer.checked ? '1' : '0');
      localStorage.setItem('med_optFills', optFills.checked ? '1' : '0');
      localStorage.setItem('med_optReview', optReview.checked ? '1' : '0');
    } catch(e) {}
    if(!optAnswer.checked && !optFills.checked){ optTip.textContent = '⚠ 回答和填空都关了'; }
    else if(!optAnswer.checked && optFills.checked){ optTip.textContent = '只出填空' + (optReview.checked ? '' : '·每档3道'); }
    else if(!optFills.checked){ optTip.textContent = '当前不出填空'; }
    else if(!optReview.checked){ optTip.textContent = '知识点关 → 每档3道'; }
    else { optTip.textContent = ''; }
  }
  optAnswer.addEventListener('change', syncOpt);
  optFills.addEventListener('change', syncOpt);
  optReview.addEventListener('change', syncOpt);
  syncOpt();

  function addMsg(role, text){
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    var av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = role === 'user' ? '🙂' : '🩺';
    var b = document.createElement('div');
    b.className = 'bubble';
    renderRich(b, text);
    wrap.appendChild(av);
    wrap.appendChild(b);
    list.appendChild(wrap);
    list.scrollTop = list.scrollHeight;
    return b;
  }
  /* 轻量渲染：Markdown 管道表格 -> HTML <table>；其余文本安全显示 */
  function renderRich(b, text){
    var lines = String(text || '').split('\\n');
    var buf = [], rows = null;
    function flushText(){ if(buf.length){ b.appendChild(document.createTextNode(buf.join('\\n'))); buf = []; } }
    function flushTable(){
      if(rows && rows.length){
        var tb = document.createElement('table');
        rows.forEach(function(r, i){
          var tr = document.createElement('tr');
          r.forEach(function(c){
            var cell = document.createElement(i === 0 ? 'th' : 'td');
            cell.textContent = c;
            tr.appendChild(cell);
          });
          tb.appendChild(tr);
        });
        b.appendChild(tb);
        rows = null;
      }
    }
    lines.forEach(function(line){
      var m = line.match(/^\s*\|(.*)\|\s*$/);
      if(m && m[1].indexOf('|') >= 0){
        flushText();
        var cells = m[1].split('|').map(function(s){ return s.trim(); });
        var isSep = cells.length > 0 && cells.every(function(c){ return /^:?-{2,}:?$/.test(c); });
        if(isSep){ return; }
        if(!rows){ rows = []; }
        rows.push(cells);
      } else {
        flushTable();
        buf.push(line);
      }
    });
    flushTable();
    flushText();
  }
  function addTyping(){
    var wrap = document.createElement('div');
    wrap.className = 'msg ai typing';
    var av = document.createElement('div'); av.className='avatar'; av.textContent='🩺';
    var b = document.createElement('div'); b.className='bubble';
    b.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    wrap.appendChild(av); wrap.appendChild(b);
    list.appendChild(wrap);
    list.scrollTop = list.scrollHeight;
    return wrap;
  }

  function setBusy(busy){
    send.disabled = busy;
    if(busy){ send.textContent = '…'; } else { send.textContent = '➤'; }
  }

  function finish(typing, text){
    if(typing && typing.parentNode){ typing.parentNode.removeChild(typing); }
    if(text){ addMsg('ai', text); }
    setBusy(false);
    input.focus();
  }

  function ask(){
    var q = input.value.trim();
    if(!q) return;
    input.value = '';
    autoGrow();
    addMsg('user', q);
    var typing = addTyping();
    setBusy(true);
    var xhr = new XMLHttpRequest();
    try {
      xhr.open('POST', '/ask');
    } catch(e) {
      finish(typing, '❌ 无法连接服务器：' + e.message);
      return;
    }
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.timeout = 90000;   // 首次问答可能较慢，最长等 90 秒
    xhr.onload = function(){
      var d = null;
      try { d = JSON.parse(xhr.responseText); } catch(e) {}
      if(d && d.status === 'success'){ finish(typing, d.answer); }
      else { finish(typing, '❌ ' + (d && d.message ? d.message : '服务器返回异常')); }
    };
    xhr.onerror = function(){ finish(typing, '❌ 网络错误：无法连接到服务器，请确认电脑端服务在运行'); };
    xhr.ontimeout = function(){ finish(typing, '⏳ 回答超时（90秒），请重试或换一个更具体的问题'); };
    xhr.send(JSON.stringify({question: q, session_id: sid,
                             enable_answer: optAnswer.checked,
                             enable_fills: optFills.checked,
                             enable_review: optReview.checked}));
  }

  function autoGrow(){
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 110) + 'px';
  }
  input.addEventListener('input', autoGrow);
  input.addEventListener('keydown', function(e){
    if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); ask(); }
  });
  send.addEventListener('click', ask);

  addMsg('ai', '你好！我是你的医学知识库助手 🩺\\n可以问我任何医学问题，例如：\\n· 高血压怎么治疗？\\n· 糖尿病的治疗药物有哪些？\\n· 电子传递链的过程是什么？');
  input.focus();
})();
</script>
</body>
</html>
"""

# ---------- 预热加载（提前加载模型和索引） ----------
def preload_models():
    """在服务启动时加载模型和索引，避免首次请求延迟"""
    logger.info("🔄 正在预热加载模型和索引...")
    try:
        from utils import get_bge_model, load_vector_db
        from retrieval_test import search_knowledge
        get_bge_model()
        data = load_vector_db()
        if data:
            logger.info(f"✅ 向量库加载完成，共 {len(data.get('chunks', []))} 个文本块")
        else:
            logger.warning("⚠️ 向量库未找到，请先运行 build_vector_db.py")
        # 预热一次检索
        search_knowledge("预热查询")
        logger.info("✅ 检索器预热完成")
    except Exception as e:
        logger.error(f"❌ 预热失败: {e}")

# ---------- Flask 路由 ----------
@app.route('/')
def index():
    resp = app.make_response(render_template_string(HTML_PAGE))
    # 禁止浏览器缓存页面，确保手机每次都能拿到最新界面（修复旧缓存导致的失效）
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/favicon.ico')
def favicon():
    """内联 SVG 图标（🩺），避免浏览器 404 请求"""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           '<rect width="100" height="100" rx="22" fill="#2563EB"/>'
           '<text x="50" y="72" font-size="62" text-anchor="middle">🩺</text></svg>')
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'status':'error', 'message':'问题不能为空'})

    sid = (data.get('session_id') or 'default')[:64]
    session_history = get_session_history(sid)

    # 1. 先查 med_qa 的精确缓存
    cached_answer, cached_image = get_cached_answer(question)
    if cached_answer is not None:
        logger.info(f"✅ 精确缓存命中: {question[:30]}...")
        push_session(sid, question, cached_answer)
        return jsonify({'status':'success', 'answer': cached_answer})

    # 2. 查 med_qa 的语义缓存
    semantic_answer, semantic_image = _semantic_cache.get(question)
    if semantic_answer is not None:
        logger.info(f"✅ 语义缓存命中: {question[:30]}...")
        push_session(sid, question, semantic_answer)
        return jsonify({'status':'success', 'answer': semantic_answer})

    # 3. 缓存未命中，调用完整 RAG 流程（带会话历史 + 输出开关）
    #    开关：填空题 / 知识点串联（关闭知识点时填空增至每档 3 道）
    try:
        enable_answer = bool(data.get('enable_answer', True))
        enable_fills = bool(data.get('enable_fills', True))
        enable_review = bool(data.get('enable_review', True))
        full_answer = ""
        for chunk in med_exam_chat_stream(question, session_history,
                                          enable_answer=enable_answer,
                                          enable_fills=enable_fills,
                                          enable_review=enable_review):
            full_answer += chunk
        clean_answer = re.sub(r'\[IMAGE:.*?\]', '', full_answer)
        # med_exam_chat_stream 内部已经保存了缓存，不需要再手动保存
        push_session(sid, question, clean_answer)
        return jsonify({'status':'success', 'answer': clean_answer})
    except Exception as e:
        logger.error(f"问答出错: {e}")
        return jsonify({'status':'error', 'message': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("医学知识库 Web 服务启动中（直接利用缓存）")
    print("=" * 50)
    preload_models()
    print("\n服务已就绪")
    print("请在手机浏览器访问: http://你的IP:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
