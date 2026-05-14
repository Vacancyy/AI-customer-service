"""
南京宁惠保 AI客服 Web界面
启动: python web_demo.py
访问: http://localhost:8000
"""

import os
import sys
import json
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from ai_qa_system_v2 import ImprovedQASystem

app = FastAPI()

# 全局初始化（启动时只加载一次）
qa_system = None
# 注意：会话历史由前端JavaScript管理，后端不再存储
# 日志记录统一存入MySQL ai_chat_log表，支持多用户并发


class Question(BaseModel):
    question: str
    history: list = []  # 多轮对话历史


@app.on_event("startup")
def startup():
    global qa_system
    print("正在初始化AI客服系统...")
    qa_system = ImprovedQASystem()
    print("初始化完成！")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/ask")
async def ask(q: Question):
    if not q.question.strip():
        return {"answer": "", "error": "请输入问题"}

    start = time.time()
    # 支持多轮对话：传入历史
    answer = qa_system.get_answer(q.question.strip(), history=q.history)
    elapsed = time.time() - start

    # 获取检索信息
    matched = qa_system.search_knowledge(q.question.strip(), top_k=3)
    sources = []
    for item in matched:
        sources.append({
            "question": item['qa']['std_question'],
            "category": item['qa'].get('primary_category', ''),
            "score": f"{item['score']:.3f}",
        })

    # 日志已由ai_qa_system_v2写入MySQL，无需内存存储

    return {
        "answer": answer,
        "time": f"{elapsed:.1f}s",
        "sources": sources,
    }


@app.get("/logs")
async def get_logs(limit=50):
    """查看最近对话日志（从MySQL读取）"""
    try:
        import pymysql
        conn = pymysql.connect(
            host=ImprovedQASystem.DB_HOST, port=ImprovedQASystem.DB_PORT,
            user=ImprovedQASystem.DB_USER, password=ImprovedQASystem.DB_PASS,
            database=ImprovedQASystem.DB_NAME
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT id, question, answer, top1_score, confidence_level, model_name, response_time, created_at "
            "FROM ai_chat_log ORDER BY id DESC LIMIT %s", (limit,)
        )
        rows = cur.fetchall()
        conn.close()
        logs = []
        for r in rows:
            logs.append({
                "id": r[0], "question": r[1], "answer": r[2][:100] if r[2] else "",
                "top1_score": r[3], "confidence": r[4],
                "model": r[5], "time": f"{r[6]:.1f}s", "created_at": str(r[7])
            })
        return {"logs": logs}
    except Exception as e:
        return {"logs": [], "error": str(e)}


# ==================== 前端页面 ====================
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>南京宁惠保 AI客服 测试版</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, PingFang SC, Microsoft YaHei, sans-serif;
    background: #f0f2f5;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
.header {
    background: linear-gradient(135deg, #1a73e8, #4285f4);
    color: white;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.header h1 { font-size: 18px; font-weight: 600; }
.header .badge {
    background: rgba(255,255,255,0.2);
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 12px;
}
.new-chat-btn {
    margin-left: auto;
    padding: 6px 14px;
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 6px;
    background: rgba(255,255,255,0.1);
    color: white;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}
.new-chat-btn:hover { background: rgba(255,255,255,0.2); }
.disclaimer-bar {
    background: #fff3cd;
    color: #856404;
    padding: 8px 16px;
    font-size: 13px;
    text-align: center;
    border-bottom: 1px solid #ffeaa7;
}
.disclaimer-bar strong { color: #d63031; }
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    max-width: 800px;
    width: 100%;
    margin: 0 auto;
}
.msg {
    margin-bottom: 16px;
    display: flex;
    gap: 10px;
    animation: fadeIn 0.3s;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.msg.user { justify-content: flex-end; }
.msg .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.msg.ai .avatar { background: #e8f0fe; color: #1a73e8; }
.msg.user .avatar { background: #e8f5e9; color: #2e7d32; }
.bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.8;
    font-size: 14px;
    word-break: break-word;
    white-space: pre-wrap;
}
.msg.ai .bubble { background: white; color: #333; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.msg.user .bubble { background: #1a73e8; color: white; }
.sources {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #eee;
    font-size: 12px;
    color: #888;
}
.sources span {
    display: inline-block;
    background: #f5f5f5;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 4px 2px 0;
}
.typing {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
}
.typing span {
    width: 8px; height: 8px;
    background: #bbb;
    border-radius: 50%;
    animation: bounce 1.4s infinite both;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0.6); } 40% { transform: scale(1); } }
.input-area {
    padding: 16px;
    background: white;
    border-top: 1px solid #e0e0e0;
    max-width: 800px;
    width: 100%;
    margin: 0 auto;
}
.privacy-note {
    font-size: 12px;
    color: #999;
    text-align: center;
    margin-bottom: 6px;
}
.input-wrap {
    display: flex;
    gap: 10px;
    align-items: flex-end;
}
.input-wrap textarea {
    flex: 1;
    padding: 10px 14px;
    border: 2px solid #e0e0e0;
    border-radius: 20px;
    font-size: 14px;
    resize: none;
    outline: none;
    font-family: inherit;
    max-height: 120px;
    transition: border-color 0.2s;
}
.input-wrap textarea:focus { border-color: #1a73e8; }
.input-wrap button {
    width: 44px; height: 44px;
    border-radius: 50%;
    border: none;
    background: #1a73e8;
    color: white;
    font-size: 18px;
    cursor: pointer;
    transition: background 0.2s;
    flex-shrink: 0;
}
.input-wrap button:hover { background: #1557b0; }
.input-wrap button:disabled { background: #ccc; cursor: not-allowed; }
.quick-q {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.quick-q button {
    padding: 6px 14px;
    border: 1px solid #e0e0e0;
    border-radius: 16px;
    background: white;
    color: #555;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}
.quick-q button:hover { border-color: #1a73e8; color: #1a73e8; background: #e8f0fe; }
.welcome { text-align: center; padding: 60px 20px; color: #888; }
.welcome h2 { color: #333; margin-bottom: 12px; }
</style>
</head>
<body>

<div class="header">
    <h1>南京宁惠保 AI客服</h1>
    <span class="badge">测试版</span>
    <button type="button" id="newChatBtn" class="new-chat-btn">新对话</button>
</div>

<div class="disclaimer-bar">
    <strong>AI回答仅供参考</strong>，具体保障责任以保险条款为准，如有疑问请咨询人工客服 4000040181
</div>

<div class="chat-container" id="chat">
    <div class="welcome">
        <h2>欢迎使用AI客服测试</h2>
        <p>请输入您的问题，或点击下方快捷问题开始</p>
    </div>
</div>

<div class="input-area">
    <div class="privacy-note">请勿在此输入身份证号、病历等敏感信息</div>
    <div class="quick-q">
        <button type="button" data-q="理赔需要哪些材料">理赔材料</button>
        <button type="button" data-q="理赔门槛是多少">理赔门槛</button>
        <button type="button" data-q="如何退保">退保流程</button>
        <button type="button" data-q="保障范围有哪些">保障范围</button>
        <button type="button" data-q="保费多少钱">保费多少</button>
        <button type="button" data-q="如何投保">如何投保</button>
    </div>
    <div class="input-wrap">
        <textarea id="input" rows="1" placeholder="请输入您的问题..."></textarea>
        <button type="button" id="sendBtn">➤</button>
    </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');

// 会话历史（保存最近5轮对话）
let sessionHistory = [];

// 新对话按钮
newChatBtn.addEventListener('click', () => {
    sessionHistory = [];
    chat.innerHTML = '<div class="welcome"><h2>欢迎使用AI客服测试</h2><p>请输入您的问题，或点击下方快捷问题开始</p></div>';
    input.focus();
});

// 快捷按钮事件
document.querySelector('.quick-q').addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON' && e.target.dataset.q) {
        input.value = e.target.dataset.q;
        send();
    }
});

// 发送按钮事件
sendBtn.addEventListener('click', send);

// 自动调整textarea高度
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

function addMsg(role, html) {
    // 移除欢迎语
    const welcome = chat.querySelector('.welcome');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `msg ${role}`;
    const avatarText = role === 'ai' ? 'AI' : '我';
    div.innerHTML = `<div class="avatar">${avatarText}</div><div class="bubble">${html}</div>`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'msg ai';
    div.id = 'typing';
    div.innerHTML = '<div class="avatar">AI</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
}

async function send() {
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    addMsg('user', escapeHtml(question));
    showTyping();

    try {
        // 发送历史对话（最近5轮）
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question,
                history: sessionHistory.slice(-10)  // 最近5轮（每轮2条）
            }),
        });
        const data = await res.json();
        hideTyping();

        let html = escapeHtml(data.answer)
            .replace(/【/g, '<b>【').replace(/】/g, '】</b>')
            .replace(/①/g, '<br>①').replace(/②/g, '<br>②').replace(/③/g, '<br>③')
            .replace(/④/g, '<br>④').replace(/⑤/g, '<br>⑤').replace(/⑥/g, '<br>⑥');

        // 检索来源（测试版显示，便于验证）
        if (data.sources && data.sources.length > 0) {
            html += '<div class="sources">';
            html += '参考: ';
            data.sources.forEach(s => {
                html += `<span>${escapeHtml(s.question)}</span>`;
            });
            html += ` | 响应 ${data.time}`;
            html += '</div>';
        }

        addMsg('ai', html);

        // 保存到会话历史
        sessionHistory.push({question: question, answer: data.answer});
        // 限制历史长度（最近5轮）
        if (sessionHistory.length > 10) {
            sessionHistory = sessionHistory.slice(-10);
        }
    } catch (e) {
        hideTyping();
        addMsg('ai', '网络错误，请重试');
    }

    sendBtn.disabled = false;
    input.focus();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

input.focus();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print(f"\n{'='*50}")
    print(f"  南京宁惠保 AI客服 测试版")
    print(f"  访问: http://localhost:{port}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host=host, port=port)
