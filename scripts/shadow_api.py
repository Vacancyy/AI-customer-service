"""
影子测试API服务
供合力客服系统调用，记录客户问题和AI回答，但不展示给用户

启动方式：
python shadow_api.py

API接口：
- POST /shadow/question  : 接收客户问题
- POST /shadow/answer    : 接收人工客服回答
- GET  /shadow/stats     : 查看统计
- GET  /shadow/export    : 导出对比报告
"""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.ai_qa_system_v2 import ImprovedQASystem, is_personal_query

# 设置环境变量
os.environ.setdefault('DASHSCOPE_API_KEY', os.environ.get('DASHSCOPE_API_KEY', ''))
os.environ.setdefault('DB_HOST', os.environ.get('DB_HOST', 'localhost'))
os.environ.setdefault('DB_PORT', os.environ.get('DB_PORT', '3308'))
os.environ.setdefault('DB_USER', os.environ.get('DB_USER', 'root'))
os.environ.setdefault('DB_PASS', os.environ.get('DB_PASS', ''))
os.environ.setdefault('DB_NAME', os.environ.get('DB_NAME', 'ai_customer_service'))

app = FastAPI(title="AI客服影子测试API")

# 全局变量
qa_system = None
shadow_db_path = os.path.join(os.path.dirname(__file__), '..', '01_source', 'shadow_realtime.db')


class QuestionRequest(BaseModel):
    """客户问题请求"""
    session_id: str          # 会话ID
    visitor_id: str          # 客户ID
    question: str            # 客户问题内容
    question_time: str = ""  # 问题时间（可选）
    channel: str = ""        # 渠道（可选）


class AnswerRequest(BaseModel):
    """人工客服回答请求"""
    session_id: str          # 会话ID
    visitor_id: str          # 客户ID
    agent_id: str            # 客服ID
    agent_name: str          # 客服名称
    answer: str              # 客服回答内容
    answer_time: str = ""    # 回答时间（可选）


@app.on_event("startup")
def startup():
    """启动时初始化"""
    global qa_system
    print("正在初始化AI客服系统...")
    qa_system = ImprovedQASystem()
    _init_shadow_db()
    print("影子测试API启动完成！")
    print(f"日志数据库: {shadow_db_path}")


def _init_shadow_db():
    """初始化影子测试数据库"""
    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS shadow_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        visitor_id TEXT,
        customer_question TEXT,
        question_time TEXT,

        -- AI回答
        ai_answer TEXT,
        ai_match_score REAL,
        ai_match_question TEXT,
        ai_category TEXT,
        ai_process_time REAL,

        -- 人工回答（后续补充）
        agent_id TEXT,
        agent_name TEXT,
        agent_answer TEXT,
        agent_answer_time TEXT,
        agent_response_seconds REAL,

        -- 状态
        has_agent_answer INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()


@app.post("/shadow/question")
async def receive_question(req: QuestionRequest):
    """
    接收客户问题（合力客服系统调用）

    功能：
    - 记录客户问题
    - AI生成回答（不返回给合力）
    - 后台存储，等待人工回答补充

    返回：
    - success: 是否成功
    - ai_generated: AI已生成回答（影子模式）
    - 注意：不返回AI回答内容！
    """
    if not req.question.strip():
        return {"success": False, "error": "问题内容为空"}

    start_time = time.time()

    # AI处理
    matched = qa_system.search_knowledge(req.question, top_k=1)
    score = matched[0]['score'] if matched else 0
    matched_q = matched[0]['qa']['std_question'] if matched else '无匹配'

    is_personal = is_personal_query(req.question)
    ai_answer = qa_system.get_answer(req.question)

    # 分类
    if is_personal:
        ai_category = 'personal_query'
    elif score < 0.5:
        ai_category = 'low_confidence'
    elif score >= 0.7:
        ai_category = 'high_confidence'
    else:
        ai_category = 'medium_confidence'

    process_time = time.time() - start_time

    # 保存到数据库
    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()

    question_time = req.question_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
    INSERT INTO shadow_log (
        session_id, visitor_id, customer_question, question_time,
        ai_answer, ai_match_score, ai_match_question, ai_category, ai_process_time,
        has_agent_answer
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (
        req.session_id, req.visitor_id, req.question, question_time,
        ai_answer, score, matched_q, ai_category, process_time
    ))

    log_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # 返回结果（不返回AI回答！）
    return {
        "success": True,
        "log_id": log_id,
        "ai_generated": True,
        "ai_category": ai_category,
        "ai_match_score": round(score, 3),
        "process_time": round(process_time, 2),
        "message": "影子测试：AI回答已生成并记录，未展示给用户"
    }


@app.post("/shadow/answer")
async def receive_answer(req: AnswerRequest):
    """
    接收人工客服回答（合力客服系统调用）

    功能：
    - 补充人工客服回答到已有记录
    - 计算人工响应时间

    返回：
    - success: 是否成功
    - matched: 是否找到对应的问题记录
    """
    if not req.answer.strip():
        return {"success": False, "error": "回答内容为空"}

    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()

    # 查找该会话中最近的未匹配人工回答的客户问题
    cursor.execute('''
    SELECT id, question_time FROM shadow_log
    WHERE session_id = ? AND visitor_id = ? AND has_agent_answer = 0
    ORDER BY id DESC LIMIT 1
    ''', (req.session_id, req.visitor_id))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "error": "未找到对应的客户问题记录"}

    log_id, question_time_str = row

    # 计算响应时间
    answer_time = req.answer_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        q_time = datetime.strptime(question_time_str, '%Y-%m-%d %H:%M:%S')
        a_time = datetime.strptime(answer_time, '%Y-%m-%d %H:%M:%S')
        response_seconds = (a_time - q_time).total_seconds()
    except:
        response_seconds = 0

    # 更新记录
    cursor.execute('''
    UPDATE shadow_log SET
        agent_id = ?,
        agent_name = ?,
        agent_answer = ?,
        agent_answer_time = ?,
        agent_response_seconds = ?,
        has_agent_answer = 1
    WHERE id = ?
    ''', (req.agent_id, req.agent_name, req.answer, answer_time, response_seconds, log_id))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "log_id": log_id,
        "response_seconds": round(response_seconds, 1),
        "message": "人工回答已记录，可进行对比分析"
    }


@app.get("/shadow/stats")
async def get_stats():
    """
    查看影子测试统计

    返回：
    - 总记录数
    - 有人工回答的记录数
    - AI分类统计
    """
    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()

    # 总记录
    cursor.execute('SELECT COUNT(*) FROM shadow_log')
    total = cursor.fetchone()[0]

    # 有人工回答的
    cursor.execute('SELECT COUNT(*) FROM shadow_log WHERE has_agent_answer = 1')
    with_answer = cursor.fetchone()[0]

    # AI分类统计
    cursor.execute('''
    SELECT ai_category, COUNT(*) as count, AVG(ai_match_score) as avg_score, AVG(ai_process_time) as avg_time
    FROM shadow_log GROUP BY ai_category
    ''')
    categories = cursor.fetchall()

    # 人工响应时间
    cursor.execute('SELECT AVG(agent_response_seconds) FROM shadow_log WHERE has_agent_answer = 1 AND agent_response_seconds > 0')
    avg_agent_time = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "total_questions": total,
        "with_agent_answer": with_answer,
        "avg_agent_response_time": round(avg_agent_time, 1),
        "categories": [
            {
                "category": cat,
                "count": count,
                "avg_score": round(avg_score, 3),
                "avg_ai_time": round(avg_time, 2)
            }
            for cat, count, avg_score, avg_time in categories
        ]
    }


@app.get("/shadow/export")
async def export_report():
    """
    导出对比报告（Markdown格式）

    返回：
    - report_path: 报告文件路径
    - summary: 报告摘要
    """
    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM shadow_log WHERE has_agent_answer = 1')
    total = cursor.fetchone()[0]

    if total == 0:
        return {"success": False, "error": "暂无完整对话记录，请等待更多数据"}

    # 获取完整对话记录
    cursor.execute('''
    SELECT
        customer_question,
        ai_answer,
        ai_match_score,
        ai_match_question,
        ai_category,
        ai_process_time,
        agent_answer,
        agent_name,
        agent_response_seconds
    FROM shadow_log
    WHERE has_agent_answer = 1
    ORDER BY id DESC
    ''')
    records = cursor.fetchall()

    # 分类统计
    cursor.execute('''
    SELECT ai_category, COUNT(*)
    FROM shadow_log WHERE has_agent_answer = 1
    GROUP BY ai_category
    ''')
    categories = cursor.fetchall()

    conn.close()

    # 生成报告
    report_path = os.path.join(os.path.dirname(__file__), '..', '05_analyze', 'reports', '影子测试对比报告.md')
    report = _build_report(total, categories, records)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # 计算核心指标
    ai_answered = sum(c for cat, c in categories if cat in ['high_confidence', 'medium_confidence'])

    return {
        "success": True,
        "report_path": report_path,
        "summary": {
            "total_records": total,
            "ai_answered_ratio": round(ai_answered / total * 100, 1),
            "avg_ai_time": round(sum(r[5] for r in records) / len(records), 2),
            "avg_agent_time": round(sum(r[8] for r in records if r[8] > 0) / len([r for r in records if r[8] > 0]), 1)
        }
    }


def _build_report(total, categories, records):
    """构建对比报告"""
    report = f"""# AI客服影子测试对比报告

## 测试概况

- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试模式**: 影子测试（AI回答不展示给用户）
- **完整对话数**: {total} 条（有AI+人工双方回答）

---

## 一、核心指标对比

### 1.1 AI vs 人工响应时间

| 指标 | AI | 人工 |
|------|-----|------|
| 平均响应时间 | {sum(r[5] for r in records)/len(records):.1f}秒 | {sum(r[8] for r in records if r[8]>0)/len([r for r in records if r[8]>0]):.1f}秒 |
| 响应速度优势 | AI更快 | - |

### 1.2 AI处理能力

| 分类 | 数量 | 占比 |
|------|------|------|
"""
    for cat, count in categories:
        report += f"| {cat} | {count} | {count/total*100:.1f}% |\n"

    ai_answered = sum(c for cat, c in categories if cat in ['high_confidence', 'medium_confidence'])
    report += f"""

**AI可独立回答比例**: {ai_answered}/{total} = {ai_answered/total*100:.1f}%

---

## 二、典型案例对比

### 2.1 高置信度案例

"""
    high_cases = [r for r in records if r[4] == 'high_confidence'][:5]
    for i, r in enumerate(high_cases, 1):
        report += f"""
**案例{i}**
- 客户问题: {r[0]}
- AI匹配: [{r[2]:.2f}] {r[3]}
- AI回答: {r[1][:200]}
- 人工回答: {r[6][:200]}
- 客服: {r[7]}
"""
    report += """

---

## 三、接入说明

本报告基于影子测试数据生成。

影子测试API接口：
- POST /shadow/question - 接收客户问题
- POST /shadow/answer - 接收人工回答
- GET /shadow/stats - 查看统计
- GET /shadow/export - 导出报告

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return report


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("AI客服影子测试API")
    print("=" * 60)
    print()
    print("API接口：")
    print("  POST /shadow/question  - 接收客户问题（合力调用）")
    print("  POST /shadow/answer    - 接收人工回答（合力调用）")
    print("  GET  /shadow/stats     - 查看统计")
    print("  GET  /shadow/export    - 导出对比报告")
    print()
    print("启动地址: http://0.0.0.0:8001")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)