#!/usr/bin/env python3
"""
从heli.sqlite3数据库提取高质量QA对
"""

import sqlite3
import json
import re
from collections import defaultdict

DB_PATH = "/home/xiecheng/customer-service/01_source/heli.sqlite3"
OUTPUT_PATH = "/home/xiecheng/customer-service/05_analyze/reports/heli_extracted_qa.json"

# 业务关键词（高质量回答的特征）
QUALITY_KEYWORDS = [
    "公众号", "我的南京", "支付宝", "微信",
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "点击", "登录", "下载", "关注", "绑定", "注册",
    "上传", "提交", "申请", "下载", "扫描", "识别",
    "菜单", "中心", "首页", "页面", "链接",
    "步骤", "流程", "操作", "方法", "方式",
    "查询", "缴费", "报销", "理赔", "参保",
    "身份证", "医保卡", "社保卡", "银行卡",
    "保单", "保险", "保费", "投保", "退保",
    "免赔额", "赔付", "报销比例"
]

# 问题业务关键词（确保问题是关于业务的）
QUESTION_KEYWORDS = [
    "理赔", "报销", "保单", "保费", "保险", "参保", "投保", "退保",
    "缴费", "续保", "查询", "我的南京", "支付宝", "微信", "公众号",
    "怎么", "如何", "什么", "为什么", "哪里", "能否", "可以", "是否",
    "多少", "什么时候", "多久", "为什么", "能不能", "行不行",
    "材料", "证明", "发票", "出院", "住院", "门诊", "医保",
    "身份证", "银行卡", "账号", "密码", "登录",
    "免赔额", "赔付", "比例", "金额", "费用",
    "特药", "药房", "药", "治疗", "医院",
    "时间", "期限", "到期", "生效"
]

# 无效回答关键词
INVALID_KEYWORDS = [
    "转人工", "智能机器人", "自动回复",
    "访客已离开", "长时间未回复", "坐席超时",
    "客服正在努力处理", "请稍等", "稍等",
    "欢迎您", "有什么可以帮您", "欢迎咨询"
]

# 相似问题模式（用于合并）
SIMILAR_PATTERNS = [
    (r"[\?？。！，,、\s]+$", ""),  # 移除末尾标点
    (r"^[\?？。！，,、\s]+", ""),  # 移除开头标点
    (r"[【\[].*?[\]】]", ""),  # 移除方括号内容
]

def clean_content(content):
    """清理消息内容"""
    if not content:
        return ""
    # 移除HTML标签
    content = re.sub(r'<[^>]+>', '', content)
    # 移除特殊空白字符
    content = re.sub(r'&nbsp;|&ensp;|&emsp;', ' ', content)
    # 移除多余空白
    content = re.sub(r'\s+', ' ', content)
    return content.strip()

def is_quality_answer(answer):
    """判断是否是高质量回答"""
    if not answer:
        return False

    length = len(answer)
    # 放宽长度限制到30-600字
    if length < 30 or length > 600:
        return False

    # 检查是否包含无效关键词
    for kw in INVALID_KEYWORDS:
        if kw in answer:
            return False

    # 检查是否包含质量关键词
    for kw in QUALITY_KEYWORDS:
        if kw in answer:
            return True

    return False

def is_valid_question(question):
    """判断是否是有效问题"""
    if not question:
        return False

    question = question.strip()
    length = len(question)

    # 过滤太短或太长的
    if length < 8 or length > 200:
        return False

    # 过滤纯数字、纯特殊字符开头的问题
    if re.match(r'^[\d\s\-_,\.，。、]+$', question):
        return False
    if re.match(r'^[\W_]+$', question[:3]):  # 前三个字符都是特殊字符
        return False

    # 过滤以特殊字符开头的问题
    if question.startswith(('!', '-', '_', '.', ',', '?', '？', '。', '/', '\\', '&', '*')):
        return False

    # 过滤系统消息和欢迎语
    invalid_patterns = [
        "欢迎", "您好，请咨询", "有什么可以帮您",
        "访客", "客服", "坐席", "欢迎您", "自动回复",
        "请于", "扫码确认",  # 系统消息
        "!请", "&lt;", "---",  # 特殊格式
    ]
    for pattern in invalid_patterns:
        if pattern in question:
            return False

    # 问题必须包含至少一个业务关键词
    has_keyword = False
    for kw in QUESTION_KEYWORDS:
        if kw in question:
            has_keyword = True
            break

    if not has_keyword:
        return False

    # 问题必须包含中文字符
    if not re.search(r'[\u4e00-\u9fff]', question):
        return False

    return True

def normalize_question(question):
    """标准化问题文本，用于合并相似问题"""
    q = question.strip()
    for pattern, replacement in SIMILAR_PATTERNS:
        q = re.sub(pattern, replacement, q)
    return q.strip()

def merge_similar_questions(qa_list):
    """合并相似问题，保留最佳回答"""
    # 按标准化问题分组
    groups = defaultdict(list)
    for qa in qa_list:
        normalized = normalize_question(qa['question'])
        groups[normalized].append(qa)

    # 从每组中选择最佳QA
    merged = []
    for normalized_q, group in groups.items():
        # 选择回答最长的
        best = max(group, key=lambda x: len(x['answer']))
        # 选择最原始的问题形式
        best_question = min(group, key=lambda x: len(x['question']))
        merged.append({
            'question': best_question['question'],
            'answer': best['answer']
        })

    return merged

def extract_qa_pairs():
    """提取QA对"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 获取有客服接待的会话及其VisitorId
    print("正在获取有客服接待的会话...")
    cursor.execute("""
        SELECT SessionID, VisitorId, AgentName
        FROM heli_session
        WHERE AgentName IS NOT NULL AND AgentName != ''
    """)
    sessions = cursor.fetchall()
    print(f"找到 {len(sessions)} 个有客服接待的会话")

    # 构建SessionID -> VisitorId的映射
    session_visitor = {row[0]: row[1] for row in sessions}
    session_ids = list(session_visitor.keys())

    # 2. 批量获取消息
    print("正在获取消息...")
    all_messages = []
    batch_size = 10000

    for i in range(0, len(session_ids), batch_size):
        batch = session_ids[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cursor.execute(f"""
            SELECT SessionID, FromUserName, ToUserName, Content, CreateTime
            FROM heli_message
            WHERE SessionID IN ({placeholders})
            ORDER BY SessionID, CreateTime
        """, batch)
        all_messages.extend(cursor.fetchall())
        if (i // batch_size + 1) % 5 == 0:
            print(f"  已处理 {min(i + batch_size, len(session_ids))}/{len(session_ids)} 个会话的消息")

    print(f"共获取 {len(all_messages)} 条消息")

    # 3. 按会话分组消息
    session_messages = defaultdict(list)
    for msg in all_messages:
        session_id, from_user, to_user, content, create_time = msg
        session_messages[session_id].append({
            'from': from_user,
            'to': to_user,
            'content': clean_content(content),
            'time': create_time
        })

    # 4. 提取QA对
    print("正在提取QA对...")
    qa_pairs = []

    for session_id, visitor_id in session_visitor.items():
        messages = session_messages.get(session_id, [])
        if not messages:
            continue

        # 找出访客消息和客服消息
        visitor_msgs = []
        agent_msgs = []

        for msg in messages:
            # 访客发的消息（FromUserName是VisitorId）
            if msg['from'] == visitor_id:
                visitor_msgs.append(msg)
            # 客服发给访客的消息（ToUserName是VisitorId，且不是系统消息）
            elif msg['to'] == visitor_id:
                agent_msgs.append(msg)

        # 配对问题和回答
        # 找到访客问题和紧随其后的客服回答
        for i, v_msg in enumerate(visitor_msgs):
            question = v_msg['content']
            if not is_valid_question(question):
                continue

            # 找这个问题之后最近的客服回答
            v_time = v_msg['time']
            best_answer = None

            for a_msg in agent_msgs:
                if a_msg['time'] > v_time:
                    answer = a_msg['content']
                    if is_quality_answer(answer):
                        best_answer = answer
                        break

            if best_answer:
                qa_pairs.append({
                    'question': question,
                    'answer': best_answer,
                    'session_id': session_id
                })

    print(f"初步提取到 {len(qa_pairs)} 个QA对")

    # 5. 去重
    print("正在进行去重处理...")

    # 按问题文本去重，保留回答最长的
    qa_dict = {}
    for qa in qa_pairs:
        q = qa['question']
        if q not in qa_dict or len(qa['answer']) > len(qa_dict[q]['answer']):
            qa_dict[q] = qa

    unique_qa = list(qa_dict.values())
    print(f"去重后剩余 {len(unique_qa)} 个QA对")

    # 6. 相似问题合并
    print("正在合并相似问题...")
    merged_qa = merge_similar_questions(unique_qa)
    print(f"合并后剩余 {len(merged_qa)} 个QA对")

    # 按问题排序
    merged_qa.sort(key=lambda x: x['question'])

    conn.close()

    return merged_qa

def main():
    print("=" * 60)
    print("从heli.sqlite3提取高质量QA对")
    print("=" * 60)

    qa_pairs = extract_qa_pairs()

    # 保存结果
    print(f"\n保存结果到: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 共提取 {len(qa_pairs)} 个高质量QA对")

    # 显示一些示例
    print("\n示例QA对:")
    for i, qa in enumerate(qa_pairs[:5]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"问: {qa['question'][:100]}..." if len(qa['question']) > 100 else f"问: {qa['question']}")
        print(f"答: {qa['answer'][:100]}..." if len(qa['answer']) > 100 else f"答: {qa['answer']}")

if __name__ == "__main__":
    main()