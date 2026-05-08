"""
在线对话处理脚本
从 heli_session 和 heli_message 表提取在线客服对话，生成 CSV 文件

逻辑：
1. 每个 SessionID 对应一个在线对话
2. VisitorId 是客户
3. Message 中：
   - FromUserName == VisitorId → 客户发的消息
   - ToUserName == VisitorId → 客服发给客户的消息
4. 过滤噪音：FromUserName != VisitorId 且 ToUserName != VisitorId → 系统消息，丢弃

使用方式：
  python process_online_chat.py              # 处理全部
  python process_online_chat.py 1000         # 处理前 1000 个 session
  python process_online_chat.py --start-date 2026-03-01  # 从指定日期开始
"""

import os
import sys
import sqlite3
import csv
import time
from datetime import datetime

# 配置
from config import DB_PATH, ONLINE_CHAT_CSV_DIR as OUTPUT_DIR


def fetch_sessions(limit=None, start_date=None):
    """从数据库获取有消息的 session 列表（按时间倒序，从最新开始）

    Args:
        limit: 期望生成的 CSV 数量，会多获取 session 以确保足够有效对话
        start_date: 开始日期
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 多获取 session 以确保有足够的有效对话（放大 5 倍）
    fetch_limit = limit * 5 if limit else None

    if start_date:
        query = """
            SELECT DISTINCT s.SessionID, s.VisitorId, s.AgentId, s.AgentName, s.BeginTime, s.EndTime, s.CusNickName
            FROM heli_session s
            JOIN heli_message m ON s.SessionID = m.SessionID
            WHERE s.SessionID IS NOT NULL
              AND s.VisitorId IS NOT NULL
              AND s.VisitorId <> ''
              AND s.BeginTime IS NOT NULL
              AND m.Content IS NOT NULL
              AND m.Content <> ''
              AND s.channelName LIKE '%宁惠保%'
              AND date(s.BeginTime) >= date(?)
            ORDER BY s.BeginTime DESC
        """
        if fetch_limit:
            query += f" LIMIT {fetch_limit}"
        cur.execute(query, (start_date,))
    else:
        query = """
            SELECT DISTINCT s.SessionID, s.VisitorId, s.AgentId, s.AgentName, s.BeginTime, s.EndTime, s.CusNickName
            FROM heli_session s
            JOIN heli_message m ON s.SessionID = m.SessionID
            WHERE s.SessionID IS NOT NULL
              AND s.VisitorId IS NOT NULL
              AND s.VisitorId <> ''
              AND s.BeginTime IS NOT NULL
              AND m.Content IS NOT NULL
              AND m.Content <> ''
              AND s.channelName LIKE '%宁惠保%'
            ORDER BY s.BeginTime DESC
        """
        if fetch_limit:
            query += f" LIMIT {fetch_limit}"
        cur.execute(query)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            'session_id': row[0],
            'visitor_id': row[1],
            'agent_id': row[2],
            'agent_name': row[3],
            'begin_time': row[4],
            'end_time': row[5],
            'cus_nick_name': row[6],
        }
        for row in rows
    ]


def fetch_messages_for_session(session_id):
    """获取某个 session 的所有消息"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
        SELECT FromUserName, ToUserName, Content, CreateTime, MsgType
        FROM heli_message
        WHERE SessionID = ?
        ORDER BY CreateTime ASC
    """
    cur.execute(query, (session_id,))
    rows = cur.fetchall()
    conn.close()

    return [
        {
            'from_user': row[0],
            'to_user': row[1],
            'content': row[2],
            'create_time': row[3],
            'msg_type': row[4],
        }
        for row in rows
    ]


def clean_content(content):
    """清理消息内容"""
    if not content:
        return ""

    # 去除 HTML 标签
    import re
    content = re.sub(r'<[^>]+>', '', content)

    # 去除多余的空白和换行
    content = content.strip()
    content = re.sub(r'\s+', ' ', content)

    # 去除常见的噪音内容
    noise_patterns = [
        r'客服.*?访客.*?欢迎您.*?',
        r'下拉查看历史消息.*?',
        r'历史记录.*?',
        r'&nbsp;',
    ]
    for pattern in noise_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    content = content.strip()
    return content


def process_session(session):
    """处理单个 session，返回对话列表"""
    visitor_id = session['visitor_id']
    messages = fetch_messages_for_session(session['session_id'])

    dialog_list = []

    for msg in messages:
        from_user = msg['from_user'] or ''
        to_user = msg['to_user'] or ''
        content = clean_content(msg['content'])

        if not content:
            continue

        # 判断发送者身份
        if from_user == visitor_id:
            # 客户发送的消息
            speaker = "客户"
        elif to_user == visitor_id:
            # 客服发给客户的消息，但需要判断是否是客户触发的问题
            # 如果内容是问句形式（包含？或?，且以疑问词开头或结尾），可能是客户点击选项触发
            if ('？' in content or '?' in content) and not content.startswith('Hi') and not content.startswith('您好'):
                # 看起来像是客户的问题（如"南京宁惠保的保障范围是什么？"）
                speaker = "客户"
            else:
                speaker = "客服"
        else:
            # 噪音：发送者和接收者都不是客户，跳过
            continue

        # 过滤系统消息
        if speaker == "客服":
            # 过滤一些明显的系统通知
            system_keywords = ['访客已离开', '坐席超时', '长时间未回复', '系统消息', '访客已离开！']
            if any(kw in content for kw in system_keywords):
                continue

        dialog_list.append((speaker, content, msg['create_time']))

    return dialog_list


def save_csv(session, dialog_list, output_dir):
    """保存对话到 CSV 文件（格式与 dialog_csv 一致）"""
    if not dialog_list:
        return False

    # 过滤只有机器人自动问候的无效对话
    # 条件：只有客服消息，没有客户消息；或者只有一条机器人问候
    has_customer_msg = any(speaker == "客户" for speaker, _, _ in dialog_list)
    if not has_customer_msg:
        return False

    # 过滤只有一条机器人问候的情况
    if len(dialog_list) == 1:
        content = dialog_list[0][1]
        robot_keywords = ['智能机器人', '很高兴为您服务']
        if any(kw in content for kw in robot_keywords):
            return False

    # 生成文件名：session_id + 开始时间
    begin_time = session['begin_time']
    if begin_time:
        time_str = begin_time.replace(':', '-').replace(' ', '_')
    else:
        time_str = 'unknown'

    filename = f"{time_str}_{session['session_id'][:8]}.csv"
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['对话者', '内容'])
        for speaker, content, create_time in dialog_list:
            writer.writerow([speaker, content])

    return True


def main(limit=None, start_date=None):
    """主处理流程"""
    print(f"\n{'='*60}")
    print("在线客服对话处理")
    print(f"{'='*60}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取 session 列表
    date_info = f"（从 {start_date} 开始）" if start_date else ""
    limit_info = f"前 {limit} 条" if limit else "全部"
    print(f"正在获取 {limit_info} session {date_info}...")

    sessions = fetch_sessions(limit, start_date)
    if not sessions:
        print("没有找到有效的 session")
        return

    total_sessions = len(sessions)
    print(f"获取到 {total_sessions} 个 session")
    print(f"时间范围: {sessions[0]['begin_time']} ~ {sessions[-1]['begin_time']}")

    # 处理每个 session
    print(f"\n开始处理...")
    t_start = time.time()

    success_count = 0
    empty_count = 0
    error_count = 0

    for i, session in enumerate(sessions, 1):
        try:
            dialog_list = process_session(session)
            session_id = session['session_id']

            if dialog_list:
                if save_csv(session, dialog_list, OUTPUT_DIR):
                    success_count += 1
                    print(f"  [{success_count}] ✓ SessionID: {session_id} | {len(dialog_list)} 条对话")
                else:
                    empty_count += 1
                    print(f"  ○ SessionID: {session_id} | 无有效对话（机器人问候）")
            else:
                empty_count += 1
                print(f"  ○ SessionID: {session_id} | 无有效对话")

            # 达到期望数量后停止
            if limit and success_count >= limit:
                print(f"  已生成 {success_count} 个 CSV，达到目标数量 {limit}")
                break

        except Exception as e:
            error_count += 1
            print(f"  ✗ SessionID: {session['session_id']} | 错误: {str(e)[:30]}")

    # 统计
    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print("处理完成统计")
    print(f"{'='*60}")
    print(f"处理 session:     {total_sessions} 个")
    print(f"生成 CSV:         {success_count} 个")
    print(f"空对话:           {empty_count} 个")
    print(f"错误:             {error_count} 个")
    print(f"总耗时:           {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
    print(f"输出目录:         {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='在线客服对话处理')
    parser.add_argument('limit', type=int, nargs='?', default=None,
                        help='处理的 session 数量，默认全部')
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')

    args = parser.parse_args()
    main(args.limit, args.start_date)