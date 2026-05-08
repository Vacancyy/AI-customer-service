"""
对话分析数据存储模块
将 CSV 分析结果存储到 MySQL 数据库
"""

import os
import pymysql
from datetime import datetime, date
from typing import Optional, List, Dict


# 数据库配置
DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}


def get_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def create_table():
    """创建 dialog_analysis 表"""
    sql = """
    CREATE TABLE IF NOT EXISTS dialog_analysis (
        id INT AUTO_INCREMENT PRIMARY KEY,
        source_type VARCHAR(20) COMMENT '数据来源：phone/online',
        source_file VARCHAR(255) COMMENT '原始文件路径',
        dialog_date DATE COMMENT '对话日期',
        primary_intent VARCHAR(20) COMMENT '一级意图分类',
        secondary_intent VARCHAR(50) COMMENT '二级意图分类',
        sentiment VARCHAR(20) COMMENT '情感倾向：满意/中立/不满',
        issue_type VARCHAR(100) COMMENT '问题类型',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
        INDEX idx_source_type (source_type),
        INDEX idx_dialog_date (dialog_date),
        INDEX idx_primary_intent (primary_intent),
        INDEX idx_sentiment (sentiment)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话分析结果表';
    """

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()
        print("表创建成功：dialog_analysis")
    finally:
        conn.close()


def extract_date_from_filename(filename: str) -> Optional[date]:
    """
    从文件名提取日期

    支持两种格式：
    - 电话录音：YYYYMMDD-HHMMSS_xxx.csv (如 20260117-125640_xxx.csv)
    - 在线对话：YYYY-MM-DD_HH-MM-SS_xxx.csv (如 2024-09-29_15-58-46_xxx.csv)

    Args:
        filename: 文件名或完整路径

    Returns:
        date 对象，无法提取时返回 None
    """
    # 只取文件名部分
    basename = os.path.basename(filename)

    # 匹配两种格式
    import re

    # 格式1: YYYY-MM-DD (在线对话)
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', basename)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    # 格式2: YYYYMMDD (电话录音)
    match = re.match(r'(\d{4})(\d{2})(\d{2})', basename)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def insert_analysis(
    source_type: str,
    source_file: str,
    primary_intent: str,
    secondary_intent: str,
    sentiment: str,
    issue_type: Optional[str] = None,
    dialog_date: Optional[date] = None
) -> int:
    """
    插入一条分析结果

    Args:
        source_type: 数据来源 (phone/online)
        source_file: 原始文件路径
        primary_intent: 一级意图分类
        secondary_intent: 二级意图分类
        sentiment: 情感倾向
        issue_type: 问题类型
        dialog_date: 对话日期（可选，默认从文件名提取）

    Returns:
        插入记录的 ID
    """
    # 如果没有提供日期，从文件名提取
    if dialog_date is None:
        dialog_date = extract_date_from_filename(source_file)

    sql = """
    INSERT INTO dialog_analysis
    (source_type, source_file, dialog_date, primary_intent, secondary_intent, sentiment, issue_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                source_type, source_file, dialog_date,
                primary_intent, secondary_intent, sentiment, issue_type
            ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def batch_insert(analysis_list: List[Dict]) -> int:
    """
    批量插入分析结果

    Args:
        analysis_list: 分析结果列表，每项包含：
            - source_type: str
            - source_file: str
            - primary_intent: str
            - secondary_intent: str
            - sentiment: str
            - issue_type: Optional[str]
            - dialog_date: Optional[date] (可选，默认从文件名提取)

    Returns:
        插入记录数
    """
    sql = """
    INSERT INTO dialog_analysis
    (source_type, source_file, dialog_date, primary_intent, secondary_intent, sentiment, issue_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = []
    for item in analysis_list:
        # 如果没有提供日期，从文件名提取
        dialog_date = item.get('dialog_date')
        if dialog_date is None:
            dialog_date = extract_date_from_filename(item['source_file'])

        values.append((
            item['source_type'],
            item['source_file'],
            dialog_date,
            item['primary_intent'],
            item['secondary_intent'],
            item['sentiment'],
            item.get('issue_type')
        ))

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
        conn.commit()
        return len(values)
    finally:
        conn.close()


def get_statistics() -> Dict:
    """获取统计信息"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 总数
            cursor.execute("SELECT COUNT(*) as total FROM dialog_analysis")
            total = cursor.fetchone()['total']

            # 按来源统计
            cursor.execute("""
                SELECT source_type, COUNT(*) as count
                FROM dialog_analysis
                GROUP BY source_type
            """)
            by_source = {row['source_type']: row['count'] for row in cursor.fetchall()}

            # 按一级意图统计
            cursor.execute("""
                SELECT primary_intent, COUNT(*) as count
                FROM dialog_analysis
                GROUP BY primary_intent
                ORDER BY count DESC
            """)
            by_intent = {row['primary_intent']: row['count'] for row in cursor.fetchall()}

            # 按情感统计
            cursor.execute("""
                SELECT sentiment, COUNT(*) as count
                FROM dialog_analysis
                GROUP BY sentiment
            """)
            by_sentiment = {row['sentiment']: row['count'] for row in cursor.fetchall()}

            return {
                'total': total,
                'by_source': by_source,
                'by_intent': by_intent,
                'by_sentiment': by_sentiment
            }
    finally:
        conn.close()


def check_file_exists(source_file: str) -> bool:
    """检查文件是否已处理过"""
    sql = "SELECT COUNT(*) as cnt FROM dialog_analysis WHERE source_file = %s"

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (source_file,))
            return cursor.fetchone()[0] > 0
    finally:
        conn.close()


if __name__ == '__main__':
    # 测试连接和建表
    try:
        print("测试数据库连接...")
        conn = get_connection()
        print("连接成功！")
        conn.close()

        print("\n创建表...")
        create_table()

        # 测试从文件名提取日期
        test_file = 'dialog_csv/20260117-125640_N000000036181_34122183_18061703386.csv'
        extracted_date = extract_date_from_filename(test_file)
        print(f"\n从文件名提取日期: {test_file} -> {extracted_date}")

        print("\n测试插入（自动提取日期）...")
        test_id = insert_analysis(
            source_type='phone',
            source_file=test_file,
            primary_intent='咨询',
            secondary_intent='产品了解',
            sentiment='满意',
            issue_type='保险范围咨询'
        )
        print(f"插入成功，ID: {test_id}")

        print("\n统计信息：")
        stats = get_statistics()
        print(f"总数: {stats['total']}")
        print(f"按来源: {stats['by_source']}")
        print(f"按意图: {stats['by_intent']}")
        print(f"按情感: {stats['by_sentiment']}")

    except Exception as e:
        print(f"错误: {e}")