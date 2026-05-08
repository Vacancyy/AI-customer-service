"""
投诉类深度分析脚本 - HTML版本

专门针对投诉类对话进行深度分析，输出为HTML格式：
1. 投诉原因细分（理赔时效、拒赔、金额异议、服务态度等）
2. 投诉涉及的理赔阶段分析
3. 投诉客户情绪轨迹分析
4. 投诉处理效果评估
5. 高频投诉问题及客服应对策略
6. 优化方案和建议

使用方式：
  python scripts/analyze_complaint.py --type online   # 分析在线对话投诉（从数据库）
  python scripts/analyze_complaint.py --type phone    # 分析电话录音投诉（从数据库）
  python scripts/analyze_complaint.py --type all      # 分析所有投诉（从数据库）
  python scripts/analyze_complaint.py --limit 100     # 限制数量测试
"""

import os
import sys
import csv
import time
import re
import requests
import argparse
import pymysql
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter, defaultdict
from datetime import datetime

# matplotlib 中文支持
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 导入路径配置
from config import DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR, REPORTS_DIR

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10

INPUT_PRICE = 0.0005
OUTPUT_PRICE = 0.002

# 数据库配置
DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def query_complaints_from_db(source_type=None, limit=None):
    """从数据库查询投诉类数据

    Args:
        source_type: 数据来源 (online/phone/None表示全部)
        limit: 限制数量
    """
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
            SELECT id, source_type, source_file, secondary_intent, sentiment, dialog_date
            FROM dialog_analysis
            WHERE primary_intent = '投诉'
            """
            if source_type:
                sql += f" AND source_type = '{source_type}'"
            sql += " ORDER BY dialog_date DESC"
            if limit:
                sql += f" LIMIT {limit}"

            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        conn.close()


# ==================== 文本清理 ====================

def clean_text(text):
    """清理文本中的特殊字符，保留换行符"""
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    # 移除emoji和特殊符号，但保留换行符（\n）
    # \u000A 是换行符，需要单独处理
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r'[^\u0020-\u007E\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]', '', line)
        cleaned_lines.append(cleaned_line.strip())
    return '\n'.join(cleaned_lines)

def clean_api_result(text):
    """清理API返回结果中的方括号和多余文字"""
    if not text:
        return '未知'
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[【】\[\]]', '', text)
    text = re.sub(r'客服.*', '', text)
    text = text.strip()
    return text if text else '未知'

# ==================== Prompt 定义 ====================

COMPLAINT_TYPE_PROMPT = """分析以下客服对话，判断客户投诉的具体类型。

对话内容：
{dialog}

请从以下投诉类型中选择最符合的一项，只输出类型名称：
理赔时效投诉、拒赔投诉、理赔金额异议、服务态度投诉、流程复杂投诉、其他投诉

注：流程复杂投诉指客户对理赔流程繁琐、材料要求多不满的情况。

只输出一个类型名称。"""

COMPLAINT_STAGE_PROMPT = """分析以下投诉对话，判断客户投诉发生在哪个理赔阶段。

对话内容：
{dialog}

请选择最符合的阶段，只输出阶段名称：
投保阶段、理赔申请阶段、材料审核阶段、等待打款阶段、理赔完成后、其他阶段

只输出一个阶段名称。"""

COMPLAINT_SEVERITY_PROMPT = """分析以下投诉对话，判断投诉的严重程度。

对话内容：
{dialog}

请根据以下标准判断严重程度，只输出一个级别：

- 严重：客户情绪激动、多次追问质疑、威胁向监管部门投诉、涉及金额较大（超过1万元）、明确表示强烈不满
- 中等：有明显不满情绪、问题需要跟进处理、投诉倾向明显、对结果有异议但态度尚可
- 轻微：仅咨询性质、情绪平和、问题简单可快速解决、投诉关键词但不强烈

只输出一个级别：严重、中等、轻微"""

COMPLAINT_ROOT_CAUSE_PROMPT = """分析以下投诉对话，找出投诉的根本原因。

对话内容：
{dialog}

请用一句话概括根本原因（20字以内）。只输出原因描述。"""

COMPLAINT_RESOLUTION_PROMPT = """分析以下投诉对话，判断客服的处理方式和效果。

对话内容：
{dialog}

请判断处理效果，只输出一个结果：
已解决、部分解决、未解决、升级处理

只输出一个结果。"""

# ==================== API 调用 ====================

progress_lock = Lock()
progress_counter = 0
total_tokens_input = 0
total_tokens_output = 0

def call_api(prompt, max_retries=3):
    """调用 API"""
    global total_tokens_input, total_tokens_output

    for retry in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": API_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "enable_thinking": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            usage = result.get("usage", {})
            with progress_lock:
                total_tokens_input += usage.get("prompt_tokens", 0)
                total_tokens_output += usage.get("completion_tokens", 0)
            return content
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def read_dialog_csv(csv_path):
    """读取 CSV 文件"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                return None, 0, []

            # 跳过标题行（可能的标题：角色、发送者、dialog者、对话者）
            if rows[0][0] in ['角色', '发送者', 'sender', '对话者', '﻿对话者']:
                rows = rows[1:]

            dialog_lines = []
            agent_replies = []
            for row in rows:
                if len(row) >= 2:
                    role = row[0]
                    content = row[1]
                    dialog_lines.append(f"{role}: {content}")
                    if role in ['客服', 'agent', '坐席']:
                        agent_replies.append(content)

            dialog = '\n'.join(dialog_lines)
            return dialog, len(rows), agent_replies
    except:
        return None, 0, []


def process_complaint_file(csv_path):
    """处理单个投诉对话文件"""

    dialog, line_count, agent_replies = read_dialog_csv(csv_path)
    if not dialog:
        return None

    # 调用 API 进行深度分析
    complaint_type = call_api(COMPLAINT_TYPE_PROMPT.format(dialog=dialog))
    complaint_stage = call_api(COMPLAINT_STAGE_PROMPT.format(dialog=dialog))
    severity = call_api(COMPLAINT_SEVERITY_PROMPT.format(dialog=dialog))
    root_cause = call_api(COMPLAINT_ROOT_CAUSE_PROMPT.format(dialog=dialog))
    resolution = call_api(COMPLAINT_RESOLUTION_PROMPT.format(dialog=dialog))

    # 过滤自动回复和营销内容
    blacklist = ['保障升级', '参保开启', '我时刻都准备', '本次服务将在', '在线客服人员',
                 '请问有什么可以帮您', '感谢您的支持', '长时间未对话']
    valid_replies = [r for r in agent_replies if not any(kw in r for kw in blacklist) and len(r) >= 30]

    return {
        'file': csv_path,
        'dialog': clean_text(dialog),
        'line_count': line_count,
        'complaint_type': clean_api_result(complaint_type),
        'complaint_stage': clean_api_result(complaint_stage),
        'severity': clean_api_result(severity),
        'root_cause': clean_text(root_cause or '未知原因'),
        'resolution': clean_api_result(resolution),
        'agent_replies': [clean_text(r) for r in valid_replies[:5]]
    }


# ==================== 图表生成 ====================

def create_pie_chart(data, labels, title, output_path):
    """创建饼图"""
    plt.figure(figsize=(10, 8))
    colors_list = ['#E74C3C', '#C0392B', '#9B59B6', '#8E44AD', '#3498DB', '#2980B9', '#F39C12', '#16A085']

    # 确保颜色数量匹配
    colors = colors_list[:len(data)]

    wedges, texts, autotexts = plt.pie(data, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)

    # 设置字体大小
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


def create_bar_chart(data, labels, title, output_path, horizontal=True):
    """创建柱状图"""
    plt.figure(figsize=(12, 7))
    colors_list = ['#E74C3C', '#C0392B', '#9B59B6', '#8E44AD', '#3498DB', '#2980B9', '#F39C12', '#16A085']

    colors = colors_list[:len(data)]

    if horizontal:
        bars = plt.barh(range(len(data)), data, color=colors)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('数量', fontsize=12)
        # 在柱状图上显示数值
        for i, (bar, val) in enumerate(zip(bars, data)):
            plt.text(val + 0.5, i, str(val), va='center', fontsize=10)
    else:
        bars = plt.bar(range(len(data)), data, color=colors)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.xlabel('类型', fontsize=12)
        for bar, val in zip(bars, data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha='center', fontsize=10)

    plt.ylabel('数量', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


# ==================== HTML 报告生成 ====================

def generate_html_report(results, output_dir, source_name):
    """生成投诉类深度分析HTML报告"""

    if not results:
        print("没有投诉类数据")
        return None

    # 统计分析
    type_dist = Counter(r['complaint_type'] for r in results)
    stage_dist = Counter(r['complaint_stage'] for r in results)
    severity_dist = Counter(r['severity'] for r in results)
    resolution_dist = Counter(r['resolution'] for r in results)
    root_cause_dist = Counter(r['root_cause'] for r in results)

    total = len(results)
    avg_length = sum(r['line_count'] for r in results) / total

    # 创建图表目录
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # 生成图表
    chart_files = {}

    if type_dist:
        chart_files['type'] = os.path.join(charts_dir, 'complaint_type.png')
        create_pie_chart(list(type_dist.values()), list(type_dist.keys()),
                        '投诉类型分布', chart_files['type'])

    if stage_dist:
        chart_files['stage'] = os.path.join(charts_dir, 'complaint_stage.png')
        create_pie_chart(list(stage_dist.values()), list(stage_dist.keys()),
                        '投诉发生阶段分布', chart_files['stage'])

    if severity_dist:
        chart_files['severity'] = os.path.join(charts_dir, 'complaint_severity.png')
        create_pie_chart(list(severity_dist.values()), list(severity_dist.keys()),
                        '投诉严重程度分布', chart_files['severity'])

    if root_cause_dist:
        chart_files['causes'] = os.path.join(charts_dir, 'complaint_causes.png')
        top_causes = root_cause_dist.most_common(10)
        create_bar_chart([c for _, c in top_causes], [l for l, _ in top_causes],
                        '投诉根本原因 TOP10', chart_files['causes'])

    if resolution_dist:
        chart_files['resolution'] = os.path.join(charts_dir, 'complaint_resolution.png')
        create_pie_chart(list(resolution_dist.values()), list(resolution_dist.keys()),
                        '投诉处理效果分布', chart_files['resolution'])

    # 生成 HTML
    html_path = os.path.join(output_dir, f'{source_name}_投诉类深度分析报告.html')

    # 类型描述
    type_desc = {
        '理赔时效投诉': '对理赔审核、打款时间过慢不满',
        '拒赔投诉': '对拒赔结果有异议，认为应该理赔',
        '理赔金额异议': '对赔付金额不满意，认为金额不合理',
        '服务态度投诉': '对客服态度、专业性不满',
        '流程复杂投诉': '对理赔流程繁琐、材料要求多不满',
        '其他投诉': '其他类型投诉',
    }

    # 严重程度颜色
    severity_colors = {
        '严重': '#E74C3C',
        '中等': '#F39C12',
        '轻微': '#27AE60',
    }

    # 处理结果颜色
    resolution_colors = {
        '已解决': '#27AE60',
        '部分解决': '#F39C12',
        '未解决': '#E74C3C',
        '升级处理': '#9B59B6',
    }

    # CSS样式（使用单花括号，因为是普通字符串）
    css_content = '''
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: "Microsoft YaHei", "SimHei", sans-serif;
            background: #f5f6fa;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
        }
        .nav-links {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .nav-links a {
            color: #666;
            text-decoration: none;
            padding: 8px 15px;
            border-radius: 5px;
            background: #f0f0f0;
            transition: all 0.3s;
        }
        .nav-links a:hover {
            background: #E74C3C;
            color: white;
        }
        .nav-links a.active {
            background: #E74C3C;
            color: white;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: #E74C3C;
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #E74C3C;
        }
        .section h3 {
            color: #C0392B;
            font-size: 16px;
            margin-bottom: 15px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card .number {
            font-size: 36px;
            font-weight: bold;
            color: #E74C3C;
        }
        .stat-card .label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .stat-card.warning .number {
            color: #F39C12;
        }
        .stat-card.danger .number {
            color: #E74C3C;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #E74C3C;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-danger { background: #E74C3C; color: white; }
        .badge-warning { background: #F39C12; color: white; }
        .badge-success { background: #27AE60; color: white; }
        .badge-info { background: #3498DB; color: white; }
        .case-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #E74C3C;
        }
        .case-card .case-header {
            font-weight: 600;
            color: #E74C3C;
            margin-bottom: 10px;
        }
        .case-card .dialog-content {
            background: white;
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        .recommendation-card {
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #E74C3C;
        }
        .recommendation-card .rec-type {
            font-weight: 600;
            color: #E74C3C;
            font-size: 16px;
        }
        .recommendation-card .rec-content {
            margin-top: 10px;
        }
        .recommendation-card .rec-item {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        .recommendation-card .rec-item:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: #E74C3C;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }
        .highlight-box {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .highlight-box.danger {
            background: #f8d7da;
            border-color: #E74C3C;
        }
        .highlight-box.success {
            background: #d4edda;
            border-color: #27AE60;
        }
    </style>'''

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>投诉类深度分析报告</title>
    {css_content}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>投诉类深度分析报告</h1>
            <div class="subtitle">数据来源：{source_name} | 分析时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>

        <div class="nav-links">
            <a href="#" class="active">投诉类分析</a>
            <a href="咨询类深度分析报告.html">咨询类分析</a>
            <a href="查询类深度分析报告.html">查询类分析</a>
            <a href="办理类深度分析报告.html">办理类分析</a>
            <a href="总览汇总报告.html">总览汇总</a>
        </div>

        <!-- 一、数据概览 -->
        <div class="section">
            <h2>一、投诉数据概览</h2>
            <div class="stats-grid">
                <div class="stat-card danger">
                    <div class="number">{total}</div>
                    <div class="label">投诉对话总数</div>
                </div>
                <div class="stat-card">
                    <div class="number">{avg_length:.1f}</div>
                    <div class="label">平均对话长度</div>
                </div>
                <div class="stat-card danger">
                    <div class="number">{severity_dist.get('严重', 0)}</div>
                    <div class="label">严重投诉数</div>
                </div>
                <div class="stat-card warning">
                    <div class="number">{resolution_dist.get('未解决', 0) + resolution_dist.get('升级处理', 0)}</div>
                    <div class="label">未解决投诉数</div>
                </div>
            </div>

            {generate_warning_box(severity_dist, total, resolution_dist)}
        </div>

        <!-- 二、投诉类型分布 -->
        <div class="section">
            <h2>二、投诉类型分布分析</h2>
            <table>
                <tr>
                    <th>投诉类型</th>
                    <th>数量</th>
                    <th>占比</th>
                    <th>说明</th>
                </tr>
                {generate_table_rows(type_dist, total, type_desc)}
            </table>
        </div>

        <!-- 三、投诉发生阶段 -->
        <div class="section">
            <h2>三、投诉发生理赔阶段分析</h2>
            <table>
                <tr>
                    <th>理赔阶段</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
                {generate_simple_table_rows(stage_dist, total)}
            </table>
        </div>

        <!-- 四、投诉严重程度 -->
        <div class="section">
            <h2>四、投诉严重程度分析</h2>

            <div class="severity-criteria-box" style="background:#f8f9fa; padding:15px; border-radius:8px; margin-bottom:20px; border-left:4px solid #E74C3C;">
                <strong>严重程度判断标准：</strong>
                <ul style="margin:10px 0 0 20px; font-size:13px;">
                    <li><span style="color:#E74C3C; font-weight:bold;">严重</span>：客户情绪激动、多次追问质疑、威胁向监管部门投诉、涉及金额较大（超过1万元）、明确表示强烈不满</li>
                    <li><span style="color:#F39C12; font-weight:bold;">中等</span>：有明显不满情绪、问题需要跟进处理、投诉倾向明显、对结果有异议但态度尚可</li>
                    <li><span style="color:#27AE60; font-weight:bold;">轻微</span>：仅咨询性质、情绪平和、问题简单可快速解决、投诉关键词但不强烈</li>
                </ul>
            </div>

            <table>
                <tr>
                    <th>严重程度</th>
                    <th>数量</th>
                    <th>占比</th>
                    <th>处理优先级</th>
                </tr>
                {generate_severity_table_rows(severity_dist, total)}
            </table>
        </div>

        <!-- 五、投诉根本原因 -->
        <div class="section">
            <h2>五、投诉根本原因 TOP10</h2>
            <table>
                <tr>
                    <th>排名</th>
                    <th>根本原因</th>
                    <th>出现次数</th>
                </tr>
                {generate_causes_table_rows(root_cause_dist)}
            </table>
        </div>

        <!-- 六、投诉处理效果 -->
        <div class="section">
            <h2>六、投诉处理效果分析</h2>
            <table>
                <tr>
                    <th>处理结果</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
                {generate_resolution_table_rows(resolution_dist, total)}
            </table>

            <div class="highlight-box {get_resolution_highlight_class(resolution_dist, total)}">
                <strong>处理效果分析：</strong>
                已解决率 {resolution_dist.get('已解决', 0)/total*100:.1f}%，
                未解决率 {(resolution_dist.get('未解决', 0) + resolution_dist.get('升级处理', 0))/total*100:.1f}%。
                {generate_resolution_comment(resolution_dist, total)}
            </div>
        </div>

        <!-- 七、高频投诉问题分析 -->
        <div class="section">
            <h2>七、高频投诉问题分析</h2>
            <table>
                <tr>
                    <th>排名</th>
                    <th>投诉问题</th>
                    <th>出现次数</th>
                    <th>严重程度</th>
                    <th>投诉类型</th>
                    <th>发生阶段</th>
                    <th>客户诉求摘要</th>
                    <th>客服回复摘要</th>
                </tr>
                {generate_case_cards(results, severity_colors)}
            </table>
        </div>

        <!-- 八、优化方案 -->
        <div class="section">
            <h2>八、投诉优化方案与建议</h2>
            {generate_recommendations(type_dist, stage_dist, severity_dist, resolution_dist, total)}
        </div>

        <!-- 九、总体建议 -->
        <div class="section">
            <h2>九、总体优化建议</h2>
            <div class="recommendation-card">
                <div class="rec-type">系统性改进建议</div>
                <div class="rec-content">
                    <div class="rec-item">建立投诉预警机制：对理赔时长超过规定期限的案件自动预警，提前介入</div>
                    <div class="rec-item">优化沟通流程：制定标准化投诉处理话术，提升客服专业度和同理心</div>
                    <div class="rec-item">提升透明度：提供理赔进度实时查询功能，让客户随时了解处理状态</div>
                    <div class="rec-item">建立回访机制：投诉处理完成后48小时内主动回访确认满意度</div>
                    <div class="rec-item">数据分析闭环：每周分析投诉数据，持续优化产品和服务流程</div>
                    <div class="rec-item">培训赋能：定期组织投诉处理培训，分享典型案例和处理技巧</div>
                </div>
            </div>
        </div>

        <div class="footer">
            报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
            分析脚本：analyze_complaint.py |
            数据来源：{source_name}
        </div>
    </div>
</body>
</html>'''

    # 写入HTML文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path


# ==================== HTML辅助函数 ====================

def generate_warning_box(severity_dist, total, resolution_dist):
    """生成警告提示框"""
    severe_ratio = severity_dist.get('严重', 0) / total * 100
    unresolved_ratio = (resolution_dist.get('未解决', 0) + resolution_dist.get('升级处理', 0)) / total * 100

    if severe_ratio > 10 or unresolved_ratio > 30:
        return f'''<div class="highlight-box danger">
            <strong>⚠️ 需要重点关注：</strong>
            严重投诉占比 {severe_ratio:.1f}%，未解决投诉占比 {unresolved_ratio:.1f}%。
            建议立即启动投诉处理专项改进计划。
        </div>'''
    elif severe_ratio > 5 or unresolved_ratio > 20:
        return f'''<div class="highlight-box">
            <strong>⚠️ 提示：</strong>
            严重投诉占比 {severe_ratio:.1f}%，未解决投诉占比 {unresolved_ratio:.1f}%。
            建议加强客服培训和流程优化。
        </div>'''
    else:
        return f'''<div class="highlight-box success">
            <strong>✅ 整体状况良好：</strong>
            严重投诉占比 {severe_ratio:.1f}%，未解决投诉占比 {unresolved_ratio:.1f}%。
            建议继续保持服务质量监控。
        </div>'''


def generate_table_rows(dist, total, desc_map):
    """生成表格行"""
    rows = []
    for item, count in dist.most_common():
        ratio = count / total * 100
        desc = desc_map.get(item, '')
        rows.append(f'''<tr>
            <td>{item}</td>
            <td>{count}</td>
            <td>{ratio:.1f}%</td>
            <td>{desc}</td>
        </tr>''')
    return '\n'.join(rows)


def generate_simple_table_rows(dist, total):
    """生成简单表格行"""
    rows = []
    for item, count in dist.most_common():
        ratio = count / total * 100
        rows.append(f'''<tr>
            <td>{item}</td>
            <td>{count}</td>
            <td>{ratio:.1f}%</td>
        </tr>''')
    return '\n'.join(rows)


def generate_severity_table_rows(severity_dist, total):
    """生成严重程度表格行"""
    priority_map = {'严重': '立即处理', '中等': '优先处理', '轻微': '常规处理'}
    color_map = {'严重': 'badge-danger', '中等': 'badge-warning', '轻微': 'badge-success'}

    rows = []
    for item, count in severity_dist.most_common():
        ratio = count / total * 100
        priority = priority_map.get(item, '')
        badge_class = color_map.get(item, '')
        rows.append(f'''<tr>
            <td><span class="badge {badge_class}">{item}</span></td>
            <td>{count}</td>
            <td>{ratio:.1f}%</td>
            <td>{priority}</td>
        </tr>''')
    return '\n'.join(rows)


def generate_causes_table_rows(root_cause_dist):
    """生成原因表格行"""
    rows = []
    for i, (cause, count) in enumerate(root_cause_dist.most_common(10), 1):
        rows.append(f'''<tr>
            <td>{i}</td>
            <td>{cause}</td>
            <td>{count}</td>
        </tr>''')
    return '\n'.join(rows)


def generate_resolution_table_rows(resolution_dist, total):
    """生成处理效果表格行"""
    color_map = {'已解决': 'badge-success', '部分解决': 'badge-warning',
                 '未解决': 'badge-danger', '升级处理': 'badge-info'}

    rows = []
    for item, count in resolution_dist.most_common():
        ratio = count / total * 100
        badge_class = color_map.get(item, '')
        rows.append(f'''<tr>
            <td><span class="badge {badge_class}">{item}</span></td>
            <td>{count}</td>
            <td>{ratio:.1f}%</td>
        </tr>''')
    return '\n'.join(rows)


def get_resolution_highlight_class(resolution_dist, total):
    """获取处理效果提示框样式"""
    resolved_ratio = resolution_dist.get('已解决', 0) / total * 100
    if resolved_ratio > 50:
        return 'success'
    elif resolved_ratio > 30:
        return ''
    else:
        return 'danger'


def generate_resolution_comment(resolution_dist, total):
    """生成处理效果评价"""
    resolved_ratio = resolution_dist.get('已解决', 0) / total * 100
    if resolved_ratio > 50:
        return '投诉解决率较高，客服处理能力良好。'
    elif resolved_ratio > 30:
        return '投诉解决率中等，需要进一步提升客服处理能力。'
    else:
        return '投诉解决率偏低，需要重点改进投诉处理流程和客服培训。'


def generate_chart_html(chart_type, chart_files, title):
    """生成图表HTML（使用base64嵌入图片）"""
    import base64

    if chart_type in chart_files and os.path.exists(chart_files[chart_type]):
        # 将图片转换为base64嵌入
        with open(chart_files[chart_type], 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return f'''<div class="chart-container">
            <img src="data:image/png;base64,{img_base64}" alt="{title}" style="max-width:100%; height:auto;">
        </div>'''
    return ''


def generate_case_cards(results, severity_colors):
    """生成高频投诉问题分析卡片（简化版）"""
    # 统计高频投诉问题
    issue_counter = Counter(r['root_cause'] for r in results)
    top_issues = issue_counter.most_common(10)

    # 为每个高频问题生成卡片
    cards = []
    for i, (issue, count) in enumerate(top_issues[:10], 1):
        # 找到该问题的典型案例
        issue_cases = [r for r in results if r['root_cause'] == issue]
        if not issue_cases:
            continue

        typical_case = issue_cases[0]

        # 严重程度颜色
        color = severity_colors.get(typical_case['severity'], '#666')

        # 从对话中提取客户诉求
        dialog_text = typical_case.get('dialog', '')
        lines = dialog_text.split('\n')
        customer_lines = []
        for line in lines:
            if line.startswith('客户:'):
                content = line.replace('客户:', '').strip()
                if len(content) > 5 and '您好' not in content and '人工' not in content:
                    customer_lines.append(content)

        # 取第一条有意义的客户诉求
        customer_summary = customer_lines[0][:50] if customer_lines else '-'

        # 使用已过滤的客服回复字段
        agent_replies = typical_case.get('agent_replies', [])
        agent_summary = agent_replies[0][:50] if agent_replies else '-'

        cards.append(f'''<tr>
            <td>{i}</td>
            <td>{issue}</td>
            <td>{count}</td>
            <td><span class="badge" style="background:{color}; color:white;">{typical_case['severity']}</span></td>
            <td>{typical_case['complaint_type']}</td>
            <td>{typical_case['complaint_stage']}</td>
            <td style="max-width:180px; font-size:12px;">{customer_summary}</td>
            <td style="max-width:180px; font-size:12px;">{agent_summary}</td>
        </tr>''')

    return '\n'.join(cards)


def generate_recommendations(type_dist, stage_dist, severity_dist, resolution_dist, total):
    """生成针对性建议"""
    recommendations = []

    # 理赔时效投诉
    if type_dist.get('理赔时效投诉', 0) > total * 0.2:
        recommendations.append({
            'type': '理赔时效投诉优化',
            'problem': '理赔审核和处理时间过长，客户等待焦虑',
            'solution': '建立理赔进度实时查询系统，定期推送进度通知，设置时效预警',
            'suggestion': '优化审核流程，设立时效监控指标，超过10天自动预警，专人跟进'
        })

    # 拒赔投诉
    if type_dist.get('拒赔投诉', 0) > total * 0.1:
        recommendations.append({
            'type': '拒赔投诉优化',
            'problem': '客户对拒赔结果不理解、不接受',
            'solution': '拒赔时提供详细书面解释，告知条款依据和申诉渠道',
            'suggestion': '优化拒赔沟通模板，提供条款解读服务，建立拒赔复核机制'
        })

    # 金额异议
    if type_dist.get('理赔金额异议', 0) > total * 0.1:
        recommendations.append({
            'type': '理赔金额异议优化',
            'problem': '客户对赔付金额不满意，不理解计算方式',
            'solution': '提供金额计算明细说明，解释赔付比例和扣减项目',
            'suggestion': '优化金额计算透明度，提供在线计算明细查询功能'
        })

    # 服务态度
    if type_dist.get('服务态度投诉', 0) > total * 0.05:
        recommendations.append({
            'type': '服务态度优化',
            'problem': '客服沟通方式和态度问题',
            'solution': '加强客服培训，建立投诉话术标准，提升同理心',
            'suggestion': '实施客服满意度评价，定期质检录音，建立考核激励机制'
        })

    # 流程复杂
    if type_dist.get('流程复杂投诉', 0) > total * 0.05:
        recommendations.append({
            'type': '流程简化优化',
            'problem': '理赔流程繁琐，材料要求多',
            'solution': '简化理赔材料清单，提供一站式理赔服务指引',
            'suggestion': '优化线上理赔流程，减少重复材料提交，智能预填表单'
        })

    # 严重投诉比例高
    severe_ratio = severity_dist.get('严重', 0) / total * 100
    if severe_ratio > 10:
        recommendations.append({
            'type': '严重投诉处理机制',
            'problem': '严重投诉比例较高，可能升级或造成负面影响',
            'solution': '建立严重投诉快速响应机制，专人跟进，24小时内处理',
            'suggestion': '设立投诉升级预警系统，严重投诉自动标记，管理层介入'
        })

    # 未解决率高
    unresolved_ratio = (resolution_dist.get('未解决', 0) + resolution_dist.get('升级处理', 0)) / total * 100
    if unresolved_ratio > 20:
        recommendations.append({
            'type': '投诉解决率提升',
            'problem': '投诉解决率偏低，客户不满持续',
            'solution': '加强客服培训和授权，提升一线处理能力',
            'suggestion': '建立投诉复盘机制，分析未解决原因，针对性改进'
        })

    # 理赔申请阶段投诉多
    if stage_dist.get('理赔申请阶段', 0) > total * 0.5:
        recommendations.append({
            'type': '理赔申请阶段优化',
            'problem': '理赔申请阶段投诉集中',
            'solution': '优化理赔申请引导，提供材料清单和操作指南',
            'suggestion': '简化申请流程，提供在线材料上传和预审功能'
        })

    # 生成HTML卡片
    cards = []
    for rec in recommendations:
        cards.append(f'''<div class="recommendation-card">
            <div class="rec-type">{rec['type']}</div>
            <div class="rec-content">
                <div class="rec-item"><strong>问题诊断：</strong>{rec['problem']}</div>
                <div class="rec-item"><strong>解决方案：</strong>{rec['solution']}</div>
                <div class="rec-item"><strong>预防建议：</strong>{rec['suggestion']}</div>
            </div>
        </div>''')

    if not cards:
        cards.append('''<div class="recommendation-card">
            <div class="rec-type">整体服务优化</div>
            <div class="rec-content">
                <div class="rec-item">当前投诉数据整体可控，建议继续保持服务质量监控</div>
                <div class="rec-item">定期进行客服培训和服务质量评估</div>
                <div class="rec-item">建立客户满意度跟踪机制</div>
            </div>
        </div>''')

    return '\n'.join(cards)


# ==================== 主流程 ====================

def analyze_complaints(csv_dir, limit=None):
    """分析投诉类对话"""
    global progress_counter, total_tokens_input, total_tokens_output

    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    if limit:
        csv_files = csv_files[:limit]

    total_files = len(csv_files)
    print(f"\n{'='*60}")
    print(f"投诉类深度分析")
    print(f"目录: {csv_dir}")
    print(f"文件数量: {total_files}")
    print(f"{'='*60}\n")

    print("识别投诉类对话...")

    # 先筛选可能包含投诉的文件
    complaint_keywords = ['投诉', '不满', '太慢', '拒赔', '金额不对', '态度差',
                          '不满意', '为什么不', '怎么这么慢', '搞什么', '很久',
                          '拖延', '时间太长', '等了好久', '还没立案', '还没处理',
                          '怎么还没', '为什么还没', '太慢长', '为什么不理',
                          '为什么不赔', '金额不对', '对理赔', '对金额',
                          '对结果', '拒赔了', '被拒', '不能理赔', '不理赔',
                          '审核太久', '打款太慢', '这都好几天', '都多久了']

    potential_complaints = []
    for f in csv_files:
        dialog, _, _ = read_dialog_csv(f)
        if dialog and any(kw in dialog for kw in complaint_keywords):
            potential_complaints.append(f)

    print(f"潜在投诉对话: {len(potential_complaints)} 个")

    # 深度分析
    results = []
    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = {executor.submit(process_complaint_file, f): f for f in potential_complaints}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            progress_counter += 1
            if progress_counter % 50 == 0:
                print(f"进度: {progress_counter}/{len(potential_complaints)}")

    print(f"\n识别到投诉类对话: {len(results)} 个")

    # 计算成本
    total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE
    print(f"API 成本: ¥{total_cost:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="投诉类深度分析脚本")
    parser.add_argument("--type", choices=['online', 'phone', 'all'],
                       default='all', help="数据来源类型: online=在线对话, phone=电话录音, all=全部")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    args = parser.parse_args()

    # 确定数据来源
    if args.type == 'phone':
        source_type = 'phone'
        source_name = '电话录音'
    elif args.type == 'online':
        source_type = 'online'
        source_name = '在线对话'
    else:
        source_type = None  # 全部
        source_name = '全量数据'

    print(f"\n{'='*60}")
    print(f"投诉类深度分析 - {source_name}")
    print(f"{'='*60}")

    # 从数据库查询投诉数据
    print(f"\n从数据库查询投诉数据...")
    db_records = query_complaints_from_db(source_type, args.limit)
    print(f"查询到 {len(db_records)} 条投诉记录")

    if not db_records:
        print("没有投诉数据")
        return

    # 分析投诉数据
    print(f"\n开始深度分析投诉对话...")
    results = analyze_complaints_from_db(db_records)

    if not results:
        print("没有成功分析的投诉对话")
        return

    # 生成报告
    output_dir = REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n生成投诉类深度分析HTML报告...")
    html_path = generate_html_report(results, output_dir, source_name)

    if html_path:
        print(f"报告生成成功: {html_path}")
        print(f"\n可用浏览器打开查看")


def analyze_complaints_from_db(db_records):
    """从数据库记录分析投诉数据"""
    global progress_counter, total_tokens_input, total_tokens_output

    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    total_files = len(db_records)
    print(f"待分析数量: {total_files}")

    results = []
    progress_lock = Lock()

    def process_record(record):
        global progress_counter
        csv_path = record['source_file']

        # 检查文件是否存在
        if not os.path.exists(csv_path):
            return None

        # 读取对话内容
        dialog, line_count, agent_replies = read_dialog_csv(csv_path)
        if not dialog:
            return None

        # 调用 API 进行深度分析
        complaint_type = call_api(COMPLAINT_TYPE_PROMPT.format(dialog=dialog))
        complaint_stage = call_api(COMPLAINT_STAGE_PROMPT.format(dialog=dialog))
        severity = call_api(COMPLAINT_SEVERITY_PROMPT.format(dialog=dialog))
        root_cause = call_api(COMPLAINT_ROOT_CAUSE_PROMPT.format(dialog=dialog))
        resolution = call_api(COMPLAINT_RESOLUTION_PROMPT.format(dialog=dialog))

        # 过滤自动回复和营销内容
        blacklist = ['保障升级', '参保开启', '我时刻都准备', '本次服务将在', '在线客服人员',
                     '请问有什么可以帮您', '感谢您的支持', '长时间未对话']
        valid_replies = [r for r in agent_replies if not any(kw in r for kw in blacklist) and len(r) >= 30]

        with progress_lock:
            progress_counter += 1
            if progress_counter % 50 == 0 or progress_counter == total_files:
                print(f"进度: {progress_counter}/{total_files}")

        return {
            'file': csv_path,
            'dialog': clean_text(dialog),
            'line_count': line_count,
            'secondary_intent': record['secondary_intent'],  # 使用数据库的分类
            'complaint_type': clean_api_result(complaint_type),
            'complaint_stage': clean_api_result(complaint_stage),
            'severity': clean_api_result(severity),
            'root_cause': clean_text(root_cause or '未知原因'),
            'resolution': clean_api_result(resolution),
            'agent_replies': [clean_text(r) for r in valid_replies[:5]]
        }

    # 使用多线程并发处理
    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, r) for r in db_records]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    print(f"\n处理完成！成功: {len(results)}, 失败: {total_files - len(results)}")

    # 计算成本
    total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE
    print(f"API 成本: ¥{total_cost:.2f}")

    return results


if __name__ == "__main__":
    main()