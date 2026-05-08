"""
客服对话详细分析脚本

功能：
1. 按一级分类（咨询、查询、办理、投诉、其他）分别生成分析报告
2. 每个类别内的二级意图分布
3. 每个类别的高频问题 TOP10
4. 情感分布、关键词提取、典型案例等

使用方式：
  python analyze_detail.py --dir online_chat_csv              # 分析在线对话
  python analyze_detail.py --dir dialog_csv                   # 分析电话录音
  python analyze_detail.py --dir online_chat_csv --limit 5000 # 限制处理数量
"""

import os
import sys
import csv
import time
import re
import requests
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter, defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# matplotlib 中文支持
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置 matplotlib 全局中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 文本清理函数 ====================

def clean_text_for_pdf(text):
    """清理文本中的特殊字符，确保PDF正确显示

    移除以下类型的字符：
    - Emoji表情符号
    - 特殊符号（星星、箭头等）
    - 弯引号（替换为直引号）
    - 控制字符
    - 私用区域字符
    """
    import re

    # 先替换弯引号为直引号
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")

    # 保留：ASCII字符、中文、常用标点、空格
    # Unicode范围：
    # \u0000-\u007F: ASCII基本字符
    # \u4E00-\u9FFF: CJK统一汉字
    # \u3000-\u303F: CJK符号和标点
    # \uFF00-\uFFEF: 全角字符
    # \u0020-\u002F: 基本标点前半部分
    # \u003A-\u003F: 标点后半部分
    # \u005B-\u0060: 方括号等
    # \u007B-\u007E: 大括号等
    # \s: 空白字符

    # 更严格的清理：只保留ASCII、中文和常用标点
    cleaned = re.sub(r'[^\u0020-\u007E\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]', '', text)
    return cleaned.strip()

# 不使用 FontProperties，直接用 rcParams 全局设置

# ==================== 配置 ====================

from config import DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR, REPORTS_DIR

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10

# 计费标准
INPUT_PRICE = 0.0005
OUTPUT_PRICE = 0.002

# ==================== 分类体系 ====================

INTENT_CATEGORIES = {
    "咨询": ["产品了解", "条款解释", "费率查询", "理赔流程了解", "理赔材料了解", "退保流程了解", "保障范围了解", "其他了解"],
    "查询": ["理赔进度查询", "理赔到账查询", "保单状态查询", "其他查询"],
    "办理": ["理赔申请", "事故报案", "新保投保", "续保办理", "保单变更", "退保办理", "减保办理"],
    "投诉": ["理赔时效投诉", "拒赔投诉", "理赔金额异议", "服务态度投诉", "其他投诉"],
    "其他": ["回访确认", "信息核实"]
}

PRIMARY_CATEGORIES = ["咨询", "查询", "办理", "投诉", "其他"]

# ==================== Prompt 定义 ====================

INTENT_PROMPT = """下面是一段保险客服对话的内容，请分析客户的咨询原因。

对话内容：
{dialog}

请从以下分类体系中选择最符合的原因，输出格式为"一级分类-二级分类"：

【判断规则】
- 如果客户问"是什么、怎么做、多少钱" → 咨询（了解信息）
- 如果客户问"到哪了、什么时候、有没有" → 查询（查状态）
- 如果客户说"我要、帮我、申请" → 办理（办业务）
- 如果客户有抱怨、不满、质疑 → 投诉

【咨询】客户询问了解信息
- 产品了解：问产品保障内容
- 条款解释：问条款含义
- 费率查询：问保费价格
- 理赔流程了解：问理赔步骤
- 理赔材料了解：问理赔材料
- 退保流程了解：问退保流程
- 保障范围了解：问保障范围
- 其他了解：其他咨询

【查询】客户查询具体状态
- 理赔进度查询：查理赔审核进度
- 理赔到账查询：查理赔款到账
- 保单状态查询：查保单状态
- 其他查询：其他状态查询

【办理】客户要办理业务
- 理赔申请：提交理赔
- 事故报案：报告事故
- 新保投保：新买保险
- 续保办理：续保操作
- 保单变更：变更信息
- 退保办理：办理退保
- 减保办理：减少保额

【投诉】客户有不满投诉
- 理赔时效投诉：投诉理赔慢
- 拒赔投诉：投诉拒赔
- 理赔金额异议：对金额不满
- 服务态度投诉：投诉服务
- 其他投诉：其他不满

【其他】回访确认、信息核实

只输出一个分类（如：查询-保单状态查询），不要解释。"""

SENTIMENT_PROMPT = """分析以下客服对话中客户的情绪状态。

对话内容：
{dialog}

请判断客户的情绪倾向，只输出一个标签：
- 满意：客户问题得到解决，态度积极
- 中立：客户情绪平稳，无明显倾向
- 不满：客户有抱怨、质疑、投诉倾向

只输出一个词（满意/中立/不满），不要解释。"""

ISSUE_PROMPT = """分析以下客服对话，判断客户咨询的核心问题。

对话内容：
{dialog}

请从以下选项中选择唯一一个最符合的问题类型，只输出类型名称：

理赔进度查询、理赔款项到账查询、理赔时效咨询、理赔材料准备、理赔资料补交、理赔金额计算、理赔金额异议、报销比例咨询、保障范围咨询、理赔条件确认、既往症咨询、退保流程咨询、退保金额计算、犹豫期退保、保单查询、保单变更、信息修改、续保咨询、产品条款解释、保费计算、投保条件咨询

只输出一个类型名称，不要解释。"""

KEYWORDS_PROMPT = """分析以下客服对话，提取客户关注的关键信息点（3-5个关键词）。

对话内容：
{dialog}

只输出关键词，用逗号分隔，不要解释。如：理赔进度,审核时间,材料补交"""


# ==================== API 调用 ====================

progress_lock = Lock()
progress_counter = 0
total_tokens_input = 0
total_tokens_output = 0

def call_api(prompt, max_retries=3):
    """调用 API，带重试机制"""
    global total_tokens_input, total_tokens_output

    for retry in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
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
    """读取 CSV 文件，返回对话内容和客服回答"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                return None, 0, []

            # 检查标题行
            first_row = rows[0]
            if first_row[0] in ['角色', '发送者', 'sender']:
                rows = rows[1:]

            # 拼接对话内容，并提取客服回答
            dialog_lines = []
            agent_replies = []  # 客服回答列表
            for row in rows:
                if len(row) >= 2:
                    role = row[0]
                    content = row[1]
                    dialog_lines.append(f"{role}: {content}")
                    # 提取客服回答（客服、agent、坐席等）
                    if role in ['客服', 'agent', '坐席', '工作人员']:
                        agent_replies.append(content)

            dialog = '\n'.join(dialog_lines)
            return dialog, len(rows), agent_replies
    except Exception as e:
        return None, 0, []


def parse_intent(result):
    """解析意图分类结果"""
    if not result:
        return "其他", "其他了解"

    valid_primary = ["咨询", "查询", "办理", "投诉", "其他"]

    if '-' in result:
        parts = result.split('-', 1)
        primary = parts[0].strip()
        secondary = parts[1].strip() if len(parts) > 1 else "其他"

        if primary not in valid_primary:
            # 从二级分类推断一级分类
            if any(k in result for k in ['了解', '咨询', '解释']):
                primary = "咨询"
            elif any(k in result for k in ['查询', '进度', '状态']):
                primary = "查询"
            elif any(k in result for k in ['申请', '办理', '报案']):
                primary = "办理"
            elif any(k in result for k in ['投诉', '不满']):
                primary = "投诉"
            else:
                primary = "其他"

        return primary, secondary

    return "其他", result


def process_single_file(csv_path):
    """处理单个文件，返回分析结果"""
    global progress_counter

    dialog, line_count, agent_replies = read_dialog_csv(csv_path)
    if not dialog:
        return None

    # 调用 API 分析
    intent_result = call_api(INTENT_PROMPT.format(dialog=dialog))
    sentiment_result = call_api(SENTIMENT_PROMPT.format(dialog=dialog))
    issue_result = call_api(ISSUE_PROMPT.format(dialog=dialog))
    keywords_result = call_api(KEYWORDS_PROMPT.format(dialog=dialog))

    # 解析意图
    primary_intent, secondary_intent = parse_intent(intent_result)

    # 提取关键词
    keywords = []
    if keywords_result:
        keywords = [k.strip() for k in keywords_result.replace('，', ',').split(',') if k.strip()]

    with progress_lock:
        progress_counter += 1

    return {
        'file': csv_path,
        'primary_intent': primary_intent,
        'secondary_intent': secondary_intent,
        'sentiment': sentiment_result or '中立',
        'issue_type': issue_result or '未知',
        'keywords': keywords,
        'line_count': line_count,
        'dialog': dialog,  # 保留原文用于典型案例提取
        'agent_replies': agent_replies  # 客服回答列表
    }


# ==================== 数据分析 ====================

def analyze_by_category(results):
    """按一级分类分析数据"""
    category_data = defaultdict(list)
    for r in results:
        category_data[r['primary_intent']].append(r)
    return category_data


def get_top_issues(results, top_n=10):
    """获取高频问题 TOP N"""
    issue_counter = Counter(r['issue_type'] for r in results if r['issue_type'])
    return issue_counter.most_common(top_n)


def get_issue_replies(results, issue_type, max_replies=3, max_length=300):
    """获取指定问题类型的客服典型回答"""
    # 找到该问题类型的所有对话
    issue_results = [r for r in results if r['issue_type'] == issue_type and r.get('agent_replies')]

    if not issue_results:
        return []

    # 自动回复和营销文案黑名单（需要过滤的内容）
    blacklist_patterns = [
        # 自动回复
        '我时刻都准备为您服务',
        '本次服务将在',
        '分钟后结束',
        '在线客服人员',
        '请问有什么可以帮您',
        '感谢您的支持',
        '随时继续提问',
        '长时间未对话',
        'Hi，我是',
        # 营销推广
        '保障升级',
        '参保开启',
        '六期',
        '海内外特药',
        '家庭参保',
        '医保个账',
        '公众号',
        'APP',
        '扫码',
        '点击链接',
        '查看详情',
        # 问候语
        '您好，请咨询',
        '您好，',
        '你好',
    ]

    # 检查是否包含 emoji 或特殊符号（会导致 PDF 乱码）
    def has_special_chars(text):
        """检查文本是否包含无法在PDF中正确显示的特殊字符"""
        for char in text:
            cp = ord(char)
            # Emoji范围（更全面）
            if 0x1F000 <= cp <= 0x1FFFF:
                return True
            # 其他符号和象形文字
            if 0x2600 <= cp <= 0x27BF:
                return True
            if 0x2B50 <= cp <= 0x2B55:  # 星星等
                return True
            # 控制字符
            if cp < 0x20 or cp == 0x7F:
                return True
            # 私用区域
            if 0xE000 <= cp <= 0xF8FF:
                return True
        return False

    # 清理文本中的特殊字符
    def clean_text(text):
        """移除文本中的特殊字符"""
        import re
        # 移除emoji和特殊符号
        cleaned = re.sub(r'[^\u0000-\u007F\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF\u0020-\u002F\u003A-\u003F\u005B-\u0060\u007B-\u007E\s]', '', text)
        return cleaned.strip()

    # 收集所有客服回答，过滤无效内容
    all_replies = []
    for r in issue_results:
        for reply in r['agent_replies']:
            # 过滤黑名单内容
            is_blacklisted = any(pattern in reply for pattern in blacklist_patterns)
            if is_blacklisted:
                continue

            # 过滤包含特殊字符的内容（会导致 PDF 乱码）
            if has_special_chars(reply):
                continue

            # 筛选有效回答：长度适中、有实质内容
            if 30 <= len(reply) <= max_length:
                # 检查是否包含业务相关词汇
                business_keywords = ['理赔', '保单', '保险', '报销', '材料', '审核',
                                     '打款', '到账', '投保', '退保', '续保', '变更',
                                     '保障', '条款', '费用', '金额', '时效', '流程',
                                     '申请', '查询', '进度', '状态', '条件', '比例']
                has_business_content = any(kw in reply for kw in business_keywords)
                if has_business_content:
                    all_replies.append(reply)

    # 去重并选取最具代表性的回答
    reply_counter = Counter(all_replies)
    top_replies = [r for r, _ in reply_counter.most_common(max_replies)]

    # 如果筛选后没有足够回答，放宽条件再找
    if len(top_replies) < max_replies:
        for r in issue_results:
            for reply in r['agent_replies']:
                # 第二轮：过滤黑名单和特殊字符，不要求业务关键词
                is_blacklisted = any(pattern in reply for pattern in blacklist_patterns)
                if is_blacklisted or has_special_chars(reply):
                    continue
                if 30 <= len(reply) <= max_length and reply not in top_replies:
                    top_replies.append(reply)
                    if len(top_replies) >= max_replies:
                        break

    return top_replies[:max_replies]


def get_top_issues_with_replies(results, top_n=10):
    """获取高频问题 TOP N 及对应客服回答"""
    top_issues = get_top_issues(results, top_n)
    issues_with_replies = []
    for issue, count in top_issues:
        replies = get_issue_replies(results, issue)
        issues_with_replies.append((issue, count, replies))
    return issues_with_replies


def get_top_keywords(results, top_n=20):
    """获取高频关键词 TOP N"""
    keyword_counter = Counter()
    for r in results:
        for kw in r['keywords']:
            keyword_counter[kw] += 1
    return keyword_counter.most_common(top_n)


def get_secondary_distribution(results):
    """获取二级意图分布"""
    secondary_counter = Counter(r['secondary_intent'] for r in results)
    return secondary_counter.most_common()


def get_sentiment_distribution(results):
    """获取情感分布"""
    sentiment_counter = Counter(r['sentiment'] for r in results)
    return sentiment_counter


def get典型案例(results, category, n=3):
    """获取典型案例（按对话长度选代表性案例，过滤营销内容）"""
    category_results = [r for r in results if r['primary_intent'] == category]

    # 过滤掉包含营销内容的对话
    marketing_keywords = ['保障升级', '参保开启', '限时补缴', '立即参保', '早添保障',
                          '海内外特药', '医保个账', '断保用户', '续保优待', '关键时刻能托底']

    filtered_results = []
    for r in category_results:
        dialog = r.get('dialog', '')
        # 检查是否包含营销内容
        has_marketing = any(kw in dialog for kw in marketing_keywords)
        if not has_marketing:
            filtered_results.append(r)

    # 如果过滤后没有足够案例，放宽条件
    if len(filtered_results) < n:
        filtered_results = category_results

    # 按对话长度排序，选中等长度的案例
    sorted_results = sorted(filtered_results, key=lambda x: x['line_count'])
    if len(sorted_results) >= n:
        # 取前、中、后三个位置的案例
        mid = len(sorted_results) // 2
        return [
            sorted_results[0],
            sorted_results[mid],
            sorted_results[-1]
        ]
    return sorted_results[:n]


def extract_dialog_snippet(dialog, max_length=500):
    """截取对话片段，并清理特殊字符"""
    # 先清理特殊字符
    dialog = clean_text_for_pdf(dialog)
    # 截取长度
    if len(dialog) > max_length:
        return dialog[:max_length] + "..."
    return dialog


# ==================== 报告生成 ====================

def register_chinese_font():
    """注册中文字体"""
    try:
        # 使用 reportlab 内置的 UnicodeCIDFont（支持中文）
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        return 'STSong-Light'
    except Exception as e:
        print(f"UnicodeCIDFont 注册失败: {e}")
        return 'Helvetica'


def create_pie_chart(data, labels, title, output_path):
    """创建饼图"""
    plt.figure(figsize=(8, 6))
    colors_list = ['#4CAF50', '#2196F3', '#FF9800', '#f44336', '#9C27B0', '#00BCD4']
    # 使用全局 rcParams 字体设置
    plt.pie(data, labels=labels, autopct='%1.1f%%', colors=colors_list[:len(data)])
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_bar_chart(data, labels, title, output_path, horizontal=False):
    """创建柱状图"""
    plt.figure(figsize=(10, 6))
    colors_list = ['#4CAF50', '#2196F3', '#FF9800', '#f44336', '#9C27B0', '#00BCD4']

    if horizontal:
        plt.barh(range(len(data)), data, color=colors_list[:len(data)])
        plt.yticks(range(len(labels)), labels)
    else:
        plt.bar(range(len(data)), data, color=colors_list[:len(data)])
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_category_report(category, results, output_dir, font_name):
    """生成单个分类的详细报告"""
    category_results = [r for r in results if r['primary_intent'] == category]
    if not category_results:
        return None

    # 统计数据
    secondary_dist = get_secondary_distribution(category_results)
    sentiment_dist = get_sentiment_distribution(category_results)
    top_issues_with_replies = get_top_issues_with_replies(category_results, 10)
    top_keywords = get_top_keywords(category_results, 20)
    typical_cases = get典型案例(results, category, 3)

    # 计算平均对话长度
    avg_length = sum(r['line_count'] for r in category_results) / len(category_results)

    # 创建图表
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # 二级意图分布饼图
    if secondary_dist:
        secondary_data = [c for _, c in secondary_dist[:8]]
        secondary_labels = [l for l, _ in secondary_dist[:8]]
        create_pie_chart(secondary_data, secondary_labels,
                        f'{category}类二级意图分布',
                        os.path.join(charts_dir, f'{category}_secondary.png'))

    # 情感分布饼图
    if sentiment_dist:
        sentiment_data = list(sentiment_dist.values())
        sentiment_labels = list(sentiment_dist.keys())
        create_pie_chart(sentiment_data, sentiment_labels,
                        f'{category}类客户情感分布',
                        os.path.join(charts_dir, f'{category}_sentiment.png'))

    # 高频问题柱状图
    if top_issues_with_replies:
        issue_data = [c for _, c, _ in top_issues_with_replies]
        issue_labels = [l for l, _, _ in top_issues_with_replies]
        create_bar_chart(issue_data, issue_labels,
                        f'{category}类高频问题 TOP10',
                        os.path.join(charts_dir, f'{category}_issues.png'),
                        horizontal=True)

    # 高频关键词柱状图
    if top_keywords:
        kw_data = [c for _, c in top_keywords[:10]]
        kw_labels = [l for l, _ in top_keywords[:10]]
        create_bar_chart(kw_data, kw_labels,
                        f'{category}类高频关键词 TOP10',
                        os.path.join(charts_dir, f'{category}_keywords.png'),
                        horizontal=True)

    # 生成 PDF
    report_path = os.path.join(output_dir, f'{category}类详细分析报告.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ChineseTitle', parent=styles['Title'],
                                fontName=font_name, fontSize=18,
                                alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('ChineseHeading', parent=styles['Heading1'],
                                  fontName=font_name, fontSize=14,
                                  spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle('ChineseBody', parent=styles['Normal'],
                               fontName=font_name, fontSize=11,
                               leading=18, spaceBefore=6, spaceAfter=6)

    story = []

    # 标题
    story.append(Paragraph(f'{category}类客户对话详细分析报告', title_style))
    story.append(Spacer(1, 20))

    # 数据概览
    story.append(Paragraph('一、数据概览', heading_style))
    overview_data = [
        ['指标', '数值'],
        ['对话总数', str(len(category_results))],
        ['平均对话长度', f'{avg_length:.1f} 条'],
        ['二级意图类型数', str(len(secondary_dist))],
    ]
    overview_table = Table(overview_data, colWidths=[200, 200])
    overview_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 15))

    # 二级意图分布
    story.append(Paragraph('二、二级意图分布', heading_style))
    if secondary_dist:
        secondary_data = [['二级意图', '数量', '占比']]
        total = len(category_results)
        for intent, count in secondary_dist[:10]:
            ratio = f'{count/total*100:.1f}%'
            secondary_data.append([intent, str(count), ratio])
        secondary_table = Table(secondary_data, colWidths=[180, 80, 80])
        secondary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(secondary_table)
        story.append(Spacer(1, 10))

        # 添加图表
        chart_path = os.path.join(charts_dir, f'{category}_secondary.png')
        if os.path.exists(chart_path):
            story.append(Image(chart_path, width=400, height=300))
    story.append(Spacer(1, 15))

    # 情感分布
    story.append(Paragraph('三、客户情感分布', heading_style))
    if sentiment_dist:
        sentiment_data = [['情感', '数量', '占比']]
        total = len(category_results)
        for sentiment, count in sentiment_dist.items():
            ratio = f'{count/total*100:.1f}%'
            sentiment_data.append([sentiment, str(count), ratio])
        sentiment_table = Table(sentiment_data, colWidths=[100, 80, 80])
        sentiment_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(sentiment_table)
        story.append(Spacer(1, 10))

        chart_path = os.path.join(charts_dir, f'{category}_sentiment.png')
        if os.path.exists(chart_path):
            story.append(Image(chart_path, width=400, height=300))
    story.append(Spacer(1, 15))

    # 高频问题及客服回答
    story.append(Paragraph('四、高频问题 TOP10及客服回答', heading_style))
    if top_issues_with_replies:
        total = len(category_results)
        for i, (issue, count, replies) in enumerate(top_issues_with_replies, 1):
            ratio = f'{count/total*100:.1f}%'
            # 问题标题
            story.append(Paragraph(f'{i}. {issue}（{count}次，占比{ratio}）', body_style))

            # 客服回答
            if replies:
                story.append(Paragraph('客服典型回答：', body_style))
                for j, reply in enumerate(replies[:2], 1):
                    # 截取回答内容，避免太长
                    reply_text = reply[:150] + '...' if len(reply) > 150 else reply
                    # 清理特殊字符
                    reply_text = clean_text_for_pdf(reply_text)
                    story.append(Paragraph(f'  - {reply_text}', body_style))
            else:
                story.append(Paragraph('  （暂无典型回答）', body_style))
            story.append(Spacer(1, 8))

        # 柱状图
        chart_path = os.path.join(charts_dir, f'{category}_issues.png')
        if os.path.exists(chart_path):
            story.append(Spacer(1, 10))
            story.append(Image(chart_path, width=450, height=300))
    story.append(Spacer(1, 15))

    # 高频关键词
    story.append(Paragraph('五、高频关键词 TOP10', heading_style))
    if top_keywords:
        kw_data = [['关键词', '出现次数']]
        for kw, count in top_keywords[:10]:
            kw_data.append([kw, str(count)])
        kw_table = Table(kw_data, colWidths=[150, 100])
        kw_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(kw_table)
        story.append(Spacer(1, 10))

        chart_path = os.path.join(charts_dir, f'{category}_keywords.png')
        if os.path.exists(chart_path):
            story.append(Image(chart_path, width=450, height=300))
    story.append(Spacer(1, 15))

    # 典型案例
    story.append(Paragraph('六、典型案例', heading_style))
    for i, case in enumerate(typical_cases[:3], 1):
        story.append(Paragraph(f'案例 {i}（二级意图：{case["secondary_intent"]}）', body_style))
        dialog_snippet = extract_dialog_snippet(case['dialog'], 400)
        story.append(Paragraph(dialog_snippet.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 10))

    # 洞察与建议
    story.append(Paragraph('七、分析洞察', heading_style))

    # 根据数据动态生成洞察
    insights = []

    # 情感分析洞察
    if sentiment_dist:
        dissatisfied_ratio = sentiment_dist.get('不满', 0) / len(category_results) * 100
        if dissatisfied_ratio > 15:
            insights.append(f'[警示] 不满客户占比 {dissatisfied_ratio:.1f}%，建议关注服务质量')
        elif dissatisfied_ratio < 5:
            insights.append(f'[良好] 不满客户占比仅 {dissatisfied_ratio:.1f}%，服务满意度较高')

    # 问题集中度洞察
    if top_issues_with_replies:
        top_issue_ratio = top_issues_with_replies[0][1] / len(category_results) * 100
        if top_issue_ratio > 30:
            insights.append(f'[建议] 问题集中度高："{top_issues_with_replies[0][0]}"占比{top_issue_ratio:.1f}%，建议重点优化')

    # 平均对话长度洞察
    if avg_length > 20:
        insights.append(f'[效率] 平均对话长度 {avg_length:.1f} 条，客户问题较复杂，建议提升客服效率')
    elif avg_length < 10:
        insights.append(f'[效率] 平均对话长度 {avg_length:.1f} 条，对话简洁，服务效率较高')

    for insight in insights:
        story.append(Paragraph(insight, body_style))

    doc.build(story)
    return report_path


def generate_summary_report(results, output_dir, font_name, source_name):
    """生成总览汇总报告"""
    # 统计数据
    category_dist = Counter(r['primary_intent'] for r in results)
    all_secondary_dist = get_secondary_distribution(results)
    all_sentiment_dist = get_sentiment_distribution(results)
    all_top_issues_with_replies = get_top_issues_with_replies(results, 15)
    all_top_keywords = get_top_keywords(results, 30)

    # 创建图表
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # 一级分类饼图
    if category_dist:
        create_pie_chart(list(category_dist.values()), list(category_dist.keys()),
                        '一级意图分类分布', os.path.join(charts_dir, 'summary_primary.png'))

    # 全量高频问题
    if all_top_issues_with_replies:
        issue_data = [c for _, c, _ in all_top_issues_with_replies[:10]]
        issue_labels = [l for l, _, _ in all_top_issues_with_replies[:10]]
        create_bar_chart(issue_data, issue_labels,
                        '全量高频问题 TOP10',
                        os.path.join(charts_dir, 'summary_issues.png'),
                        horizontal=True)

    # 生成汇总 PDF
    report_path = os.path.join(output_dir, f'{source_name}总览汇总报告.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                           leftMargin=2*cm, rightMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ChineseTitle', parent=styles['Title'],
                                fontName=font_name, fontSize=18,
                                alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('ChineseHeading', parent=styles['Heading1'],
                                  fontName=font_name, fontSize=14,
                                  spaceBefore=15, spaceAfter=10)
    body_style = ParagraphStyle('ChineseBody', parent=styles['Normal'],
                               fontName=font_name, fontSize=11,
                               leading=18, spaceBefore=6, spaceAfter=6)

    story = []

    story.append(Paragraph(f'{source_name}总览汇总报告', title_style))
    story.append(Spacer(1, 20))

    # 总体概览
    story.append(Paragraph('一、总体数据概览', heading_style))
    avg_length = sum(r['line_count'] for r in results) / len(results)
    overview_data = [
        ['指标', '数值'],
        ['对话总数', str(len(results))],
        ['平均对话长度', f'{avg_length:.1f} 条'],
        ['一级意图类型', str(len(category_dist))],
        ['二级意图类型', str(len(all_secondary_dist))],
    ]
    overview_table = Table(overview_data, colWidths=[200, 200])
    overview_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 15))

    # 一级分类分布
    story.append(Paragraph('二、一级意图分类分布', heading_style))
    primary_data = [['一级分类', '数量', '占比']]
    total = len(results)
    for intent in PRIMARY_CATEGORIES:
        count = category_dist.get(intent, 0)
        ratio = f'{count/total*100:.1f}%'
        primary_data.append([intent, str(count), ratio])
    primary_table = Table(primary_data, colWidths=[100, 80, 80])
    primary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(primary_table)
    story.append(Spacer(1, 10))

    chart_path = os.path.join(charts_dir, 'summary_primary.png')
    if os.path.exists(chart_path):
        story.append(Image(chart_path, width=400, height=300))
    story.append(Spacer(1, 15))

    # 全量高频问题及客服回答
    story.append(Paragraph('三、全量高频问题 TOP15及客服回答', heading_style))
    if all_top_issues_with_replies:
        for i, (issue, count, replies) in enumerate(all_top_issues_with_replies[:15], 1):
            ratio = f'{count/total*100:.1f}%'
            # 问题标题
            story.append(Paragraph(f'{i}. {issue}（{count}次，占比{ratio}）', body_style))

            # 客服回答
            if replies:
                story.append(Paragraph('客服典型回答：', body_style))
                for j, reply in enumerate(replies[:2], 1):
                    reply_text = reply[:150] + '...' if len(reply) > 150 else reply
                    # 清理特殊字符
                    reply_text = clean_text_for_pdf(reply_text)
                    story.append(Paragraph(f'  - {reply_text}', body_style))
            story.append(Spacer(1, 6))

        # 柱状图
        chart_path = os.path.join(charts_dir, 'summary_issues.png')
        if os.path.exists(chart_path):
            story.append(Spacer(1, 10))
            story.append(Image(chart_path, width=450, height=300))
    story.append(Spacer(1, 15))

    # 全量情感分布
    story.append(Paragraph('四、全量情感分布', heading_style))
    if all_sentiment_dist:
        sentiment_data = [['情感', '数量', '占比']]
        for sentiment, count in all_sentiment_dist.items():
            ratio = f'{count/total*100:.1f}%'
            sentiment_data.append([sentiment, str(count), ratio])
        sentiment_table = Table(sentiment_data, colWidths=[100, 80, 80])
        sentiment_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(sentiment_table)
    story.append(Spacer(1, 15))

    # 各分类详细报告链接
    story.append(Paragraph('五、各分类详细报告', heading_style))
    for category in PRIMARY_CATEGORIES:
        count = category_dist.get(category, 0)
        if count > 0:
            story.append(Paragraph(f'{category}类：{count} 条对话 → 查看 {category}类详细分析报告.pdf', body_style))

    doc.build(story)
    return report_path


# ==================== 主流程 ====================

def analyze_directory(csv_dir, limit=None):
    """分析指定目录的 CSV 文件"""
    global progress_counter, total_tokens_input, total_tokens_output

    # 重置计数器
    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    # 获取文件列表
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    if limit:
        csv_files = csv_files[:limit]

    total_files = len(csv_files)
    print(f"\n{'='*60}")
    print(f"详细分析目录: {csv_dir}")
    print(f"文件数量: {total_files}")
    print(f"{'='*60}\n")

    # 并发处理
    results = []
    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, f): f for f in csv_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

            # 进度显示
            if progress_counter % 100 == 0 or progress_counter == total_files:
                print(f"进度: {progress_counter}/{total_files}")

    print(f"\n处理完成！成功: {len(results)}, 失败: {total_files - len(results)}")

    # 计算成本
    total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE
    print(f"API 成本: ¥{total_cost:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="客服对话详细分析脚本")
    parser.add_argument("--dir", choices=['dialog_csv', 'online_chat_csv', 'all'],
                       default='online_chat_csv', help="分析目录")
    parser.add_argument("--limit", type=int, default=None, help="限制处理文件数")
    args = parser.parse_args()

    # 注册中文字体
    font_name = register_chinese_font()

    # 确定输出目录
    output_dir = REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 确定数据目录
    if args.dir == 'dialog_csv':
        csv_dirs = [DIALOG_CSV_DIR]
        source_names = ['电话录音']
    elif args.dir == 'online_chat_csv':
        csv_dirs = [ONLINE_CHAT_CSV_DIR]
        source_names = ['在线对话']
    else:
        csv_dirs = [DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR]
        source_names = ['电话录音', '在线对话']

    all_results = []

    for csv_dir, source_name in zip(csv_dirs, source_names):
        print(f"\n{'='*60}")
        print(f"分析: {source_name}")
        print(f"{'='*60}")

        results = analyze_directory(csv_dir, args.limit)

        if not results:
            print(f"没有有效数据，跳过 {source_name}")
            continue

        # 按分类生成详细报告
        print(f"\n生成各分类详细报告...")
        category_reports = []
        for category in PRIMARY_CATEGORIES:
            report_path = generate_category_report(category, results, output_dir, font_name)
            if report_path:
                category_reports.append(report_path)
                print(f"  ✓ {category}类详细分析报告.pdf")

        # 生成汇总报告
        print(f"\n生成总览汇总报告...")
        summary_path = generate_summary_report(results, output_dir, font_name, source_name)
        print(f"  ✓ {source_name}总览汇总报告.pdf")

        all_results.extend(results)

        print(f"\n报告目录: {output_dir}")
        print(f"生成的报告:")
        print(f"  - {source_name}总览汇总报告.pdf")
        for category in PRIMARY_CATEGORIES:
            category_count = len([r for r in results if r['primary_intent'] == category])
            if category_count > 0:
                print(f"  - {category}类详细分析报告.pdf ({category_count} 条对话)")

    # 如果分析全部数据，生成全量汇总
    if args.dir == 'all' and all_results:
        print(f"\n生成全量汇总报告...")
        generate_summary_report(all_results, output_dir, font_name, '全量')


if __name__ == "__main__":
    main()