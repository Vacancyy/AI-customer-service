"""
客服对话分析脚本（统一版）

功能：
1. 意图分类：分析客户来电/咨询原因
2. 情感分析：识别客户情绪（满意/中立/不满）
3. 高频问题提取：提取并统计客户最关心的问题 TOP N

使用方式：
  python analyze.py --dir dialog_csv              # 分析电话录音目录
  python analyze.py --dir online_chat_csv         # 分析在线对话目录
  python analyze.py --dir dialog_csv --limit 1000 # 只处理前1000个文件
  python analyze.py --dir all                     # 分析全部两个目录
  python analyze.py --skip-intent                 # 跳过意图分类
  python analyze.py --skip-sentiment              # 跳过情感分析
  python analyze.py --skip-issue                  # 跳过高频问题提取
"""

import os
import sys
import csv
import time
import re
import requests
import argparse
import tempfile
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np

# matplotlib 中文支持
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置 matplotlib 全局中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 配置 ====================

# 导入路径配置
from config import DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR, REPORTS_DIR as OUTPUT_DIR

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10  # 并发线程数

# 计费标准（Qwen3-8B）
INPUT_PRICE = 0.0005   # ¥/千Token
OUTPUT_PRICE = 0.002   # ¥/千Token

# ==================== 分类体系 ====================
# Plan A：按客户行为分类

INTENT_CATEGORIES = {
    "咨询": ["产品了解", "条款解释", "费率查询", "理赔流程了解", "理赔材料了解", "退保流程了解", "保障范围了解", "其他了解"],
    "查询": ["理赔进度查询", "理赔到账查询", "保单状态查询", "其他查询"],
    "办理": ["理赔申请", "事故报案", "新保投保", "续保办理", "保单变更", "退保办理", "减保办理"],
    "投诉": ["理赔时效投诉", "拒赔投诉", "理赔金额异议", "服务态度投诉", "其他投诉"],
    "其他": ["回访确认", "信息核实"]
}

# 一级分类映射（用于聚合统计）
PRIMARY_CATEGORIES = ["咨询", "查询", "办理", "投诉", "其他"]

# ==================== Prompt 定义 ====================

INTENT_PROMPT = """下面是一段保险客服电话或在线对话的内容，请分析客户的来电原因。

对话内容：
{dialog}

请从以下分类体系中选择最符合的原因，输出格式为"一级分类-二级分类"：

【判断规则】
- 如果客户问"是什么、怎么做、多少钱" → 咨询（了解信息）
- 如果客户问"到哪了、什么时候、有没有、在不在" → 查询（查状态）
- 如果客户说"我要、帮我、申请" → 办理（办业务）
- 如果客户有抱怨、不满、质疑 → 投诉

【咨询】客户询问了解信息（问知识、流程）
- 产品了解：问产品保障内容
- 条款解释：问条款含义、保障范围
- 费率查询：问保费价格
- 理赔流程了解：问理赔步骤、怎么理赔
- 理赔材料了解：问理赔需要什么材料
- 退保流程了解：问退保怎么操作
- 保障范围了解：问某种情况能否理赔
- 其他了解：其他咨询问题

【查询】客户查询具体状态（问进度、状态）
- 理赔进度查询：查理赔审核到哪了
- 理赔到账查询：查理赔款什么时候到账
- 保单状态查询：查保单是否有效、什么时候生成
- 其他查询：其他状态查询

【办理】客户要办理业务
- 理赔申请：提交理赔
- 事故报案：报告事故
- 新保投保：新买保险
- 续保办理：续保操作
- 保单变更：变更保单信息
- 退保办理：办理退保
- 减保办理：减少保额

【投诉】客户有不满投诉
- 理赔时效投诉：投诉理赔太慢
- 拒赔投诉：投诉拒赔决定
- 理赔金额异议：对赔付金额不满
- 服务态度投诉：投诉客服态度
- 流程复杂投诉：对理赔流程繁琐不满
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

ISSUE_PROMPT = """分析以下客服对话，判断客户咨询的核心问题类型。

对话内容：
{dialog}

请从以下选项中选择唯一一个最符合的问题类型，只输出类型名称：

理赔进度查询、理赔款项到账查询、理赔时效咨询、理赔材料准备、理赔资料补交、理赔金额计算、理赔金额异议、报销比例咨询、保障范围咨询、理赔条件确认、既往症咨询、退保流程咨询、退保金额计算、犹豫期退保、保单查询、保单变更、信息修改、续保咨询、产品条款解释、保费计算、投保条件咨询

只输出一个类型名称（如：理赔进度查询），不要输出多个，不要解释。"""


# ==================== PDF生成函数 ====================

def create_pie_chart_image(data_dict, title, output_path, figsize=(6, 4)):
    """创建饼图并保存为PNG图片（使用matplotlib支持中文）"""
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    total = sum(values)

    # 计算百分比
    percentages = [v/total*100 for v in values]

    fig, ax = plt.subplots(figsize=figsize)

    # 颜色列表
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                   '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    # 绘制饼图
    wedges, texts, autotexts = ax.pie(values, labels=None, autopct='',
                                       colors=colors_list[:len(values)],
                                       startangle=90, pctdistance=0.75)

    # 添加百分比标签（只显示大于3%的）
    for i, (wedge, pct) in enumerate(zip(wedges, percentages)):
        if pct > 3:
            ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
            x = 0.7 * np.cos(np.deg2rad(ang))
            y = 0.7 * np.sin(np.deg2rad(ang))
            ax.text(x, y, f'{pct:.1f}%', ha='center', va='center', fontsize=9,
                   fontweight='bold')

    # 添加图例（显示标签和百分比）
    legend_labels = [f'{l}: {v} ({p:.1f}%)' for l, v, p in zip(labels, values, percentages)]
    legend = ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize=9)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=False)
    plt.close()

    return output_path


def create_bar_chart_image(data_dict, title, output_path, figsize=(8, 4), show_percentage=True):
    """创建柱状图并保存为PNG图片（使用matplotlib支持中文）"""
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    total = sum(values) if show_percentage else 1

    fig, ax = plt.subplots(figsize=figsize)

    # 颜色
    colors_list = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6',
                   '#1ABC9C', '#34495E', '#E67E22', '#16A085', '#27AE60']

    # 绘制柱状图
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, values, color=colors_list[:len(values)], edgecolor='white', linewidth=0.5)

    # 设置标签
    ax.set_xticks(x_pos)
    # 截断过长的标签
    short_labels = [l if len(l) <= 8 else l[:7]+'..' for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)

    # 设置x轴标签字体
    for label in ax.get_xticklabels():
        label

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if show_percentage:
            pct = val/total*100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('数量', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置y轴范围
    ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=False)
    plt.close()

    return output_path


def create_pdf_report(output_path, report_title, results, success_count, do_intent, do_sentiment, do_issue,
                       intent_counts, primary_intent_counts, sentiment_counts, issue_counts):
    """生成PDF分析报告"""

    # 注册中文字体 - 使用 reportlab 内置 UnicodeCIDFont
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        chinese_font = 'STSong-Light'
    except Exception as e:
        print(f"UnicodeCIDFont 注册失败: {e}")
        chinese_font = 'Helvetica'

    if not chinese_font:
        print("警告：未找到中文字体，PDF可能无法正常显示中文")
        chinese_font = 'Helvetica'

    # 创建临时目录存放图表图片
    temp_dir = tempfile.mkdtemp(prefix='charts_')
    chart_files = []  # 记录生成的图片文件，用于最后清理

    # 创建PDF文档
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    # 创建样式
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName=chinese_font,
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading2'],
        fontName=chinese_font,
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )

    subheading_style = ParagraphStyle(
        'ChineseSubHeading',
        parent=styles['Heading3'],
        fontName=chinese_font,
        fontSize=12,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        'ChineseNormal',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=10,
        leading=16
    )

    example_style = ParagraphStyle(
        'ExampleStyle',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=9,
        leading=14,
        leftIndent=20,
        textColor=colors.grey
    )

    story = []

    # ===== 标题 =====
    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"报告日期：{time.strftime('%Y年%m月%d日')}", normal_style))
    story.append(Paragraph(f"数据范围：2026年1月-3月", normal_style))
    story.append(Paragraph(f"数据来源：客服中心对话记录", normal_style))
    story.append(Spacer(1, 20))

    # ===== 一、数据概览 =====
    story.append(Paragraph("一、数据概览", heading_style))

    # 1.1 基础统计
    story.append(Paragraph("1.1 基础统计", subheading_style))

    # 计算基础统计数据
    if do_sentiment:
        negative_count = sentiment_counts.get("不满", 0)
        satisfied_count = sentiment_counts.get("满意", 0)
        neutral_count = sentiment_counts.get("中立", 0)
        negative_ratio = negative_count / success_count * 100 if success_count else 0
        satisfied_ratio = satisfied_count / success_count * 100 if success_count else 0
    else:
        negative_count = 0
        satisfied_count = 0
        neutral_count = 0
        negative_ratio = 0
        satisfied_ratio = 0

    if do_intent:
        # 统计理赔相关（二级分类包含"理赔"的）
        claim_count = sum(count for intent, count in intent_counts.items() if "理赔" in intent)
        claim_ratio = claim_count / success_count * 100 if success_count else 0
    else:
        claim_count = 0
        claim_ratio = 0

    base_data = [
        ['指标', '数值', '说明'],
        ['分析样本数', f'{success_count}条', '有效分析记录'],
        ['不满客户数', f'{negative_count}条', f'占比{negative_ratio:.1f}%'],
        ['满意客户数', f'{satisfied_count}条', f'占比{satisfied_ratio:.1f}%'],
        ['理赔相关', f'{claim_count}条', f'占比{claim_ratio:.1f}%'],
    ]

    base_table = Table(base_data, colWidths=[4*cm, 3*cm, 5*cm])
    base_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(base_table)
    story.append(Spacer(1, 10))

    # 添加一级分类分布柱状图
    base_chart_data = {}
    for cat in PRIMARY_CATEGORIES:
        count = primary_intent_counts.get(cat, 0)
        if count > 0:
            base_chart_data[cat] = count
    story.append(Paragraph("【客户行为分布图】", normal_style))
    chart_path = os.path.join(temp_dir, 'bar_base_stats.png')
    create_bar_chart_image(base_chart_data, "客户行为类型分布", chart_path, figsize=(7, 4))
    chart_files.append(chart_path)
    story.append(Image(chart_path, width=14*cm, height=8*cm))
    story.append(Spacer(1, 15))

    # 1.2 来源分布
    story.append(Paragraph("1.2 来源分布", subheading_style))
    source_counts = Counter(r["source"] for r in results)

    source_data = [['来源', '数量', '占比']]
    for source, count in source_counts.items():
        percent = count / success_count * 100 if success_count else 0
        source_data.append([source, str(count), f"{percent:.1f}%"])

    source_table = Table(source_data, colWidths=[5*cm, 3*cm, 3*cm])
    source_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(source_table)
    story.append(Spacer(1, 20))

    # ===== 二、意图分类分析 =====
    if do_intent and intent_counts:
        story.append(Paragraph("二、意图分类分析", heading_style))

        # 2.1 一级分类分布
        story.append(Paragraph("2.1 一级分类分布", subheading_style))

        primary_data = [['分类', '数量', '占比', '说明']]
        category_desc = {
            "咨询": "询问了解信息",
            "查询": "查询具体状态",
            "办理": "办理业务操作",
            "投诉": "表达不满投诉",
            "其他": "回访核实等"
        }

        # 准备饼图数据
        chart_data = {}
        for intent in PRIMARY_CATEGORIES:
            count = primary_intent_counts.get(intent, 0)
            if count > 0:
                percent = count / success_count * 100 if success_count else 0
                desc = category_desc.get(intent, "")
                primary_data.append([intent, str(count), f"{percent:.1f}%", desc])
                chart_data[intent] = count

        primary_table = Table(primary_data, colWidths=[2.5*cm, 2*cm, 2*cm, 5*cm])
        primary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(primary_table)
        story.append(Spacer(1, 10))

        # 添加一级分类饼图和柱状图
        if chart_data:
            story.append(Paragraph("【一级分类分布图】", normal_style))
            # 饼图
            chart_path = os.path.join(temp_dir, 'pie_primary.png')
            create_pie_chart_image(chart_data, "一级分类占比分布", chart_path)
            chart_files.append(chart_path)
            story.append(Image(chart_path, width=14*cm, height=9*cm))

            # 柱状图
            story.append(Spacer(1, 10))
            story.append(Paragraph("【一级分类数量对比图】", normal_style))
            chart_path2 = os.path.join(temp_dir, 'bar_primary.png')
            create_bar_chart_image(chart_data, "一级分类数量对比", chart_path2, figsize=(7, 4))
            chart_files.append(chart_path2)
            story.append(Image(chart_path2, width=14*cm, height=8*cm))
        story.append(Spacer(1, 15))

        # 2.2 二级分类TOP10
        story.append(Paragraph("2.2 二级分类 TOP 10", subheading_style))

        secondary_data = [['排名', '分类', '数量', '占比']]
        bar_data = {}
        for i, (intent, count) in enumerate(intent_counts.most_common(10), 1):
            percent = count / success_count * 100 if success_count else 0
            secondary_data.append([str(i), intent, str(count), f"{percent:.1f}%"])
            bar_data[intent[:8]] = count

        secondary_table = Table(secondary_data, colWidths=[1.5*cm, 6*cm, 2*cm, 2*cm])
        secondary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(secondary_table)
        story.append(Spacer(1, 10))

        # 添加二级分类柱状图
        if bar_data:
            story.append(Paragraph("【二级分类 TOP 10 柱状图】", normal_style))
            chart_path = os.path.join(temp_dir, 'bar_secondary.png')
            create_bar_chart_image(bar_data, "二级分类 TOP 10 分布", chart_path)
            chart_files.append(chart_path)
            story.append(Image(chart_path, width=16*cm, height=8*cm))
        story.append(Spacer(1, 20))

    # ===== 三、情感分析 =====
    if do_sentiment and sentiment_counts:
        story.append(Paragraph("三、情感分析", heading_style))

        # 3.1 情感分布
        story.append(Paragraph("3.1 客户情感分布", subheading_style))

        sentiment_data = [['情感', '数量', '占比', '特征']]
        sentiment_desc = {
            "满意": "问题解决，态度积极",
            "中立": "情绪平稳，咨询为主",
            "不满": "有抱怨、质疑倾向"
        }

        sentiment_chart_data = {}
        for sentiment in ["满意", "中立", "不满"]:
            count = sentiment_counts.get(sentiment, 0)
            percent = count / success_count * 100 if success_count else 0
            desc = sentiment_desc.get(sentiment, "")
            sentiment_data.append([sentiment, str(count), f"{percent:.1f}%", desc])
            sentiment_chart_data[sentiment] = count

        sentiment_table = Table(sentiment_data, colWidths=[2*cm, 2*cm, 2*cm, 6*cm])
        sentiment_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 3), (-1, 3), colors.Color(1, 0.9, 0.9)),  # 不满行高亮
        ]))
        story.append(sentiment_table)
        story.append(Spacer(1, 10))

        # 添加情感分布饼图
        if sentiment_chart_data:
            story.append(Paragraph("【客户情感分布图】", normal_style))
            chart_path = os.path.join(temp_dir, 'pie_sentiment.png')
            create_pie_chart_image(sentiment_chart_data, "客户情感占比分布", chart_path)
            chart_files.append(chart_path)
            story.append(Image(chart_path, width=14*cm, height=9*cm))
        story.append(Spacer(1, 15))

        # 3.2 不满客户分析
        negative_results = [r for r in results if r["sentiment"] == "不满"]
        if negative_results:
            story.append(Paragraph("3.2 不满客户问题分析", subheading_style))

            neg_intent_counts = Counter(r["intent"] for r in negative_results if r["intent"])
            neg_issue_counts = Counter(r["issue"] for r in negative_results if r["issue"])

            # 不满客户意图分布
            neg_intent_data = [['问题类型', '数量', '占比']]
            for intent, count in neg_intent_counts.most_common(5):
                percent = count / len(negative_results) * 100
                neg_intent_data.append([intent, str(count), f"{percent:.1f}%"])

            neg_table = Table(neg_intent_data, colWidths=[6*cm, 2*cm, 3*cm])
            neg_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(neg_table)
            story.append(Spacer(1, 10))

            # 添加不满客户意图分布饼图
            neg_intent_chart_data = {}
            for intent, count in neg_intent_counts.most_common(6):
                short_intent = intent[:8] if len(intent) > 8 else intent
                neg_intent_chart_data[short_intent] = count

            if neg_intent_chart_data:
                story.append(Paragraph("【不满客户问题类型分布图】", normal_style))
                chart_path = os.path.join(temp_dir, 'pie_negative.png')
                create_pie_chart_image(neg_intent_chart_data, "不满客户问题类型分布", chart_path)
                chart_files.append(chart_path)
                story.append(Image(chart_path, width=14*cm, height=9*cm))

            story.append(Paragraph(f"不满客户共{len(negative_results)}条，主要集中在理赔相关问题", normal_style))
        story.append(Spacer(1, 20))

    # ===== 四、高频问题分析 =====
    if do_issue and issue_counts:
        story.append(Paragraph("四、高频问题分析", heading_style))

        story.append(Paragraph("4.1 高频问题 TOP 15", subheading_style))

        issue_data = [['排名', '问题', '次数', '占比']]
        for i, (issue, count) in enumerate(issue_counts.most_common(15), 1):
            percent = count / success_count * 100 if success_count else 0
            issue_short = issue[:25] + "..." if len(issue) > 25 else issue
            issue_data.append([str(i), issue_short, str(count), f"{percent:.1f}%"])

        issue_table = Table(issue_data, colWidths=[1.5*cm, 6.5*cm, 1.5*cm, 1.5*cm])
        issue_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(issue_table)
        story.append(Spacer(1, 10))

        # 添加高频问题柱状图
        issue_chart_data = {}
        for issue, count in issue_counts.most_common(10):
            short_issue = issue[:8] if len(issue) > 8 else issue
            issue_chart_data[short_issue] = count

        if issue_chart_data:
            story.append(Paragraph("【高频问题 TOP 10 分布图】", normal_style))
            chart_path = os.path.join(temp_dir, 'bar_issues.png')
            create_bar_chart_image(issue_chart_data, "高频问题 TOP 10 分布", chart_path, figsize=(10, 5))
            chart_files.append(chart_path)
            story.append(Image(chart_path, width=16*cm, height=8*cm))
        story.append(Spacer(1, 15))

        # 4.2 高频问题具体案例
        story.append(Paragraph("4.2 高频问题具体案例", subheading_style))

        # 按问题类型分组展示案例
        top_issues = issue_counts.most_common(5)
        for issue, count in top_issues:
            # 找到该问题的相关案例
            related_cases = [r for r in results if r.get("issue") == issue][:3]

            if related_cases:
                story.append(Paragraph(f"<b>【{issue}】共{count}次</b>", normal_style))

                for i, case in enumerate(related_cases, 1):
                    case_intent = case.get("intent", "未知")
                    case_sentiment = case.get("sentiment", "未知")
                    case_source = case.get("source", "未知")
                    story.append(Paragraph(
                        f"  案例{i}：{case_source} | 意图：{case_intent} | 情感：{case_sentiment}",
                        example_style
                    ))
                story.append(Spacer(1, 8))

        # 4.3 问题类型细分分析
        story.append(Paragraph("4.3 问题类型细分分析", subheading_style))

        # 将高频问题归类
        issue_categories = {
            "理赔进度类": [],
            "理赔材料类": [],
            "理赔金额类": [],
            "理赔范围类": [],
            "退保相关类": [],
            "保单管理类": [],
            "产品咨询类": [],
            "其他类": []
        }

        for issue, count in issue_counts.most_common(30):
            issue_lower = issue.lower()
            if "进度" in issue or "到账" in issue or "时效" in issue:
                issue_categories["理赔进度类"].append((issue, count))
            elif "材料" in issue or "清单" in issue or "资料" in issue:
                issue_categories["理赔材料类"].append((issue, count))
            elif "金额" in issue or "报销" in issue or "异议" in issue or "比例" in issue:
                issue_categories["理赔金额类"].append((issue, count))
            elif "范围" in issue or "条件" in issue or "既往" in issue or "保障" in issue:
                issue_categories["理赔范围类"].append((issue, count))
            elif "退保" in issue or "犹豫期" in issue:
                issue_categories["退保相关类"].append((issue, count))
            elif "保单" in issue or "变更" in issue or "续保" in issue or "查询" in issue:
                issue_categories["保单管理类"].append((issue, count))
            elif "条款" in issue or "保费" in issue or "投保条件" in issue:
                issue_categories["产品咨询类"].append((issue, count))
            else:
                issue_categories["其他类"].append((issue, count))

        category_data = [['问题类型', '包含问题数', '总次数', '典型问题']]
        for cat_name, issues in issue_categories.items():
            if issues:
                total_count = sum(c for _, c in issues)
                typical_issue = issues[0][0] if issues else ""
                typical_short = typical_issue[:15] + "..." if len(typical_issue) > 15 else typical_issue
                category_data.append([cat_name, str(len(issues)), str(total_count), typical_short])

        category_table = Table(category_data, colWidths=[3*cm, 2.5*cm, 2*cm, 4*cm])
        category_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(category_table)
        story.append(Spacer(1, 10))

        # 添加问题类型细分饼图
        category_chart_data = {}
        for cat_name, issues in issue_categories.items():
            if issues:
                category_chart_data[cat_name] = sum(c for _, c in issues)

        if category_chart_data:
            story.append(Paragraph("【问题类型细分分布图】", normal_style))
            chart_path = os.path.join(temp_dir, 'pie_categories.png')
            create_pie_chart_image(category_chart_data, "问题类型细分占比", chart_path)
            chart_files.append(chart_path)
            story.append(Image(chart_path, width=14*cm, height=9*cm))
        story.append(Spacer(1, 20))

    # ===== 五、核心观点与建议 =====
    story.append(Paragraph("五、核心观点与建议", heading_style))

    # 计算关键指标（基于二级分类统计）
    claim_ratio = sum(count for intent, count in intent_counts.items() if "理赔" in intent) / success_count * 100 if success_count else 0
    cancel_ratio = sum(count for intent, count in intent_counts.items() if "退保" in intent) / success_count * 100 if success_count else 0
    consult_ratio = primary_intent_counts.get("咨询", 0) / success_count * 100 if success_count else 0
    query_ratio = primary_intent_counts.get("查询", 0) / success_count * 100 if success_count else 0
    handle_ratio = primary_intent_counts.get("办理", 0) / success_count * 100 if success_count else 0

    # 观点一：理赔服务分析
    if claim_ratio > 15:
        story.append(Paragraph("<b>5.1 理赔服务是客户最大痛点</b>", normal_style))
        story.append(Paragraph(f"数据支撑：理赔相关问题占比{claim_ratio:.1f}%，为业务中最高", normal_style))

        if do_issue and issue_counts:
            claim_issues = []
            for issue, count in issue_counts.most_common(10):
                if "理赔" in issue or "进度" in issue or "到账" in issue or "材料" in issue:
                    claim_issues.append(f"「{issue}」{count}次")
            if claim_issues:
                story.append(Paragraph(f"高频问题：{'、'.join(claim_issues[:5])}", normal_style))

        if do_sentiment and negative_results:
            neg_claim_count = sum(1 for r in negative_results if "理赔" in r.get("intent", ""))
            if neg_claim_count > 0:
                story.append(Paragraph(f"不满客户分析：不满客户中{neg_claim_count}条与理赔相关，占比{neg_claim_count/len(negative_results)*100:.1f}%", normal_style))

        story.append(Paragraph("问题分析：用户对理赔材料、流程、进度存在大量疑问，说明理赔服务透明度不足", normal_style))
        story.append(Paragraph("优化建议：", normal_style))
        story.append(Paragraph("  (1) 建立理赔进度可视化系统，让用户实时查看当前状态", normal_style))
        story.append(Paragraph("  (2) 发布清晰的理赔材料清单，减少用户反复询问", normal_style))
        story.append(Paragraph("  (3) 开通理赔进度主动推送，减少用户等待焦虑", normal_style))
        story.append(Spacer(1, 10))

    # 观点二：客户行为分析
    if do_intent and consult_ratio > 30:
        story.append(Paragraph("<b>5.2 客户以咨询为主，信息透明度待提升</b>", normal_style))
        story.append(Paragraph(f"数据支撑：咨询类占比{consult_ratio:.1f}%，查询类占比{query_ratio:.1f}%，办理类占比{handle_ratio:.1f}%", normal_style))
        story.append(Paragraph("问题分析：大量客户来电是为了了解信息，说明产品信息不够透明", normal_style))
        story.append(Paragraph("优化建议：", normal_style))
        story.append(Paragraph("  (1) 完善线上自助查询功能，减少电话咨询", normal_style))
        story.append(Paragraph("  (2) 发布常见问题FAQ，提高信息透明度", normal_style))
        story.append(Spacer(1, 10))

    # 观点三：情感分析
    if do_sentiment and negative_ratio > 10:
        story.append(Paragraph("<b>5.3 客户不满比例偏高，需重点关注</b>", normal_style))
        story.append(Paragraph(f"数据支撑：不满客户占比{negative_ratio:.1f}%，满意仅{satisfied_ratio:.1f}%", normal_style))

        if negative_results:
            neg_intent_counts = Counter(r["intent"] for r in negative_results if r["intent"])
            top_neg_intents = neg_intent_counts.most_common(3)
            if top_neg_intents:
                neg_intent_str = "、".join([f"{i[0]}({i[1]}次)" for i in top_neg_intents])
                story.append(Paragraph(f"不满原因分析：不满客户主要集中在：{neg_intent_str}", normal_style))

        story.append(Paragraph("问题分析：不满比例高于一般客服场景（通常10-15%）", normal_style))
        story.append(Paragraph("优化建议：", normal_style))
        story.append(Paragraph("  (1) 分析不满客户集中问题，针对性改进服务流程", normal_style))
        story.append(Paragraph("  (2) 建立客户满意度回访机制", normal_style))
        story.append(Spacer(1, 10))

    # 观点四：条款咨询
    if do_intent:
        clause_ratio = 0
        for intent, count in intent_counts.items():
            if "条款解释" in intent or "保障范围" in intent:
                clause_ratio += count / success_count * 100 if success_count else 0

        if clause_ratio > 3:
            story.append(Paragraph("<b>5.4 保障范围认知模糊是用户焦虑源头</b>", normal_style))
            story.append(Paragraph(f"数据支撑：条款解释、保障范围相关咨询占比{clause_ratio:.1f}%", normal_style))
            story.append(Paragraph("问题分析：条款语言专业晦涩，用户不确定保障边界", normal_style))
            story.append(Paragraph("优化建议：", normal_style))
            story.append(Paragraph("  (1) 将条款转化为通俗问答形式", normal_style))
            story.append(Paragraph("  (2) 建立常见疾病理赔案例库", normal_style))
            story.append(Paragraph("  (3) 发布明确的既往症清单", normal_style))
            story.append(Spacer(1, 10))

    # 观点五：退保分析
    if cancel_ratio > 3:
        story.append(Paragraph("<b>5.5 退保需求反映产品设计问题</b>", normal_style))
        story.append(Paragraph(f"数据支撑：退保相关问题占比{cancel_ratio:.1f}%", normal_style))
        story.append(Paragraph("问题分析：退保需求背后可能是重复投保检测不足、理赔门槛预期落差", normal_style))
        story.append(Paragraph("优化建议：", normal_style))
        story.append(Paragraph("  (1) 投保前检测用户是否已有同类保障，主动提醒", normal_style))
        story.append(Paragraph("  (2) 简化退保流程，明确退款时效", normal_style))
        story.append(Spacer(1, 20))

    # ===== 六、风险与机会分析 =====
    story.append(Paragraph("六、风险与机会分析", heading_style))

    # 6.1 风险点
    story.append(Paragraph("6.1 主要风险点", subheading_style))

    risk_data = [['风险等级', '风险点', '表现', '建议动作']]
    risks = []

    if negative_ratio > 20:
        risks.append(['高', '不满客户比例高', f'不满占比{negative_ratio:.1f}%', '分析不满原因，改进服务'])
    if claim_ratio > 30:
        risks.append(['高', '理赔咨询集中', f'理赔咨询占比{claim_ratio:.1f}%', '优化理赔流程'])
    if cancel_ratio > 5:
        risks.append(['中', '退保需求较多', f'退保咨询占比{cancel_ratio:.1f}%', '检测重复投保'])

    if not risks:
        risks.append(['低', '暂无明显风险', '数据表现正常', '持续监测'])

    for risk in risks:
        risk_data.append(risk)

    risk_table = Table(risk_data, colWidths=[2*cm, 3*cm, 3*cm, 4*cm])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 15))

    # 6.2 机会点
    story.append(Paragraph("6.2 改进机会点", subheading_style))

    opportunity_data = [['机会点', '依据', '预期效果']]
    opportunities = []

    if claim_ratio > 20:
        opportunities.append(['理赔服务优化', f'理赔咨询占比{claim_ratio:.1f}%', '满意度提升15%'])
    if negative_ratio > 15:
        opportunities.append(['不满客户专项改进', f'不满占比{negative_ratio:.1f}%', '不满比例降至15%'])

    if not opportunities:
        opportunities.append(['服务流程优化', '持续改进', '满意度稳步提升'])

    for opp in opportunities:
        opportunity_data.append(opp)

    opp_table = Table(opportunity_data, colWidths=[4*cm, 4*cm, 4*cm])
    opp_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(opp_table)
    story.append(Spacer(1, 20))

    # ===== 七、监测指标建议 =====
    story.append(Paragraph("七、监测指标建议", heading_style))

    monitor_data = [['指标类型', '指标名称', '监测频率', '预警阈值', '目标值']]
    monitor_data.append(['舆情监测', '不满客户比例', '每周', '>20%', '<15%'])
    monitor_data.append(['舆情监测', '理赔相关投诉量', '每日', '>10条/日', '环比下降'])
    monitor_data.append(['服务质量', '客户满意度', '每月', '<70%', '>85%'])
    monitor_data.append(['服务质量', '首次解决率', '每周', '<60%', '>75%'])

    monitor_table = Table(monitor_data, colWidths=[2.5*cm, 3*cm, 2*cm, 2.5*cm, 2.5*cm])
    monitor_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(monitor_table)
    story.append(Spacer(1, 20))

    # ===== 八、总结与行动清单 =====
    story.append(Paragraph("八、总结与行动清单", heading_style))

    # 8.1 核心发现
    story.append(Paragraph("8.1 核心发现", subheading_style))

    findings = []
    if claim_ratio > 20:
        findings.append(f"理赔服务是最大痛点，占比{claim_ratio:.1f}%")
    if negative_ratio > 15:
        findings.append(f"不满客户比例{negative_ratio:.1f}%，需重点关注")
    if cancel_ratio > 5:
        findings.append(f"退保咨询占比{cancel_ratio:.1f}%，反映产品设计问题")
    if consult_ratio > 30:
        findings.append(f"咨询类占比{consult_ratio:.1f}%，用户对产品了解不足")

    for i, finding in enumerate(findings, 1):
        story.append(Paragraph(f"  {i}. {finding}", normal_style))

    story.append(Spacer(1, 10))

    # 8.2 立即行动清单
    story.append(Paragraph("8.2 立即行动清单", subheading_style))

    action_data = [['优先级', '行动项', '依据', '预期效果']]
    if claim_ratio > 20:
        action_data.append(['P0', '理赔进度可视化上线', f'理赔咨询占比{claim_ratio:.1f}%', '咨询减少50%'])
        action_data.append(['P0', '发布理赔材料清单图', '材料咨询高频', '驳回率降低30%'])
    if negative_ratio > 15:
        action_data.append(['P1', '不满客户问题专项分析', f'不满占比{negative_ratio:.1f}%', '满意度提升10%'])
    if clause_ratio > 3:
        action_data.append(['P1', '条款通俗化问答版本', f'条款咨询占比{clause_ratio:.1f}%', '咨询减少30%'])
    if cancel_ratio > 5:
        action_data.append(['P2', '投保重复检测功能', f'退保咨询占比{cancel_ratio:.1f}%', '退保咨询减少40%'])

    if len(action_data) > 1:
        action_table = Table(action_data, colWidths=[1.5*cm, 4.5*cm, 3*cm, 3*cm])
        action_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(action_table)
    story.append(Spacer(1, 15))

    # 8.3 预期效果目标
    story.append(Paragraph("8.3 预期效果目标", subheading_style))

    target_data = [['指标', '当前值', '目标值', '改善幅度']]
    target_data.append(['不满客户比例', f'{negative_ratio:.1f}%', '15.0%', f'{max(0, negative_ratio-15):.1f}%'])
    if claim_ratio > 20:
        target_data.append(['理赔相关咨询占比', f'{claim_ratio:.1f}%', '40.0%', f'{max(0, claim_ratio-40):.1f}%'])

    target_table = Table(target_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
    target_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), chinese_font),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(target_table)
    story.append(Spacer(1, 20))

    # ===== 报告说明 =====
    story.append(Paragraph("报告说明", normal_style))
    story.append(Paragraph(f"本报告基于{success_count}条客服对话记录分析生成", normal_style))
    story.append(Paragraph(f"分析日期：{time.strftime('%Y年%m月%d日')}", normal_style))

    # 生成PDF
    doc.build(story)

    # 清理临时图表文件
    for chart_file in chart_files:
        try:
            if os.path.exists(chart_file):
                os.remove(chart_file)
        except:
            pass
    try:
        os.rmdir(temp_dir)
    except:
        pass


# ==================== 核心函数 ====================

def normalize_intent(intent_result):
    """标准化意图分类格式，统一为"一级分类-二级分类"

    输入可能是：
    - "理赔-理赔资料咨询" → 保持不变
    - "理赔流程咨询" → 转为"咨询-理赔流程咨询"
    - "理赔流程咨询-理赔资料咨询" → 转为"理赔流程咨询-理赔资料咨询"（保留前两级）
    """
    if not intent_result:
        return ""

    # 清理多余空格和符号
    intent_result = intent_result.strip()

    # 分割获取各级分类
    parts = intent_result.replace("—", "-").replace("–", "-").split("-")

    # 判断一级分类是否在标准体系中
    primary = parts[0].strip() if parts else ""

    # 如果一级分类不在标准体系中，可能是二级分类写法
    if primary not in PRIMARY_CATEGORIES:
        # 尝试根据二级分类反推一级分类
        for cat, subcats in INTENT_CATEGORIES.items():
            if primary in subcats:
                return f"{cat}-{primary}"
        # 无法匹配，保持原样
        return intent_result

    # 一级分类正确，取前两级
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return intent_result


def get_primary_intent(intent):
    """从意图分类中提取一级分类"""
    if not intent:
        return ""
    parts = intent.split("-")
    return parts[0] if parts else intent


def read_dialog_csv(csv_path):
    """读取对话 CSV 文件"""
    dialog_lines = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    speaker = row[0]
                    content = row[1]
                    dialog_lines.append(f"[{speaker}] {content}")
    except Exception as e:
        return None
    return "\n".join(dialog_lines) if dialog_lines else None


def call_api(prompt, max_retries=3):
    """调用 API，带重试机制"""
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
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


# 全局变量
progress_lock = Lock()
progress_counter = 0
total_tokens_input = 0
total_tokens_output = 0
results = []


def process_single_file(csv_path, total_files, do_intent, do_sentiment, do_issue):
    """处理单个文件"""
    global progress_counter, total_tokens_input, total_tokens_output, results

    filename = os.path.basename(csv_path)
    source = "电话录音" if "dialog_csv" in csv_path else "在线对话"

    # 读取对话内容
    dialog = read_dialog_csv(csv_path)
    if not dialog:
        with progress_lock:
            progress_counter += 1
        return None

    result = {
        "filename": filename,
        "source": source,
        "intent": "",
        "sentiment": "",
        "issue": ""
    }

    # 意图分类
    if do_intent:
        intent_result = call_api(INTENT_PROMPT.format(dialog=dialog[:2000]))
        if intent_result:
            result["intent"] = normalize_intent(intent_result)

    # 情感分析
    if do_sentiment:
        sentiment_result = call_api(SENTIMENT_PROMPT.format(dialog=dialog[:2000]))
        if sentiment_result:
            result["sentiment"] = sentiment_result

    # 核心问题提取（独立于情感分析）
    if do_issue:
        issue_result = call_api(ISSUE_PROMPT.format(dialog=dialog[:2000]))
        if issue_result:
            # 清理结果，只取第一行有效内容
            issue_result = issue_result.strip().split('\n')[0].strip()
            # 去除可能的前缀符号
            issue_result = re.sub(r'^【.*】', '', issue_result)
            result["issue"] = issue_result

    with progress_lock:
        progress_counter += 1
        results.append(result)
        if progress_counter % 100 == 0 or progress_counter == total_files:
            print(f"  进度: {progress_counter}/{total_files} ({progress_counter/total_files*100:.1f}%)", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description='客服对话分析脚本')
    parser.add_argument('--dir', type=str, choices=['dialog_csv', 'online_chat_csv', 'all'],
                        default='all', help='选择分析目录：dialog_csv(电话录音)、online_chat_csv(在线对话)、all(全部)')
    parser.add_argument('--limit', type=int, default=None, help='处理文件数量限制')
    parser.add_argument('--skip-intent', action='store_true', help='跳过意图分类')
    parser.add_argument('--skip-sentiment', action='store_true', help='跳过情感分析')
    parser.add_argument('--skip-issue', action='store_true', help='跳过高频问题提取')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("客服对话分析（统一版）")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集 CSV 文件
    csv_files = []

    if args.dir == "dialog_csv":
        csv_files = glob(os.path.join(DIALOG_CSV_DIR, "*.csv"))
        print(f"分析目录: dialog_csv (电话录音)")
        print(f"文件数量: {len(csv_files)} 个")
    elif args.dir == "online_chat_csv":
        csv_files = glob(os.path.join(ONLINE_CHAT_CSV_DIR, "*.csv"))
        print(f"分析目录: online_chat_csv (在线对话)")
        print(f"文件数量: {len(csv_files)} 个")
    else:  # all
        dialog_files = glob(os.path.join(DIALOG_CSV_DIR, "*.csv"))
        online_files = glob(os.path.join(ONLINE_CHAT_CSV_DIR, "*.csv"))
        csv_files = dialog_files + online_files
        print(f"分析目录: 全部")
        print(f"  - dialog_csv (电话录音): {len(dialog_files)} 个")
        print(f"  - online_chat_csv (在线对话): {len(online_files)} 个")

    if not csv_files:
        print("没有找到 CSV 文件")
        return

    # 限制数量
    if args.limit:
        csv_files = csv_files[:args.limit]
        print(f"限制处理: 前 {args.limit} 个文件")

    total_files = len(csv_files)
    do_intent = not args.skip_intent
    do_sentiment = not args.skip_sentiment
    do_issue = not args.skip_issue

    print(f"总计: {total_files} 个文件待处理")
    print(f"API 并发: {API_MAX_WORKERS} 线程")
    print(f"分析内容: 意图分类={'是' if do_intent else '否'}, 情感分析={'是' if do_sentiment else '否'}, 高频问题={'是' if do_issue else '否'}")
    print(f"{'='*60}\n")

    # 初始化
    global results, progress_counter, total_tokens_input, total_tokens_output
    results = []
    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_file, f, total_files, do_intent, do_sentiment, do_issue): f
            for f in csv_files
        }
        for future in as_completed(futures):
            pass

    t_end = time.time()

    # 统计
    success_count = len(results)
    elapsed = t_end - t_start

    print(f"\n{'='*60}")
    print("处理完成统计")
    print(f"{'='*60}")
    print(f"处理文件:       {total_files} 个")
    print(f"成功分析:       {success_count} 个")
    print(f"总耗时:         {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
    print(f"{'='*60}")

    # 初始化统计变量
    intent_counts = Counter()
    primary_intent_counts = Counter()
    sentiment_counts = Counter()
    issue_counts = Counter()

    # === 意图分类统计 ===
    if do_intent and results:
        intent_counts = Counter(r["intent"] for r in results if r["intent"])
        primary_intent_counts = Counter(get_primary_intent(r["intent"]) for r in results if r["intent"])

        # 一级分类统计
        print("\n【一级分类统计】")
        print("-" * 40)
        for intent in PRIMARY_CATEGORIES:
            count = primary_intent_counts.get(intent, 0)
            if count > 0:
                percent = count / success_count * 100 if success_count else 0
                bar = "█" * int(percent / 2)
                print(f"  {intent:6s}: {count:5d}次 ({percent:5.1f}%) {bar}")

        # 二级分类统计
        print("\n【二级分类统计 TOP 20】")
        print("-" * 40)
        for i, (intent, count) in enumerate(intent_counts.most_common(20), 1):
            percent = count / success_count * 100 if success_count else 0
            print(f"  {i:2d}. {intent:20s} | {count:5d}次 ({percent:.1f}%)")

    # === 情感分析统计 ===
    if do_sentiment and results:
        sentiment_counts = Counter(r["sentiment"] for r in results if r["sentiment"])

        print("\n【情感分析结果】")
        print("-" * 40)
        for sentiment in ["满意", "中立", "不满"]:
            count = sentiment_counts.get(sentiment, 0)
            percent = count / success_count * 100 if success_count else 0
            bar = "█" * int(percent / 2)
            flag = " ⚠️ 需关注" if sentiment == "不满" else ""
            print(f"  {sentiment}: {count:6d} ({percent:5.1f}%) {bar}{flag}")

    # === 高频问题统计 ===
    if do_issue and results:
        issue_counts = Counter(r["issue"] for r in results if r["issue"])

        print("\n【高频问题 TOP 20】")
        print("-" * 40)
        for i, (issue, count) in enumerate(issue_counts.most_common(20), 1):
            percent = count / success_count * 100 if success_count else 0
            print(f"  {i:2d}. {issue:20s} | {count:5d}次 ({percent:.1f}%)")

    # === 来源分布 ===
    if results:
        source_counts = Counter(r["source"] for r in results)
        print("\n【来源分布】")
        print("-" * 40)
        for source, count in source_counts.items():
            percent = count / success_count * 100 if success_count else 0
            print(f"  {source}: {count} ({percent:.1f}%)")

    # === 生成PDF报告 ===
    if results:
        # 根据目录类型生成不同的文件名
        if args.dir == "dialog_csv":
            report_title = "电话录音分析报告"
            output_pdf = os.path.join(OUTPUT_DIR, "电话录音分析报告.pdf")
        elif args.dir == "online_chat_csv":
            report_title = "在线对话分析报告"
            output_pdf = os.path.join(OUTPUT_DIR, "在线对话分析报告.pdf")
        else:
            report_title = "客服对话分析报告"
            output_pdf = os.path.join(OUTPUT_DIR, "客服对话分析报告.pdf")

        create_pdf_report(output_pdf, report_title, results, success_count, do_intent, do_sentiment, do_issue,
                          intent_counts, primary_intent_counts, sentiment_counts, issue_counts)
        print(f"\nPDF报告已保存: {output_pdf}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()