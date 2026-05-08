"""
咨询类问题与回答分析脚本

分析咨询类对话（不涉及用户信息查询），归集高频问题和客服标准回答，
为AI客服系统提供知识库基础。

功能：
1. 按二级分类分析高频问题
2. 归集客服标准回答模板
3. 提取问答知识库
4. 生成HTML报告

使用方式：
  python scripts/analyze_consultation.py --type all          # 全部数据
  python scripts/analyze_consultation.py --type online       # 仅在线对话
  python scripts/analyze_consultation.py --type phone        # 仅电话录音
  python scripts/analyze_consultation.py --category 保障范围了解  # 指定分类
  python scripts/analyze_consultation.py --limit 500         # 限制数量测试
"""

import os
import csv
import re
import requests
import argparse
import pymysql
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter, defaultdict

# 数据库配置
DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10

INPUT_PRICE = 0.0005
OUTPUT_PRICE = 0.002

# 咨询类二级分类（通用知识类，不涉及用户信息查询）
CONSULTATION_CATEGORIES = [
    '保障范围了解',    # 问产品保障什么
    '理赔流程了解',    # 问理赔步骤
    '产品了解',        # 问产品详情
    '理赔材料了解',    # 问需要什么材料
    '条款解释',        # 问条款含义
    '退保流程了解',    # 问退保流程
    '费率查询',        # 问保费价格
]

# ==================== 数据库操作 ====================

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def query_consultations(source_type=None, category=None, limit=None):
    """从数据库查询咨询类数据"""
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
            SELECT id, source_type, source_file, secondary_intent, sentiment, dialog_date
            FROM dialog_analysis
            WHERE primary_intent = '咨询'
            """
            if source_type:
                sql += f" AND source_type = '{source_type}'"
            if category:
                sql += f" AND secondary_intent = '{category}'"
            else:
                # 默认只查询通用知识类（排除"其他了解"）
                categories_str = "','".join(CONSULTATION_CATEGORIES)
                sql += f" AND secondary_intent IN ('{categories_str}')"
            sql += " ORDER BY dialog_date DESC"
            if limit:
                sql += f" LIMIT {limit}"

            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        conn.close()


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
                    "max_tokens": 512,
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
                import time
                time.sleep(1)
            else:
                return None
    return None


# ==================== Prompt 定义 ====================

QUESTION_ANALYSIS_PROMPT = """分析以下客服对话，归纳客户咨询的问题类型。

对话内容：
{dialog}

请判断客户咨询的问题属于哪一类（从以下分类中选择最符合的）：

【保障范围类】
- 保障内容：问产品保障什么疾病/意外
- 保障期限：问保障多久、过期时间
- 既往症保障：问既往症是否保障
- 除外责任：问什么情况不赔

【理赔流程类】
- 理赔步骤：问理赔怎么申请、流程是什么
- 理赔时效：问理赔需要多久
- 理赔条件：问什么情况可以理赔
- 理赔渠道：问在哪里申请理赔

【理赔材料类】
- 所需材料：问理赔需要什么材料
- 材料要求：问材料格式、明细要求
- 材料提交：问怎么提交材料

【产品信息类】
- 产品介绍：问产品详情、特点
- 投保条件：问谁能买、年龄限制
- 续保规则：问续保条件、续保流程
- 保单查询：问保单信息

【退保流程类】
- 退保条件：问能不能退保
- 退保流程：问怎么退保
- 退保金额：问退保退多少钱
- 犹豫期：问犹豫期多久

【费率价格类】
- 保费计算：问保费怎么算
- 保费查询：问保费多少
- 缴费方式：问怎么缴费

输出格式（JSON）：
{{"category": "大类", "question_type": "具体类型"}}
只输出JSON，不要解释。"""


ANSWER_ANALYSIS_PROMPT = """分析以下客服对话中客服的回答内容。

对话内容：
{dialog}

请提取客服给客户的有效回答（排除问候语、自动回复、营销内容）：

1. 客服回答的核心内容是什么？（提取关键信息，不超过100字）
2. 回答是否准确完整？（准确/部分准确/不准确）
3. 回答是否可以直接作为标准回答模板？（是/否）

输出格式（JSON）：
{{"answer": "核心回答内容", "accuracy": "准确/部分准确/不准确", "is_template": "是/否"}}
只输出JSON。"""


QA_EXTRACTION_PROMPT = """从以下对话中提取问答知识，用于AI客服知识库。

对话内容：
{dialog}

请提取：
1. 客户问题（标准化表述）
2. 客服回答（标准化表述，可直接用于回复客户）

输出格式（JSON）：
{{"std_question": "标准化问题", "std_answer": "标准化回答"}}
只输出JSON。"""


# ==================== 文件处理 ====================

def read_dialog_csv(csv_path):
    """读取 CSV 文件"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                return None, 0, [], []

            # 跳过标题行
            if rows[0][0] in ['角色', '发送者', 'sender', '对话者', '﻿对话者']:
                rows = rows[1:]

            dialog_lines = []
            customer_questions = []
            agent_answers = []

            for row in rows:
                if len(row) >= 2:
                    role = row[0]
                    content = row[1]
                    dialog_lines.append(f"{role}: {content}")

                    if role in ['客户', 'customer', '用户']:
                        customer_questions.append(content)
                    elif role in ['客服', 'agent', '坐席', '工作人员']:
                        agent_answers.append(content)

            return '\n'.join(dialog_lines), len(rows), customer_questions, agent_answers
    except Exception as e:
        return None, 0, [], []


def clean_text(text):
    """清理文本"""
    if not text:
        return ''
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = re.sub(r'[^\u0020-\u007E\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]', '', text)
    return text.strip()


def parse_json_result(text):
    """解析JSON结果"""
    if not text:
        return None
    try:
        import json
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return None


def filter_valid_answers(answers):
    """过滤有效的客服回答（排除自动回复、营销内容、等待回复）"""

    # 黑名单（自动回复、营销内容）
    blacklist = [
        '保障升级', '参保开启', '我时刻都准备', '本次服务将在',
        '在线客服人员', '感谢您的支持', '长时间未对话',
        '扫码', '点击链接', '查看详情', '公众号', 'APP',
        '智能机器人', '很高兴为您服务', '我是智能机器人',
    ]

    # 等待类回复（不算实质回答）
    waiting_phrases = [
        '稍等', '稍等一下', '稍等片刻', '好的稍等', '稍等哈',
        '耐心等待', '辛苦等待', '请稍等', '正在处理',
        '正在努力处理', '正在查询', '正在核实', '帮你催',
        '辛苦耐心等待', '帮您催促',
    ]

    # 业务关键词（有实质回答应该包含这些）
    business_keywords = [
        '理赔', '保单', '保险', '报销', '材料', '审核',
        '打款', '到账', '投保', '退保', '续保', '变更',
        '保障', '条款', '费用', '金额', '时效', '流程',
        '申请', '查询', '进度', '状态', '条件', '比例',
        '免赔额', '医保', '门诊', '住院', '门特', '赔付',
        '发票', '清单', '出院', '结算', '直赔', '快赔',
    ]

    valid = []
    for ans in answers:
        ans_clean = ans.strip()

        # 排除黑名单内容
        if any(kw in ans for kw in blacklist):
            continue

        # 排除等待类回复（短回复且包含等待词）
        if any(kw in ans for kw in waiting_phrases) and len(ans) < 80:
            continue

        # 排除过短内容（少于30字很难有实质内容）
        if len(ans) < 30:
            continue

        # 排除纯问候/请求信息类
        greetings = ['您好', '你好', '好的', '是的', '请问有什么可以帮您',
                     '辛苦提供一下身份证号', '好的', '嗯', '好的好的',
                     '好的稍等', '辛苦耐心等待']
        if ans_clean in greetings or len(ans_clean) < 10:
            continue

        # 必须包含业务关键词才算实质回答
        has_keyword = any(kw in ans for kw in business_keywords)
        if not has_keyword:
            continue

        valid.append(ans)

    return valid


def has_real_agent_answer(csv_path):
    """检查CSV文件是否有真正的客服回答（用于预筛选）"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return False

            if rows[0][0] in ['角色', '发送者', 'sender', '对话者', '﻿对话者']:
                rows = rows[1:]

            agent_answers = []
            for row in rows:
                if len(row) >= 2 and row[0] in ['客服', 'agent', '坐席', '工作人员']:
                    agent_answers.append(row[1])

            # 过滤有效回答
            valid = filter_valid_answers(agent_answers)
            return len(valid) > 0
    except:
        return False


# ==================== 主分析流程 ====================

def analyze_consultations(source_type=None, category=None, limit=None):
    """分析咨询类对话（只分析有实质客服回答的对话）"""
    global progress_counter, total_tokens_input, total_tokens_output

    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    # 查询数据
    records = query_consultations(source_type, category, limit)

    # 数据来源名称
    if source_type == 'phone':
        source_name = '电话录音'
    elif source_type == 'online':
        source_name = '在线对话'
    else:
        source_name = '全量数据'

    print(f"\n{'='*60}")
    print(f"咨询类问题与回答分析")
    print(f"数据来源: {source_name}")
    if category:
        print(f"分析分类: {category}")
    print(f"查询记录数: {len(records)}")
    print(f"{'='*60}\n")

    if not records:
        print("没有数据")
        return None

    # 预筛选：只保留有实质客服回答的对话
    print("预筛选：检查是否有实质客服回答...")
    valid_records = []
    skipped_count = 0
    for r in records:
        if has_real_agent_answer(r['source_file']):
            valid_records.append(r)
        else:
            skipped_count += 1

    print(f"有实质回答: {len(valid_records)} 条")
    print(f"无实质回答（跳过）: {skipped_count} 条")
    print()

    if not valid_records:
        print("没有有实质回答的对话")
        return None

    # 更新records为预筛选后的
    records = valid_records

    # 分析结果
    results = []
    qa_pairs = []  # 问答知识库

    # 按二级分类统计
    category_results = defaultdict(list)

    def process_record(record):
        global progress_counter
        csv_path = record['source_file']

        if not os.path.exists(csv_path):
            return None

        dialog, line_count, questions, answers = read_dialog_csv(csv_path)
        if not dialog:
            return None

        # 过滤有效回答（即使没有也继续分析问题）
        valid_answers = filter_valid_answers(answers)

        # 调用 API 分析问题（无论有没有回答都分析）
        question_result = call_api(QUESTION_ANALYSIS_PROMPT.format(dialog=dialog[:800]))

        question_info = parse_json_result(question_result) or {}

        # 只有有有效回答才分析回答内容
        if valid_answers:
            answer_result = call_api(ANSWER_ANALYSIS_PROMPT.format(dialog=dialog[:800]))
            answer_info = parse_json_result(answer_result) or {}
            qa_result = call_api(QA_EXTRACTION_PROMPT.format(dialog=dialog[:800]))
            qa_info = parse_json_result(qa_result) or {}
        else:
            answer_info = {}
            qa_info = {}

        with progress_lock:
            progress_counter += 1
            if progress_counter % 50 == 0 or progress_counter == len(records):
                print(f"进度: {progress_counter}/{len(records)}")

        result = {
            'file': csv_path,
            'category': record['secondary_intent'],
            'question_category': question_info.get('category', '未知'),
            'question_type': question_info.get('question_type', '未知'),
            'answer': clean_text(answer_info.get('answer', valid_answers[0] if valid_answers else '')),
            'accuracy': answer_info.get('accuracy', '无有效回答') if valid_answers else '无有效回答',
            'is_template': answer_info.get('is_template', '否') if valid_answers else '否',
            'std_question': qa_info.get('std_question', ''),
            'std_answer': qa_info.get('std_answer', ''),
            'valid_answers': valid_answers,
            'has_valid_answer': len(valid_answers) > 0,  # 标记是否有有效回答
        }

        return result

    # 多线程处理
    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, r) for r in records]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                # 按分类归集
                category_results[result['category']].append(result)
                # 收集问答知识
                if result['std_question'] and result['std_answer']:
                    qa_pairs.append({
                        'category': result['category'],
                        'question': result['std_question'],
                        'answer': result['std_answer'],
                    })

    print(f"\n处理完成！成功: {len(results)}, 失败: {len(records) - len(results)}")

    # 计算成本
    total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE
    print(f"API 成本: ¥{total_cost:.2f}")

    # 统计分析
    analyze_and_report(results, category_results, qa_pairs)

    return results, category_results, qa_pairs


def analyze_and_report(results, category_results, qa_pairs):
    """统计分析并生成报告"""

    # 1. 高频问题类型统计（按大类和具体类型）
    category_counter = Counter(r['question_category'] for r in results if r['question_category'])
    type_counter = Counter(r['question_type'] for r in results if r['question_type'])

    # 有效回答统计
    has_answer_count = sum(1 for r in results if r.get('has_valid_answer'))
    no_answer_count = len(results) - has_answer_count

    print(f"\n{'='*60}")
    print("分析结果统计")
    print(f"{'='*60}")

    print(f"\n【有效回答统计】")
    print(f"  有有效回答: {has_answer_count} ({has_answer_count/len(results)*100:.1f}%)")
    print(f"  无有效回答: {no_answer_count} ({no_answer_count/len(results)*100:.1f}%)")

    print(f"\n【高频问题大类 TOP10】")
    for i, (cat, count) in enumerate(category_counter.most_common(10), 1):
        print(f"  {i}. {cat}: {count}次 ({count/len(results)*100:.1f}%)")

    print(f"\n【高频具体问题类型 TOP20】")
    for i, (t, count) in enumerate(type_counter.most_common(20), 1):
        print(f"  {i}. {t}: {count}次 ({count/len(results)*100:.1f}%)")

    # 2. 按二级分类统计
    print(f"\n【各二级分类问题数量】")
    for cat, items in sorted(category_results.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(items)} 条")

    # 3. 回答质量统计（只统计有有效回答的）
    results_with_answer = [r for r in results if r.get('has_valid_answer')]
    if results_with_answer:
        accuracy_counter = Counter(r['accuracy'] for r in results_with_answer if r['accuracy'])
        print(f"\n【回答准确率（仅统计有回答的）】")
        for acc, count in accuracy_counter.most_common():
            print(f"  {acc}: {count} ({count/len(results_with_answer)*100:.1f}%)")

    # 4. 可作为模板的回答
    template_count = sum(1 for r in results if r['is_template'] == '是')
    print(f"\n【可作为标准回答模板】: {template_count} 条")

    # 5. 问答知识库统计
    print(f"\n【提取问答知识库】: {len(qa_pairs)} 条")


def generate_html_report(results, category_results, qa_pairs, output_dir, source_name):
    """生成HTML报告"""

    # 统计数据 - 按问题类型统计
    category_counter = Counter(r['question_category'] for r in results if r['question_category'])
    type_counter = Counter(r['question_type'] for r in results if r['question_type'])

    # 按问题类型归集回答
    type_answers = defaultdict(list)
    for r in results:
        if r['question_type'] and r['answer']:
            type_answers[r['question_type']].append(r['answer'])

    # 找出每个问题类型的代表性回答（频率最高的）
    representative_answers = {}
    for qtype, answers in type_answers.items():
        answer_counter = Counter(answers)
        representative_answers[qtype] = answer_counter.most_common(3)  # 取前3个

    # 构建表格行
    def build_category_rows():
        rows = []
        for i, (cat, count) in enumerate(category_counter.most_common(10), 1):
            rows.append(f"<tr><td>{i}</td><td><span class='tag tag-blue'>{cat}</span></td><td>{count}</td><td>{count/len(results)*100:.1f}%</td></tr>")
        return "".join(rows)

    def build_type_rows():
        rows = []
        for i, (t, count) in enumerate(type_counter.most_common(30), 1):
            answers = representative_answers.get(t, [])
            if answers:
                ans_text = answers[0][0]
                if len(ans_text) > 60:
                    ans_text = ans_text[:60] + '...'
            else:
                ans_text = '暂无'
            rows.append(f"<tr><td>{i}</td><td>{t}</td><td>{count}</td><td>{count/len(results)*100:.1f}%</td><td>{ans_text}</td></tr>")
        return "".join(rows)

    def build_secondary_rows():
        rows = []
        for cat, items in sorted(category_results.items(), key=lambda x: len(x[1]), reverse=True):
            top_types = Counter(r['question_type'] for r in items).most_common(3)
            top_str = ", ".join([t for t, c in top_types]) if top_types else '-'
            rows.append(f"<tr><td>{cat}</td><td>{len(items)}</td><td>{top_str}</td></tr>")
        return "".join(rows)

    def build_qa_items():
        items = []
        for qa in qa_pairs[:50]:
            items.append(f'''
            <div class="qa-item">
                <div class="qa-question">问: {qa['question']}</div>
                <div class="qa-answer">答: {qa['answer']}</div>
                <span class="tag tag-green">{qa['category']}</span>
            </div>
            ''')
        result = "".join(items)
        if len(qa_pairs) > 50:
            result += f'<p>... 共 {len(qa_pairs)} 条问答知识，仅展示前50条</p>'
        return result

    def build_detailed_answers():
        sections = []
        for qtype, count in type_counter.most_common(15):
            answers = representative_answers.get(qtype, [])
            if not answers:
                continue
            answer_boxes = ""
            for i, (ans, c) in enumerate(answers[:3], 1):
                ans_display = ans[:150] + '...' if len(ans) > 150 else ans
                answer_boxes += f'<div class="answer-box">回答{i}: {ans_display}</div>'
            sections.append(f'''
            <div class="question-box">
                <strong>问题类型: {qtype}</strong> <span class="tag tag-orange">{count}次</span>
            </div>
            {answer_boxes}
            ''')
        return "".join(sections)

    # 计算统计数据
    template_count = sum(1 for r in results if r['is_template']=='是')
    top15_coverage = sum(count for _, count in type_counter.most_common(15))/len(results)*100
    has_answer_count = sum(1 for r in results if r.get('has_valid_answer'))
    no_answer_count = len(results) - has_answer_count

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>咨询类问题与回答分析报告</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; background: #f5f6fa; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; }}
        .header {{ background: linear-gradient(135deg, #3498DB, #2980B9); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #3498DB; border-bottom: 2px solid #3498DB; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; border-bottom: 1px solid #ddd; text-align: left; }}
        th {{ background: #3498DB; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .question-box {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3498DB; }}
        .answer-box {{ background: #f1f8e9; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #27AE60; }}
        .qa-item {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .qa-question {{ font-weight: bold; color: #2C3E50; margin-bottom: 10px; }}
        .qa-answer {{ color: #27AE60; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .tag {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px; }}
        .tag-blue {{ background: #3498DB; color: white; }}
        .tag-green {{ background: #27AE60; color: white; }}
        .tag-orange {{ background: #F39C12; color: white; }}
        .tag-red {{ background: #E74C3C; color: white; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-card .number {{ font-size: 28px; font-weight: bold; color: #3498DB; }}
        .stat-card .label {{ color: #666; margin-top: 5px; }}
        .stat-card.warning .number {{ color: #E74C3C; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>咨询类问题与回答分析报告</h1>
            <p>数据来源: {source_name} | 分析数量: {len(results)} 条对话 | 问答知识: {len(qa_pairs)} 条</p>
        </div>

        <div class="section">
            <h2>一、分析概览</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{len(results)}</div>
                    <div class="label">分析对话数</div>
                </div>
                <div class="stat-card">
                    <div class="number">{len(type_counter)}</div>
                    <div class="label">问题类型数</div>
                </div>
                <div class="stat-card">
                    <div class="number">{has_answer_count}</div>
                    <div class="label">有有效回答</div>
                </div>
                <div class="stat-card warning">
                    <div class="number">{no_answer_count}</div>
                    <div class="label">无有效回答</div>
                </div>
                <div class="stat-card">
                    <div class="number">{template_count}</div>
                    <div class="label">标准回答模板</div>
                </div>
            </div>
            <p style="color:#666;">说明："无有效回答"指客服回复仅为问候语/自动回复，无实质内容。这些对话仍可分析问题类型。</p>
        </div>

        <div class="section">
            <h2>二、高频问题大类 TOP10</h2>
            <table>
                <tr><th>排名</th><th>问题大类</th><th>出现次数</th><th>占比</th></tr>
                {build_category_rows()}
            </table>
        </div>

        <div class="section">
            <h2>三、高频具体问题类型 TOP30</h2>
            <table>
                <tr><th>排名</th><th>问题类型</th><th>出现次数</th><th>占比</th><th>代表性回答</th></tr>
                {build_type_rows()}
            </table>
        </div>

        <div class="section">
            <h2>四、各二级分类问题统计</h2>
            <table>
                <tr><th>二级分类</th><th>对话数</th><th>高频问题类型</th></tr>
                {build_secondary_rows()}
            </table>
        </div>

        <div class="section">
            <h2>五、问答知识库（AI客服可直接使用）</h2>
            <p>以下问答已标准化处理，可直接用于AI客服系统回复客户：</p>
            {build_qa_items()}
        </div>

        <div class="section">
            <h2>六、高频问题类型详细回答</h2>
            <p>针对高频问题类型，展示多个客服回答供参考：</p>
            {build_detailed_answers()}
        </div>

        <div class="section">
            <h2>七、应用建议</h2>
            <div class="question-box">
                <strong>AI客服系统建设建议：</strong>
                <ul>
                    <li><b>知识库构建</b>：将"{len(qa_pairs)}"条问答知识导入AI客服系统</li>
                    <li><b>高频问题优先</b>：优先覆盖TOP15高频问题类型，覆盖率达{top15_coverage:.1f}%</li>
                    <li><b>分类管理</b>：按问题大类组织知识库，便于检索和维护</li>
                    <li><b>回答模板</b>："{template_count}"条标准回答可直接作为模板使用</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''

    # 保存报告
    report_path = os.path.join(output_dir, '咨询类问题与回答分析.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML报告已生成: {report_path}")

    # 同时导出问答知识库为JSON
    import json
    qa_json_path = os.path.join(output_dir, '问答知识库.json')
    with open(qa_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"问答知识库JSON: {qa_json_path}")

    return report_path


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="咨询类问题与回答分析")
    parser.add_argument("--type", choices=['online', 'phone', 'all'],
                       default='all', help="数据来源: online/phone/all")
    parser.add_argument("--category", type=str, default=None,
                       help="指定二级分类（如：保障范围了解）")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    args = parser.parse_args()

    # 确定数据来源
    if args.type == 'phone':
        source_type = 'phone'
    elif args.type == 'online':
        source_type = 'online'
    else:
        source_type = None

    # 分析
    results, category_results, qa_pairs = analyze_consultations(
        source_type=source_type,
        category=args.category,
        limit=args.limit
    )

    if not results:
        return

    # 数据来源名称
    if source_type == 'phone':
        source_name = '电话录音'
    elif source_type == 'online':
        source_name = '在线对话'
    else:
        source_name = '全量数据'

    # 生成报告
    output_dir = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports'
    os.makedirs(output_dir, exist_ok=True)
    generate_html_report(results, category_results, qa_pairs, output_dir, source_name)


if __name__ == "__main__":
    main()