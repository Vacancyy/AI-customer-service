"""
理赔时效投诉深度分析脚本

从数据库筛选理赔时效投诉案例，分析具体时效问题：
- 等待天数
- 涉及阶段（立案、审核、打款）
- 客户期望时效
- 实际处理时效
"""

import os
import re
import pymysql
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import requests
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"

TIME_ANALYSIS_PROMPT = '''分析以下理赔时效投诉对话，深度挖掘客户不满的根本原因。

对话内容：
{dialog}

请分析并输出以下信息（JSON格式）：

1. slow_stage: 具体慢在哪个环节（立案阶段/审核阶段/打款阶段/整体流程）
2. slow_reason: 为什么慢（材料不全反复驳回/系统问题导致延迟/人工审核积压/流程繁琐需要多次提交/高峰期案件多/信息沟通不畅客户不知道进度）
3. customer_core_issue: 客户核心不满点（具体描述，如"提交理赔20天还没立案"、"材料补交3次还在审核"）
4. customer_expectation: 客户期望什么（具体期望，如"希望3天内立案"、"希望告知具体审核进度"、"希望加急处理"）
5. agent_response_problem: 客服回复存在的问题（具体问题，如"只说标准时效没查具体进度"、"没解释为什么慢"、"没提供解决方案"）
6. resolution_status: 是否解决（已解决/部分解决/未解决）
7. improvement_suggestion: 具体可执行的改进措施（必须具体，如：
   - 立案环节：将立案时间缩短到3个工作日内，超过3天自动提醒客户
   - 审核环节：材料不全时一次性告知所有缺失材料，避免反复驳回
   - 打款环节：打款失败时立即通知客户并提供解决方案
   - 信息沟通：增加理赔进度实时查询功能，客户可随时查看当前环节和预计完成时间
   - 客服培训：培训客服在客户投诉时效时先查询具体进度再回复，而不是只说标准时效
   )

只输出JSON格式。'''


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def query_time_complaints(source_type=None, limit=None):
    """从数据库查询理赔时效投诉案例

    Args:
        source_type: 数据来源 (online/phone/None表示全部)
        limit: 限制数量
    """
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
            SELECT id, source_file, source_type, secondary_intent, sentiment, dialog_date
            FROM dialog_analysis
            WHERE primary_intent = '投诉' AND secondary_intent = '理赔时效投诉'
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


def read_csv_content(csv_path):
    """读取CSV文件内容"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if rows[0][0] in ['角色', '发送者', 'sender', '对话者', '﻿对话者']:
                rows = rows[1:]

            dialog_lines = []
            for row in rows:
                if len(row) >= 2:
                    role = row[0]
                    content = row[1]
                    dialog_lines.append(f"{role}: {content}")

            return '\n'.join(dialog_lines)
    except Exception as e:
        print(f"读取失败: {csv_path}, 错误: {e}")
        return None


def call_api(prompt):
    """调用API分析"""
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
            timeout=30,
        )
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        # 提取JSON
        import json
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"API调用失败: {e}")
        return None


def extract_time_info_simple(dialog):
    """简单提取时间信息（不调用API）"""
    info = {
        'wait_days': None,
        'wait_type': None,
        'stage': None,
        'customer_expectation': None,
        'agent_response_time': None,
        'urgency_level': None,
        'complaint_point': None
    }

    # 提取天数关键词
    day_patterns = [
        (r'(\d+)天', '天'),
        (r'(\d+)个工作日', '工作日'),
        (r'一个月', 30),
        (r'半个月', 15),
        (r'很久', None),
    ]

    for pattern, unit in day_patterns:
        match = re.search(pattern, dialog)
        if match:
            if unit == '天':
                info['wait_days'] = int(match.group(1))
            elif unit == '工作日':
                info['wait_days'] = int(match.group(1)) * 1.4  # 工作日转天数
            elif isinstance(unit, int):
                info['wait_days'] = unit
            break

    # 提取阶段关键词
    if '立案' in dialog and ('慢' in dialog or '久' in dialog or '还没' in dialog):
        info['stage'] = '立案阶段'
        info['wait_type'] = '立案慢'
    elif '审核' in dialog and ('慢' in dialog or '久' in dialog or '还没' in dialog):
        info['stage'] = '审核阶段'
        info['wait_type'] = '审核慢'
    elif '打款' in dialog or '到账' in dialog:
        info['stage'] = '打款阶段'
        info['wait_type'] = '打款慢'
    else:
        info['stage'] = '整体流程'
        info['wait_type'] = '整体慢'

    # 提取客服承诺时效
    agent_time_match = re.search(r'(3-7个工作日|30个工作日|\d+个工作日|最晚.*工作日)', dialog)
    if agent_time_match:
        info['agent_response_time'] = agent_time_match.group(1)

    # 紧急程度
    if '投诉' in dialog or '一个月' in dialog or (info.get('wait_days') and info.get('wait_days') >= 20):
        info['urgency_level'] = '高'
    elif '慢' in dialog or '久' in dialog:
        info['urgency_level'] = '中'
    else:
        info['urgency_level'] = '低'

    # 提取不满点
    complaint_lines = []
    for line in dialog.split('\n'):
        if '客户' in line and ('慢' in line or '久' in line or '还没' in line or '等' in line):
            complaint_lines.append(line.replace('客户:', '').strip()[:50])
    if complaint_lines:
        info['complaint_point'] = complaint_lines[0]

    return info


def analyze_time_complaints(use_api=True, source_type=None, limit=None):
    """分析理赔时效投诉

    Args:
        use_api: 是否使用API分析
        source_type: 数据来源 (online/phone/None表示全部)
        limit: 限制数量
    """
    print("="*60)
    print("理赔时效投诉深度分析")
    print("="*60)

    # 从数据库查询
    records = query_time_complaints(source_type, limit)

    # 显示数据来源
    if source_type == 'phone':
        source_name = '电话录音'
    elif source_type == 'online':
        source_name = '在线对话'
    else:
        source_name = '全部数据'

    print(f"\n数据来源: {source_name}")
    print(f"从数据库查询到 {len(records)} 条理赔时效投诉记录")

    if not records:
        print("没有数据")
        return

    # 分析每个案例
    results = []
    progress_lock = Lock()
    progress_counter = 0

    def process_case(record):
        nonlocal progress_counter
        csv_path = record['source_file']
        if not os.path.exists(csv_path):
            return None

        dialog = read_csv_content(csv_path)
        if not dialog:
            return None

        # 提取时效信息
        if use_api:
            time_info = call_api(TIME_ANALYSIS_PROMPT.format(dialog=dialog[:600]))
        else:
            time_info = extract_time_info_simple(dialog)

        if time_info:
            with progress_lock:
                progress_counter += 1
                if progress_counter % 20 == 0:
                    print(f"进度: {progress_counter}/{len(records)}")
            return {
                'file': csv_path,
                'dialog_date': record['dialog_date'],
                'dialog': dialog[:300],
                **time_info
            }
        return None

    # 使用多线程加速
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_case, r) for r in records]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    print(f"\n成功分析 {len(results)} 个案例")

    # 统计分析
    analyze_results(results)

    return results


def analyze_results(results):
    """统计分析结果"""
    if not results:
        return

    print("\n" + "="*60)
    print("统计分析结果")
    print("="*60)

    # 慢的环节分布
    slow_stages = Counter(r['slow_stage'] for r in results if r.get('slow_stage'))
    print(f"\n【慢的环节分布】")
    for stage, count in slow_stages.most_common():
        print(f"  {stage}: {count} 个 ({count/len(results)*100:.1f}%)")

    # 慢的原因分布
    slow_reasons = Counter(r['slow_reason'] for r in results if r.get('slow_reason'))
    print(f"\n【慢的根本原因分布】")
    for reason, count in slow_reasons.most_common():
        print(f"  {reason}: {count} 个 ({count/len(results)*100:.1f}%)")

    # 客服回复问题
    response_problems = [r['agent_response_problem'] for r in results if r.get('agent_response_problem')]
    print(f"\n【客服回复存在的问题】")
    for i, problem in enumerate(response_problems[:10], 1):
        if problem:
            print(f"  {i}. {problem}")

    # 解决状态
    resolution_status = Counter(r['resolution_status'] for r in results if r.get('resolution_status'))
    print(f"\n【解决状态分布】")
    for status, count in resolution_status.most_common():
        print(f"  {status}: {count} 个 ({count/len(results)*100:.1f}%)")

    # 客户期望汇总
    expectations = [r['customer_expectation'] for r in results if r.get('customer_expectation')]
    print(f"\n【客户具体期望】")
    for i, exp in enumerate(expectations[:10], 1):
        if exp:
            print(f"  {i}. {exp}")

    # 典型不满点
    print(f"\n【典型客户不满点 TOP10】")
    core_issues = [r['customer_core_issue'] for r in results if r.get('customer_core_issue')]
    for i, issue in enumerate(core_issues[:10], 1):
        if issue:
            print(f"  {i}. {issue}")

    # 改进建议分类汇总
    print(f"\n【具体改进建议】")
    suggestions = [r['improvement_suggestion'] for r in results if r.get('improvement_suggestion')]

    # 按环节分类建议
    stage_suggestions = {}
    for r in results:
        stage = r.get('slow_stage', '其他')
        suggestion = r.get('improvement_suggestion')
        if suggestion:
            if stage not in stage_suggestions:
                stage_suggestions[stage] = []
            stage_suggestions[stage].append(suggestion)

    for stage, sug_list in stage_suggestions.items():
        print(f"\n  === {stage}改进措施 ===")
        for sug in sug_list[:3]:
            print(f"  - {sug}")


def generate_html_report(results, output_dir):
    """生成HTML分析报告"""
    if not results:
        print("没有数据生成报告")
        return

    # 统计数据
    slow_stages = Counter(r['slow_stage'] for r in results if r.get('slow_stage'))
    slow_reasons = Counter(r['slow_reason'] for r in results if r.get('slow_reason'))
    response_quality = Counter(r['agent_response_quality'] for r in results if r.get('agent_response_quality'))
    resolution_status = Counter(r['resolution_status'] for r in results if r.get('resolution_status'))

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>理赔时效投诉深度分析</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; background: #f5f6fa; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .header {{ background: linear-gradient(135deg, #E74C3C, #C0392B); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .section h2 {{ color: #E74C3C; border-bottom: 2px solid #E74C3C; padding-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-card .number {{ font-size: 24px; font-weight: bold; color: #E74C3C; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        th {{ background: #E74C3C; color: white; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
        .badge-high {{ background: #E74C3C; color: white; }}
        .badge-medium {{ background: #F39C12; color: white; }}
        .badge-low {{ background: #27AE60; color: white; }}
        .suggestion-box {{ background: #d4edda; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .problem-box {{ background: #f8d7da; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>理赔时效投诉深度分析报告</h1>
            <p>分析 {len(results)} 个理赔时效投诉案例 | 深度挖掘根本原因</p>
        </div>

        <div class="section">
            <h2>一、核心发现</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{len(results)}</div>
                    <div>投诉案例总数</div>
                </div>
                <div class="stat-card">
                    <div class="number">{slow_stages.most_common(1)[0][0] if slow_stages else '-'}</div>
                    <div>主要慢的环节</div>
                </div>
                <div class="stat-card">
                    <div class="number">{slow_reasons.most_common(1)[0][0] if slow_reasons else '-'}</div>
                    <div>主要慢的原因</div>
                </div>
                <div class="stat-card">
                    <div class="number">{resolution_status.get('未解决', 0)}</div>
                    <div>未解决案例数</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>二、具体慢在哪个环节？</h2>
            <table>
                <tr><th>环节</th><th>数量</th><th>占比</th></tr>
                {"".join(f"<tr><td>{s}</td><td>{c}</td><td>{c/len(results)*100:.1f}%</td></tr>" for s, c in slow_stages.most_common())}
            </table>
        </div>

        <div class="section">
            <h2>三、慢的根本原因分析</h2>
            <table>
                <tr><th>根本原因</th><th>数量</th><th>占比</th></tr>
                {"".join(f"<tr><td>{r}</td><td>{c}</td><td>{c/len(results)*100:.1f}%</td></tr>" for r, c in slow_reasons.most_common())}
            </table>
        </div>

        <div class="section">
            <h2>四、客服回复存在的问题</h2>
            <table>
                <tr><th>序号</th><th>具体问题</th><th>涉及案例</th></tr>
                {"".join(f"<tr><td>{i+1}</td><td>{r.get('agent_response_problem', '-') or '-'}</td><td>{r.get('slow_stage', '-') or '-'}</td></tr>" for i, r in enumerate(results[:15]) if r.get('agent_response_problem'))}
            </table>
        </div>

        <div class="section">
            <h2>五、典型客户不满点</h2>
            <table>
                <tr><th>序号</th><th>客户核心不满</th><th>慢的环节</th><th>慢的原因</th><th>期望</th></tr>
                {"".join(f"<tr><td>{i+1}</td><td>{r.get('customer_core_issue', '-') or '-'}</td><td>{r.get('slow_stage', '-') or '-'}</td><td>{r.get('slow_reason', '-') or '-'}</td><td>{r.get('customer_expectation', '-') or '-'}</td></tr>" for i, r in enumerate(results[:15]))}
            </table>
        </div>

        <div class="section">
            <h2>六、问题总结</h2>
            <div class="problem-box">
                <strong>发现的主要问题：</strong>
                <ul>
                    <li>主要慢的环节：<b>{slow_stages.most_common(1)[0][0] if slow_stages else '-'}</b>，占比 {slow_stages.most_common(1)[0][1]/len(results)*100:.1f}%</li>
                    <li>根本原因：<b>{slow_reasons.most_common(1)[0][0] if slow_reasons else '-'}</b>，说明流程或服务存在问题</li>
                    <li>客服回复质量：{'良好' if response_quality.get('良好', 0) > response_quality.get('差', 0) else '需要改进'}，很多客户未获得满意答复</li>
                    <li>未解决案例：{resolution_status.get('未解决', 0)} 个，需要重点跟进</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>七、具体改进措施</h2>

            <div class="suggestion-box" style="background:#fff3cd; border-left:4px solid #F39C12;">
                <strong>一、流程优化措施：</strong>
                <ul>
                    <li><b>立案环节</b>：将立案时间缩短到3个工作日以内，超过3天自动发送进度提醒短信</li>
                    <li><b>审核环节</b>：材料不全时一次性告知所有缺失材料，避免反复驳回多次提交</li>
                    <li><b>打款环节</b>：打款失败时立即电话通知客户并提供具体解决方案（如更换银行卡）</li>
                    <li><b>整体流程</b>：建立理赔进度实时查询功能，客户可随时查看当前环节、预计完成时间</li>
                </ul>
            </div>

            <div class="suggestion-box" style="background:#d4edda; border-left:4px solid #27AE60;">
                <strong>二、客服培训措施：</strong>
                <ul>
                    <li><b>时效投诉处理</b>：客服在接到时效投诉时，先查询具体进度再回复，而不是只说"3-7个工作日"</li>
                    <li><b>进度沟通</b>：培训客服解释各环节具体时间节点（立案3天、审核10天、打款5天）</li>
                    <li><b>加急处理</b>：对等待超过15天的案件，客服可提供加急申请流程</li>
                    <li><b>材料指导</b>：客服需一次性告知客户所有所需材料清单，避免客户多次补交</li>
                </ul>
            </div>

            <div class="suggestion-box" style="background:#e8f4f8; border-left:4px solid #3498DB;">
                <strong>三、系统改进措施：</strong>
                <ul>
                    <li><b>时效预警</b>：案件超过10天未立案、20天未审核完成时，系统自动预警并提醒审核人员</li>
                    <li><b>进度透明</b>：在微信公众号增加"理赔进度查询"功能，显示当前环节和预计完成日期</li>
                    <li><b>材料预审</b>：上传材料时自动检测是否齐全，不全时提示具体缺失项</li>
                    <li><b>高峰应对</b>：年底高峰期自动增加审核人员配置，缩短平均处理时间</li>
                </ul>
            </div>

            <div class="suggestion-box" style="background:#f8d7da; border-left:4px solid #E74C3C;">
                <strong>四、具体案例改进建议（TOP5）：</strong>
                <ul>
                    {''.join(f"<li>{r.get('improvement_suggestion', '-') or '-'}</li>" for r in results[:5] if r.get('improvement_suggestion'))}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''

    # 保存报告
    report_path = os.path.join(output_dir, '理赔时效投诉深度分析.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n报告已生成: {report_path}")
    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='理赔时效投诉深度分析')
    parser.add_argument('--type', choices=['online', 'phone', 'all'],
                        default='all', help='数据来源: online=在线对话, phone=电话录音, all=全部')
    parser.add_argument('--limit', type=int, default=100, help='限制分析数量（默认100）')
    parser.add_argument('--no-api', action='store_true', help='不使用API分析（快速模式）')

    args = parser.parse_args()

    # 确定数据来源
    if args.type == 'all':
        source_type = None
    else:
        source_type = args.type

    # 分析（默认使用API）
    results = analyze_time_complaints(use_api=not args.no_api, source_type=source_type, limit=args.limit)

    # 生成报告
    if results:
        output_dir = os.path.join(PROJECT_ROOT, '05_analyze/reports')
        generate_html_report(results, output_dir)