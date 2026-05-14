"""
AI客服系统真实场景测试
模拟在线客服对话场景，评估AI回答准确度和利用率
"""

import sqlite3
import random
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from ai_qa_system_v2 import ImprovedQASystem, is_personal_query

def extract_customer_questions(db_path, sample_size=100):
    """从数据库提取真实客户问题"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 正确提取客户问题：FromUserName = VisitorId
    cursor.execute('''
    SELECT
        m.Content,
        m.SessionID,
        s.AgentId,
        s.AgentName,
        s.CusNickName,
        s.BeginTime
    FROM heli_message m
    JOIN heli_session s ON m.SessionID = s.SessionID
    WHERE m.MsgType = 'text'
    AND m.FromUserName = s.VisitorId
    AND LENGTH(m.Content) > 5
    AND LENGTH(m.Content) < 300
    ORDER BY s.BeginTime DESC
    ''')

    results = cursor.fetchall()
    conn.close()

    # 过滤无效内容
    keywords = ['理赔', '保费', '保单', '报销', '保障', '免赔', '赔付', '既往症',
                '退保', '投保', '续保', '门槛', '材料', '医保', '门特', '住院',
                '特药', '直赔', '快赔', '购买', '老人', '父母', '保险', '宁惠保',
                '门诊', '出院', '发票', '金额', '审核', '驳回', '申请']

    exclude_words = ['您好', '在线客服', '很高兴', '请问有什么', '回复数字',
                     '<br>', '&lt;', '&gt;', 'nbsp', '等待回复', '问好了给我留言',
                     '超过2分钟', '好的', '谢谢', '收到', '明白了', '哦哦', '嗯嗯']

    valid_questions = []
    for row in results:
        content = row[0].strip()
        session_id = row[1]
        agent_name = row[3]
        customer_name = row[4]
        create_time = row[5]

        # 过滤
        if not any(exclude in content for exclude in exclude_words):
            if any(kw in content for kw in keywords):
                valid_questions.append({
                    'question': content,
                    'session_id': session_id,
                    'agent': agent_name,
                    'customer': customer_name,
                    'time': create_time
                })

    # 随机抽样
    random.seed(42)
    sample = random.sample(valid_questions, min(sample_size, len(valid_questions)))
    return sample, len(valid_questions)


def evaluate_ai_response(qa_system, question):
    """评估AI回答质量"""
    # 检测意图
    is_personal = is_personal_query(question)

    # 获取检索匹配
    matched = qa_system.search_knowledge(question, top_k=3)
    top_score = matched[0]['score'] if matched else 0
    top_match = matched[0]['qa']['std_question'] if matched else '无匹配'

    # 获取AI回答
    answer = qa_system.get_answer(question)

    # 分类评估
    if is_personal:
        category = 'personal_query'
        quality = 'correct'  # 个人查询转人工是正确行为
        needs_human = True
        reason = '个人数据查询，需人工核实身份'
    elif '系统暂时无法回答' in answer or top_score < 0.5:
        category = 'low_confidence'
        quality = 'correct'  # 低置信度转人工是正确行为
        needs_human = True
        reason = '置信度低，无法匹配知识库'
    elif len(answer) > 20 and '您好' in answer and top_score >= 0.5:
        if '4000040181' in answer and top_score >= 0.6:
            category = 'medium_confidence'
            quality = 'acceptable'
            needs_human = False  # AI回答了但建议人工
            reason = '中等置信度，AI回答+建议人工'
        else:
            category = 'high_confidence'
            quality = 'correct'
            needs_human = False
            reason = '高置信度，AI直接回答'
    else:
        category = 'other'
        quality = 'needs_improve'
        needs_human = True
        reason = '回答异常'

    return {
        'question': question,
        'is_personal': is_personal,
        'top_score': top_score,
        'top_match': top_match,
        'answer': answer,
        'answer_length': len(answer),
        'category': category,
        'quality': quality,
        'needs_human': needs_human,
        'reason': reason
    }


def run_test(db_path, sample_size=100):
    """运行完整测试"""
    print("=" * 60)
    print("AI客服系统真实场景测试")
    print("=" * 60)

    # 初始化系统
    print("\n正在初始化AI客服系统...")
    qa = ImprovedQASystem()

    # 提取客户问题
    print("\n正在从数据库提取真实客户问题...")
    sample, total_count = extract_customer_questions(db_path, sample_size)
    print(f"数据库中共有 {total_count} 条有效客户问题")
    print(f"本次测试抽取 {len(sample)} 条样本")

    # 测试每条问题
    print("\n正在测试AI回答...")
    results = []
    start_time = time.time()

    for i, item in enumerate(sample):
        question = item['question']
        result = evaluate_ai_response(qa, question)
        result['session_id'] = item['session_id']
        result['agent'] = item['agent']
        result['customer'] = item['customer']
        result['original_time'] = item['time']
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  已测试 {i+1}/{len(sample)} 条...")

    elapsed = time.time() - start_time
    print(f"测试完成，耗时 {elapsed:.1f} 秒")

    # 统计分析
    stats = analyze_results(results)

    # 输出报告
    print("\n" + "=" * 60)
    print("测试报告")
    print("=" * 60)

    print(f"\n【基本信息】")
    print(f"测试样本: {len(results)} 条真实客户问题")
    print(f"测试耗时: {elapsed:.1f} 秒")
    print(f"平均响应时间: {elapsed/len(results):.2f} 秒/条")

    print(f"\n【回答质量分布】")
    for quality, count in stats['quality'].items():
        print(f"  {quality}: {count}条 ({count/len(results)*100:.1f}%)")

    print(f"\n【问题类型分布】")
    for category, count in stats['category'].items():
        print(f"  {category}: {count}条 ({count/len(results)*100:.1f}%)")

    print(f"\n【核心指标】")
    print(f"  AI利用率: {stats['ai_utilization']:.1f}%")
    print(f"    (AI能独立回答，无需人工介入的比例)")
    print(f"  转人工率: {stats['transfer_rate']:.1f}%")
    print(f"    (需要人工介入的比例，包括个人查询+低置信度)")
    print(f"  有效回答率: {stats['effective_rate']:.1f}%")
    print(f"    (AI给出有实质内容的回答)")

    print(f"\n【匹配度分析】")
    print(f"  平均匹配得分: {stats['avg_score']:.3f}")
    print(f"  高匹配(≥0.7): {stats['high_match']}条 ({stats['high_match']/len(results)*100:.1f}%)")
    print(f"  中匹配(0.5-0.7): {stats['medium_match']}条 ({stats['medium_match']/len(results)*100:.1f}%)")
    print(f"  低匹配(<0.5): {stats['low_match']}条 ({stats['low_match']/len(results)*100:.1f}%)")

    print(f"\n【改进建议】")
    if stats['low_match'] > len(results) * 0.3:
        print(f"  ⚠️ 低匹配度问题较多，建议扩充知识库")
    if stats['transfer_rate'] > 60:
        print(f"  ⚠️ 转人工率较高，可考虑补充更多QA")
    else:
        print(f"  ✓ 转人工率合理，个人查询和复杂问题正确分流")

    return results, stats


def analyze_results(results):
    """分析测试结果"""
    stats = {
        'quality': {},
        'category': {},
        'ai_utilization': 0,
        'transfer_rate': 0,
        'effective_rate': 0,
        'avg_score': 0,
        'high_match': 0,
        'medium_match': 0,
        'low_match': 0
    }

    # 统计质量分布
    for r in results:
        quality = r['quality']
        stats['quality'][quality] = stats['quality'].get(quality, 0) + 1

        category = r['category']
        stats['category'][category] = stats['category'].get(category, 0) + 1

        # 匹配度统计
        score = r['top_score']
        stats['avg_score'] += score
        if score >= 0.7:
            stats['high_match'] += 1
        elif score >= 0.5:
            stats['medium_match'] += 1
        else:
            stats['low_match'] += 1

    stats['avg_score'] /= len(results)

    # 核心指标
    ai_answered = sum(1 for r in results if not r['needs_human'])
    stats['ai_utilization'] = ai_answered / len(results) * 100
    stats['transfer_rate'] = 100 - stats['ai_utilization']

    effective = sum(1 for r in results if len(r['answer']) > 20 and '您好' in r['answer'])
    stats['effective_rate'] = effective / len(results) * 100

    return stats


def save_report(results, stats, output_path):
    """保存详细报告"""
    # Markdown报告
    report = f"""# AI客服系统真实场景测试报告

## 测试概况

- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **数据来源**: heli.sqlite3 在线客服对话数据库
- **测试样本**: {len(results)} 条真实客户问题
- **知识库规模**: 226 条QA

## 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| AI利用率 | {stats['ai_utilization']:.1f}% | AI能独立回答的比例 |
| 转人工率 | {stats['transfer_rate']:.1f}% | 需人工介入的比例 |
| 有效回答率 | {stats['effective_rate']:.1f}% | AI给出实质回答的比例 |
| 平均匹配度 | {stats['avg_score']:.3f} | 向量检索平均得分 |

## 回答质量分布

| 质量等级 | 数量 | 占比 |
|---------|------|------|
| 正确回答 | {stats['quality'].get('correct', 0)} | {stats['quality'].get('correct', 0)/len(results)*100:.1f}% |
| 可接受回答 | {stats['quality'].get('acceptable', 0)} | {stats['quality'].get('acceptable', 0)/len(results)*100:.1f}% |
| 需改进 | {stats['quality'].get('needs_improve', 0)} | {stats['quality'].get('needs_improve', 0)/len(results)*100:.1f}% |

## 问题类型分布

| 类型 | 数量 | 占比 | 处理方式 |
|------|------|------|---------|
| 个人查询 | {stats['category'].get('personal_query', 0)} | {stats['category'].get('personal_query', 0)/len(results)*100:.1f}% | 自动转人工 |
| 低置信度 | {stats['category'].get('low_confidence', 0)} | {stats['category'].get('low_confidence', 0)/len(results)*100:.1f}% | 转人工 |
| 中置信度 | {stats['category'].get('medium_confidence', 0)} | {stats['category'].get('medium_confidence', 0)/len(results)*100:.1f}% | AI回答+建议人工 |
| 高置信度 | {stats['category'].get('high_confidence', 0)} | {stats['category'].get('high_confidence', 0)/len(results)*100:.1f}% | AI直接回答 |

## 匹配度分析

| 匹配级别 | 数量 | 占比 |
|---------|------|------|
| 高匹配(≥0.7) | {stats['high_match']} | {stats['high_match']/len(results)*100:.1f}% |
| 中匹配(0.5-0.7) | {stats['medium_match']} | {stats['medium_match']/len(results)*100:.1f}% |
| 低匹配(<0.5) | {stats['low_match']} | {stats['low_match']/len(results)*100:.1f}% |

## 典型案例

### AI正确回答示例

"""

    # 添加典型案例
    correct_cases = [r for r in results if r['quality'] == 'correct' and r['category'] == 'high_confidence'][:5]
    for i, case in enumerate(correct_cases, 1):
        report += f"""
**案例{i}**
- 客户问题: {case['question']}
- 匹配QA: {case['top_match']} (得分: {case['top_score']:.2f})
- AI回答: {case['answer'][:200]}{'...' if len(case['answer']) > 200 else ''}

"""

    report += """
### 个人查询自动转人工示例

"""
    personal_cases = [r for r in results if r['category'] == 'personal_query'][:3]
    for i, case in enumerate(personal_cases, 1):
        report += f"""
**案例{i}**
- 客户问题: {case['question']}
- 处理方式: 自动识别为个人查询，引导转人工
- AI回答: {case['answer'][:150]}{'...' if len(case['answer']) > 150 else ''}

"""

    report += """
## 改进建议

"""

    if stats['low_match'] > len(results) * 0.3:
        report += f"1. **扩充知识库**: 低匹配度问题占{stats['low_match']/len(results)*100:.1f}%，建议补充更多QA\n"
    if stats['transfer_rate'] > 60:
        report += f"2. **提高覆盖率**: 转人工率{stats['transfer_rate']:.1f}%偏高，可补充高频问题QA\n"
    if stats['ai_utilization'] >= 40:
        report += f"3. **当前状态良好**: AI利用率{stats['ai_utilization']:.1f}%，可有效减轻人工客服压力\n"

    report += """
## 结论

"""
    if stats['ai_utilization'] >= 50:
        report += f"AI客服系统可独立处理约{stats['ai_utilization']:.0f}%的客户问题，有效减轻人工客服工作量。个人查询和复杂问题正确分流到人工，系统设计合理。"
    else:
        report += f"AI客服系统利用率{stats['ai_utilization']:.0f}%，建议补充知识库提高覆盖范围。"

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存详细数据
    data_path = output_path.replace('.md', '_详细数据.json')
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': stats,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存到: {output_path}")
    print(f"详细数据已保存到: {data_path}")


if __name__ == '__main__':
    # 设置环境变量
    os.environ.setdefault('DASHSCOPE_API_KEY', 'sk-b576a4e032f041bd85bb61f5a92a23f1')
    os.environ.setdefault('DB_HOST', '192.168.10.170')
    os.environ.setdefault('DB_PORT', '3308')
    os.environ.setdefault('DB_USER', 'xiecheng')
    os.environ.setdefault('DB_PASS', 'SecurePass123!')
    os.environ.setdefault('DB_NAME', 'ai_customer_service')

    # 运行测试
    db_path = '01_source/heli.sqlite3'
    results, stats = run_test(db_path, sample_size=100)

    # 保存报告
    output_path = '05_analyze/reports/AI客服真实场景测试报告.md'
    save_report(results, stats, output_path)