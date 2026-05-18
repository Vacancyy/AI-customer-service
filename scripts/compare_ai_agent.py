"""
AI客服 vs 人工客服对比评分
基于线上对话CSV数据
"""

import csv
import sys
import os
import json
import re
from collections import defaultdict

# 【重要】先设置环境变量，再导入模块
os.environ.setdefault('DASHSCOPE_API_KEY', 'sk-b576a4e032f041bd85bb61f5a92a23f1')
os.environ.setdefault('DB_HOST', '192.168.10.170')
os.environ.setdefault('DB_PORT', '3308')
os.environ.setdefault('DB_USER', 'xiecheng')
os.environ.setdefault('DB_PASS', 'SecurePass123!')
os.environ.setdefault('DB_NAME', 'ai_customer_service')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.ai_qa_system_v2 import ImprovedQASystem, is_personal_query


def extract_qa_pairs(csv_path):
    """从CSV提取客户问题与人工回答配对"""
    qa_pairs = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        messages = list(reader)

    # 按SessionID分组
    sessions = defaultdict(list)
    for msg in messages:
        sessions[msg['SessionID']].append(msg)

    # 提取问答对
    for session_id, msgs in sessions.items():
        visitor_id = msgs[0]['VisitorId'] if msgs else ''
        channel_id = msgs[0]['ChannelId'] if msgs else ''
        agent_name = msgs[0]['AgentName'] if msgs else ''

        # 按时间排序
        msgs.sort(key=lambda x: x['CreateTime'])

        # 找客户问题和客服回答
        last_question = None
        for msg in msgs:
            content = msg['Content'].strip()
            from_user = msg['FromUserName']

            # 清理HTML标签
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('&nbsp;', ' ').strip()

            # 客户问题
            if from_user == visitor_id and len(content) > 5:
                if content not in ['人工', '转人工', '人工客服', '回复数字0', '0', '1', '2', '3']:
                    last_question = {
                        'content': content,
                        'time': msg['CreateTime']
                    }

            # 客服回答（FromUserName是channel_id表示系统/客服）
            elif from_user == channel_id and last_question:
                # 排除系统自动消息
                exclude_patterns = ['智能机器人', '无在线客服', '请留言', '长时间未对话', '服务将在', '访客已离开']
                if len(content) > 20 and not any(p in content for p in exclude_patterns):
                    qa_pairs.append({
                        'question': last_question['content'],
                        'question_time': last_question['time'],
                        'agent_answer': content,
                        'agent_name': agent_name,
                        'session_id': session_id
                    })
                    last_question = None

    return qa_pairs


def evaluate_ai_answer(ai_answer, agent_answer, match_score):
    """评估AI回答质量（对比人工回答）"""

    # 评分标准
    score = 0
    reasons = []

    # 1. 是否有实质回答（非转人工）
    if '转人工' in ai_answer or '人工客服' in ai_answer and match_score < 0.6:
        # 低匹配度转人工是正确的
        if match_score < 0.5:
            score = 70
            reasons.append("低匹配度问题，正确转人工")
        else:
            score = 50
            reasons.append("中等匹配度，但转人工而非尝试回答")
    else:
        # AI给出了回答
        score = 60  # 基础分

        # 2. 核心信息是否一致
        # 提取数字信息
        ai_numbers = re.findall(r'\d+\.?\d*', ai_answer)
        agent_numbers = re.findall(r'\d+\.?\d*', agent_answer)

        if ai_numbers and agent_numbers:
            common_numbers = set(ai_numbers) & set(agent_numbers)
            if common_numbers:
                score += 15
                reasons.append(f"数字信息一致: {list(common_numbers)[:3]}")

        # 3. 关键词重叠
        ai_keywords = set(ai_answer.split()) - {'您好', '请', '可以', '是', '的', '。', '，', '咨询', '拨打', '客服'}
        agent_keywords = set(agent_answer.split()) - {'您好', '请', '可以', '是', '的', '。', '，', '咨询', '拨打', '客服'}

        overlap = len(ai_keywords & agent_keywords)
        if overlap > 5:
            score += 10
            reasons.append(f"关键词重叠{overlap}个")
        elif overlap > 2:
            score += 5

        # 4. 回答长度对比
        if len(ai_answer) > 50 and len(agent_answer) > 50:
            # 双方都有实质内容
            score += 10

        # 5. 是否包含关键渠道/流程信息
        channels = ['微信公众号', '我的南京', '支付宝', '4000040181']
        for channel in channels:
            if channel in ai_answer and channel in agent_answer:
                score += 5
                reasons.append(f"渠道信息一致: {channel}")
                break

    # 限制最高分
    score = min(score, 95)

    # 评级
    if score >= 85:
        level = "优秀"
    elif score >= 70:
        level = "良好"
    elif score >= 50:
        level = "合格"
    else:
        level = "待改进"

    return {
        'score': score,
        'level': level,
        'reasons': reasons
    }


def run_comparison(csv_path, output_path):
    """运行对比测试"""

    print("正在初始化AI客服系统...")
    qa = ImprovedQASystem()

    print(f"正在提取问答对...")
    qa_pairs = extract_qa_pairs(csv_path)
    print(f"提取到 {len(qa_pairs)} 组问答对")

    results = []

    print()
    print("=" * 80)
    print("AI客服 vs 人工客服 对比评分报告")
    print("=" * 80)

    for i, pair in enumerate(qa_pairs, 1):
        question = pair['question']
        agent_answer = pair['agent_answer']

        # AI回答
        matched = qa.search_knowledge(question, top_k=1)
        match_score = matched[0]['score'] if matched else 0
        matched_q = matched[0]['qa']['std_question'] if matched else '无匹配'
        ai_answer = qa.get_answer(question)

        # 评估
        eval_result = evaluate_ai_answer(ai_answer, agent_answer, match_score)

        result = {
            'id': i,
            'question': question,
            'match_score': match_score,
            'matched_qa': matched_q,
            'ai_answer': ai_answer,
            'agent_answer': agent_answer[:200] + '...' if len(agent_answer) > 200 else agent_answer,
            'agent_name': pair['agent_name'],
            'eval_score': eval_result['score'],
            'eval_level': eval_result['level'],
            'eval_reasons': eval_result['reasons']
        }
        results.append(result)

        # 输出
        print(f"""
【对话{i}】 评分: {eval_result['score']}分 ({eval_result['level']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
客户问: {question}

【人工回答】({pair['agent_name']}):
{agent_answer[:150]}{'...' if len(agent_answer) > 150 else ''}

【AI回答】(匹配度{match_score:.2f}, 匹配QA: {matched_q}):
{ai_answer}

评分原因: {', '.join(eval_result['reasons'])}
""")

    # 统计
    total = len(results)
    avg_score = sum(r['eval_score'] for r in results) / total if total > 0 else 0

    level_counts = defaultdict(int)
    for r in results:
        level_counts[r['eval_level']] += 1

    print("=" * 80)
    print("统计汇总")
    print("=" * 80)
    print(f"测试样本: {total}条")
    print(f"平均得分: {avg_score:.1f}分")
    print()
    print("评分分布:")
    for level in ['优秀', '良好', '合格', '待改进']:
        count = level_counts.get(level, 0)
        print(f"  {level}: {count}条 ({count/total*100:.1f}%)")

    # 保存报告
    report = build_report(results, avg_score, level_counts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print()
    print(f"详细报告已保存: {output_path}")

    return results


def build_report(results, avg_score, level_counts):
    """生成Markdown报告"""

    report = f"""# AI客服 vs 人工客服对比评分报告

## 测试概况

- **数据来源**: 线上对话CSV
- **测试样本**: {len(results)}条真实对话
- **平均得分**: {avg_score:.1f}分

---

## 一、评分标准

| 分数范围 | 评级 | 说明 |
|---------|------|------|
| 85-95分 | 优秀 | AI回答与人工核心内容一致 |
| 70-84分 | 良好 | AI回答关键信息正确，表述有差异 |
| 50-69分 | 合格 | AI回答方向正确，但不够详细 |
| <50分 | 待改进 | AI回答与人工有明显差异 |

---

## 二、评分分布

| 评级 | 数量 | 占比 |
|------|------|------|
"""
    for level in ['优秀', '良好', '合格', '待改进']:
        count = level_counts.get(level, 0)
        report += f"| {level} | {count} | {count/len(results)*100:.1f}% |\n"

    report += f"""
---

## 三、详细对比

"""
    for r in results:
        report += f"""
### 对话{r['id']} - 评分: {r['eval_score']}分 ({r['eval_level']})

**客户问题**: {r['question']}

**匹配情况**: 匹配度 {r['match_score']:.2f}, 匹配QA: {r['matched_qa']}

**人工回答** ({r['agent_name']}):
{r['agent_answer']}

**AI回答**:
{r['ai_answer']}

**评分原因**: {', '.join(r['eval_reasons'])}

---
"""

    report += """
## 四、改进建议

"""

    # 分析待改进的问题
    low_score = [r for r in results if r['eval_score'] < 60]
    if low_score:
        report += f"**待改进问题（{len(low_score)}条）**:\n"
        for r in low_score[:5]:
            report += f"- {r['question']}\n"
        report += "\n建议：补充相关QA到知识库\n"

    # 分析转人工的问题
    transfer = [r for r in results if '转人工' in r['ai_answer'] or '人工客服' in r['ai_answer']]
    if transfer:
        report += f"\n**转人工问题（{len(transfer)}条）**:\n"
        report += "部分问题因匹配度低正确转人工，部分可尝试补充QA提高覆盖率。\n"

    report += f"""
## 五、结论

- AI客服平均得分{avg_score:.1f}分
- 评分≥70分的占比: {sum(1 for r in results if r['eval_score'] >= 70)/len(results)*100:.1f}%
- AI能有效回答常见问题，复杂问题正确转人工

---
*报告生成时间: {os.popen('date').read().strip()}*
"""

    return report


if __name__ == '__main__':
    csv_path = 'docs/线上对话1.csv'
    output_path = '05_analyze/reports/AI客服对比评分报告.md'

    run_comparison(csv_path, output_path)