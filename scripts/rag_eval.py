"""
RAG评估工具
用大模型对问答系统的回答进行多维度打分
评估维度：相关性、准确性、完整性、表达性（各0-10分）
"""

import os
import json
import requests
import time
import sys

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# 答题模型（被评估的系统）和评分模型（评估者）分开
QA_MODEL = "qwen3-8b"        # 答题模型
JUDGE_MODEL = "qwen-max"     # 评分模型（更强的模型，避免自己给自己打分）

QA_JSON_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json')


def call_llm(prompt, max_tokens=500):
    """调用大模型"""
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def evaluate_single(question, ground_truth, system_answer):
    """
    评估单个问答
    question: 客户问题
    ground_truth: 知识库原始回答（标准答案）
    system_answer: 系统生成的回答
    返回: 4个维度的分数 + 总评
    """
    prompt = f"""你是RAG问答系统评估专家。请对以下问答进行打分。

【客户问题】
{question}

【知识库标准答案】
{ground_truth}

【系统生成的回答】
{system_answer}

请从以下4个维度打分（0-10分）：

1. 相关性（Relevance）：系统回答是否切题，是否在回答客户问的问题
   - 10分：完全切题  0分：答非所问

2. 准确性（Faithfulness）：系统回答是否忠实于知识库内容，有没有编造信息
   - 10分：完全忠实  0分：大量编造

3. 完整性（Completeness）：知识库标准答案中的关键信息是否都被包含
   - 10分：关键信息全覆盖  0分：缺失所有关键信息

4. 表达性（Fluency）：回答是否通顺自然、以"您好"开头、简洁清晰
   - 10分：表达完美  0分：语无伦次

请严格按以下JSON格式输出，不要输出其他内容：
{{"relevance": 分数, "faithfulness": 分数, "completeness": 分数, "fluency": 分数, "comment": "简要评语"}}"""

    result = call_llm(prompt, max_tokens=200)

    # 解析JSON
    try:
        # 提取JSON部分
        if '{' in result and '}' in result:
            json_str = result[result.index('{'):result.rindex('}')+1]
            scores = json.loads(json_str)
            return {
                'relevance': int(scores.get('relevance', 0)),
                'faithfulness': int(scores.get('faithfulness', 0)),
                'completeness': int(scores.get('completeness', 0)),
                'fluency': int(scores.get('fluency', 0)),
                'comment': scores.get('comment', ''),
            }
    except:
        pass

    return {
        'relevance': 0,
        'faithfulness': 0,
        'completeness': 0,
        'fluency': 0,
        'comment': f'解析失败: {result[:100]}',
    }


def run_evaluation(sample_size=20, output_path=None):
    """运行全量评估"""
    from ai_qa_system_v2 import ImprovedQASystem

    # 加载知识库
    with open(QA_JSON_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 初始化系统
    system = ImprovedQASystem()

    # 按分类均匀采样
    from collections import defaultdict
    by_category = defaultdict(list)
    for i, qa in enumerate(qa_data):
        by_category[qa.get('primary_category', '其他')].append(i)

    sample_indices = []
    per_cat = max(1, sample_size // len(by_category))
    for cat, indices in by_category.items():
        sample_indices.extend(indices[:per_cat])

    # 补足到sample_size
    remaining = [i for i in range(len(qa_data)) if i not in sample_indices]
    sample_indices.extend(remaining[:sample_size - len(sample_indices)])
    sample_indices = sample_indices[:sample_size]

    print(f"评估样本: {len(sample_indices)} 条")
    print("=" * 70)

    results = []
    all_scores = {'relevance': [], 'faithfulness': [], 'completeness': [], 'fluency': []}

    for rank, idx in enumerate(sample_indices, 1):
        qa = qa_data[idx]
        question = qa['std_question']
        ground_truth = qa['answer']

        # 系统回答
        system_answer = system.get_answer(question)

        # 评估打分
        scores = evaluate_single(question, ground_truth, system_answer)

        # 记录
        for dim in all_scores:
            all_scores[dim].append(scores[dim])

        result = {
            'question': question,
            'category': qa.get('primary_category', ''),
            'system_answer': system_answer[:150],
            'scores': scores,
        }
        results.append(result)

        avg = sum(scores[d] for d in ['relevance', 'faithfulness', 'completeness', 'fluency']) / 4
        status = '✓' if avg >= 8 else ('△' if avg >= 6 else '✗')
        print(f"{status} [{rank}/{len(sample_indices)}] {question[:35]}... | "
              f"相关{scores['relevance']} 准确{scores['faithfulness']} "
              f"完整{scores['completeness']} 表达{scores['fluency']} | 均{avg:.1f}")

        time.sleep(0.5)  # 避免API限流

    # 汇总
    print("\n" + "=" * 70)
    print("评估报告")
    print("=" * 70)

    for dim in ['relevance', 'faithfulness', 'completeness', 'fluency']:
        scores_list = all_scores[dim]
        avg = sum(scores_list) / len(scores_list)
        dim_cn = {'relevance': '相关性', 'faithfulness': '准确性', 'completeness': '完整性', 'fluency': '表达性'}[dim]
        print(f"  {dim_cn}: {avg:.1f} / 10")

    total_avg = sum(sum(r['scores'][d] for d in ['relevance', 'faithfulness', 'completeness', 'fluency']) / 4 for r in results) / len(results)
    print(f"  总评: {total_avg:.1f} / 10")

    # 质量分布
    avgs = [sum(r['scores'][d] for d in ['relevance', 'faithfulness', 'completeness', 'fluency']) / 4 for r in results]
    excellent = sum(1 for a in avgs if a >= 9)
    good = sum(1 for a in avgs if 7 <= a < 9)
    medium = sum(1 for a in avgs if 5 <= a < 7)
    poor = sum(1 for a in avgs if a < 5)
    print(f"\n质量分布: 优秀(≥9):{excellent} | 良好(7-9):{good} | 中等(5-7):{medium} | 较差(<5):{poor}")

    # 保存结果
    if output_path:
        report = {
            'total_samples': len(sample_indices),
            'dimension_scores': {dim: sum(all_scores[dim]) / len(all_scores[dim]) for dim in all_scores},
            'overall_score': total_avg,
            'distribution': {'excellent': excellent, 'good': good, 'medium': medium, 'poor': poor},
            'details': results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {output_path}")

    return results


if __name__ == "__main__":
    size = 20
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except:
            pass

    output = os.path.join(PROJECT_ROOT, '05_analyze/reports/rag_eval_report.json')
    run_evaluation(sample_size=size, output_path=output)
