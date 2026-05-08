import os
"""
知识库自检工具
用Qwen-Max逐条审查222条QA，挑出可能有问题的条目
检查维度：
1. 回答是否包含具体数字且前后一致
2. 回答是否明显不完整或过短
3. 回答是否可能已过时
4. 同一问题是否有多个重复/矛盾的回答
"""

import json
import requests
import time
import sys
from collections import defaultdict

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
JUDGE_MODEL = "qwen-max"

QA_JSON_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_优化版.json'


def call_llm(prompt, max_tokens=300):
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


def check_entry(question, answer, category):
    """检查单条QA是否有问题"""
    prompt = f"""你是保险产品知识库审核员。请审查以下知识库条目是否有问题。

类别: {category}
问题: {question}
回答: {answer}

请检查以下4点，如有问题请指出：
1. 数字矛盾：回答中的数字是否前后矛盾（如免赔额写了两个不同的值）
2. 回答不完整：回答是否明显不完整，缺少关键信息
3. 可疑过时：回答中是否有可能过时的内容（如引用了过去的日期、旧的产品版本）
4. 不切题：回答是否没有真正回答问题

如果没有问题，回复: 无问题
如果有问题，回复格式: 问题类型|具体描述

只输出一行:"""

    result = call_llm(prompt, max_tokens=100)
    return result


def find_duplicates(qa_data):
    """找出相似/重复的问题"""
    from difflib import SequenceMatcher

    similar_groups = []
    checked = set()

    # 按分类分组
    by_category = defaultdict(list)
    for i, qa in enumerate(qa_data):
        by_category[qa.get('primary_category', '')].append(i)

    for cat, indices in by_category.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if (i, j) in checked:
                    continue
                checked.add((i, j))

                q1 = qa_data[i]['std_question']
                q2 = qa_data[j]['std_question']

                # 标准化后比较
                def normalize(t):
                    return t.replace('"', '').replace('"', '').replace('"', '').replace('？', '').replace('?', '').strip()

                similarity = SequenceMatcher(None, normalize(q1), normalize(q2)).ratio()

                if similarity > 0.7:
                    similar_groups.append({
                        'q1': q1,
                        'q2': q2,
                        'a1': qa_data[i]['answer'][:80],
                        'a2': qa_data[j]['answer'][:80],
                        'similarity': similarity,
                        'category': cat,
                    })

    return similar_groups


def run_check(sample_size=None):
    """运行自检"""
    with open(QA_JSON_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"知识库条数: {len(qa_data)}")
    print("=" * 70)

    # ===== 第一步: 重复问题检测 =====
    print("\n【第一步: 重复/相似问题检测】\n")
    similar = find_duplicates(qa_data)
    print(f"发现 {len(similar)} 组相似问题:\n")
    for s in similar:
        print(f"  [{s['category']}] 相似度{s['similarity']:.2f}")
        print(f"    Q1: {s['q1']}")
        print(f"    A1: {s['a1']}...")
        print(f"    Q2: {s['q2']}")
        print(f"    A2: {s['a2']}...")
        print()

    # ===== 第二步: LLM逐条审查 =====
    print("\n【第二步: LLM逐条审查】\n")

    if sample_size:
        import random
        indices = random.sample(range(len(qa_data)), min(sample_size, len(qa_data)))
    else:
        indices = range(len(qa_data))

    issues = []
    for rank, idx in enumerate(indices, 1):
        qa = qa_data[idx]
        result = check_entry(qa['std_question'], qa['answer'], qa.get('primary_category', ''))

        if result != "无问题" and not result.startswith("ERROR"):
            issues.append({
                'index': idx,
                'question': qa['std_question'],
                'category': qa.get('primary_category', ''),
                'issue': result,
                'answer_preview': qa['answer'][:100],
            })
            print(f"[{rank}] ⚠ {qa['std_question'][:40]}... → {result}")
        else:
            if rank % 20 == 0:
                print(f"[{rank}] 已检查...")

        time.sleep(0.3)

    print(f"\n发现 {len(issues)} 条可疑问题\n")

    # ===== 输出结果 =====
    if issues:
        print("=" * 70)
        print("可疑问题清单（需人工确认）")
        print("=" * 70)
        for i, item in enumerate(issues, 1):
            print(f"\n{i}. [{item['category']}] {item['question']}")
            print(f"   问题: {item['issue']}")
            print(f"   当前回答: {item['answer_preview']}...")

    # 保存
    output = {
        'total_checked': len(list(indices)),
        'issues_found': len(issues),
        'similar_groups': len(similar),
        'issues': issues,
        'similar_details': similar,
    }
    output_path = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库自检报告.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    size = None
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except:
            pass
    run_check(sample_size=size)
