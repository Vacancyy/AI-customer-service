"""找出知识库中相比Excel新增的问答"""
import os
import json
import pandas as pd

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json'), 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

df = pd.read_excel(os.path.join(PROJECT_ROOT, 'docs/customerQA.xlsx'))

def clean_text(t):
    return t.replace('\u201c', '').replace('\u201d', '').replace('"', '').replace('\uff1f', '').replace('?', '').strip()

# Excel问题集合
excel_questions = set()
for q in df['标准问'].dropna():
    excel_questions.add(clean_text(q))

print(f"Excel问题数: {len(excel_questions)}")
print(f"知识库总条数: {len(qa_data)}")

# 找出知识库中Excel没有的 = 增量
incremental = []
for qa in qa_data:
    q_clean = clean_text(qa['std_question'])
    if q_clean not in excel_questions:
        incremental.append(qa)

print(f"增量问答数: {len(incremental)}")

# 保存
output_path = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_增量问答.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(incremental, f, ensure_ascii=False, indent=2)
print(f"已保存: {output_path}")

# 统计增量分类
from collections import Counter
cats = Counter(qa.get('primary_category', '') for qa in incremental)
print(f"\n增量分类分布:")
for cat, count in cats.most_common():
    print(f"  {cat}: {count}")

# 按来源统计
sources = Counter(qa.get('source', '') for qa in incremental)
print(f"\n增量来源分布:")
for src, count in sources.most_common():
    print(f"  {src}: {count}")

# 列出前10条增量
print(f"\n增量问答前10条:")
for i, qa in enumerate(incremental[:10], 1):
    print(f"  {i}. [{qa.get('primary_category','')}] {qa['std_question']}")
