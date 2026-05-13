"""
RAGAS评估：用行业标准指标评估RAG系统

指标说明：
- Faithfulness: 忠实度，回答是否忠于检索内容（不编造）
- Response Relevancy: 回答相关性，回答是否切题
- Context Precision: 检索精度，检索结果中有多少是真正有用的
- Factual Correctness: 事实正确性，回答与标准答案的事实一致性

使用: python ragas_eval.py [--sample_size 20]
"""

import os
import sys
import json
import random
import time
import argparse

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

# 设置API密钥
os.environ.setdefault("DASHSCOPE_API_KEY", "")
API_KEY = os.environ.get("DASHSCOPE_API_KEY")

from ai_qa_system_v2 import ImprovedQASystem

# RAGAS相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    FactualCorrectness,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate

# ==================== 配置 ====================
QA_JSON_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/ragas_eval_report.json')

# 评判模型用Qwen-Max（更准确），不用8B
JUDGE_MODEL = "qwen-max"
EMBEDDING_MODEL = "text-embedding-v3"


def get_llm():
    """获取评判用LLM"""
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=API_KEY,
        temperature=0,
    )
    return LangchainLLMWrapper(langchain_llm=llm)


def get_embeddings():
    """获取Embedding"""
    emb = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=API_KEY,
        dimensions=1024,
        check_embedding_ctx_length=False,
    )
    return LangchainEmbeddingsWrapper(embeddings=emb)


def build_samples(qa_system, qa_data, sample_size=20):
    """构建评估样本：从知识库抽样 → RAG检索+生成 → 组装RAGAS格式"""
    random.seed(42)

    # 分层抽样，确保各分类都有
    groups = {}
    for i, qa in enumerate(qa_data):
        cat = qa.get('primary_category', '其他')
        if cat not in groups:
            groups[cat] = []
        groups[cat].append((i, qa))

    # 每个分类按比例抽样
    total = len(qa_data)
    selected = []
    for cat, items in groups.items():
        n = max(1, round(sample_size * len(items) / total))
        random.shuffle(items)
        for idx, qa in items[:n]:
            selected.append((idx, qa))

    print(f"抽取 {len(selected)} 题进行评估...")

    samples = []
    for i, (idx, qa) in enumerate(selected):
        question = qa['std_question']
        reference = qa['answer']

        print(f"  [{i+1}/{len(selected)}] {question[:30]}...", end=' ', flush=True)
        start = time.time()

        # RAG检索
        matched = qa_system.search_knowledge(question, top_k=3)
        retrieved_contexts = [item['qa']['answer'] for item in matched]

        # RAG生成
        response = qa_system.get_answer(question)
        elapsed = time.time() - start
        print(f"✓ ({elapsed:.1f}s)")

        samples.append(SingleTurnSample(
            user_input=question,
            response=response,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
            reference_contexts=[reference],  # 标准答案作为参考上下文
        ))

    return samples


def run_evaluation(sample_size=20):
    """运行RAGAS评估"""
    print("=" * 60)
    print("  RAGAS 评估 - 行业标准RAG质量评估")
    print("=" * 60)

    # 初始化RAG系统
    print("\n[1/3] 初始化RAG系统...")
    qa_system = ImprovedQASystem()

    # 加载知识库
    with open(QA_JSON_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 构建样本
    print(f"\n[2/3] 生成评估样本（{sample_size}题）...")
    samples = build_samples(qa_system, qa_data, sample_size)

    # 运行评估
    print(f"\n[3/3] 运行RAGAS评估（评判模型: {JUDGE_MODEL}）...")
    print("  评估维度: Faithfulness / Response Relevancy / Context Precision / Factual Correctness")
    print("  预计耗时: 5-10分钟\n")

    llm = get_llm()
    embeddings = get_embeddings()

    # 配置指标
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        FactualCorrectness(llm=llm),
    ]

    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)

    # 从EvaluationResult正确读取分数
    df = result.to_pandas()
    scores = {}
    metric_cols = [c for c in df.columns if c not in ['user_input', 'response', 'reference', 'retrieved_contexts', 'reference_contexts']]

    friendly = {
        'faithfulness': '忠实度（不编造）',
        'response_relevancy': '回答相关性',
        'answer_relevancy': '回答相关性',
        'llm_context_precision_with_reference': '检索精度',
        'factual_correctness(mode=f1)': '事实正确性',
    }

    for col in metric_cols:
        val = df[col].mean()
        scores[col] = round(float(val), 4)
        display_name = friendly.get(col, col)
        bar = '█' * int(val * 20) + '░' * (20 - int(val * 20))
        print(f"  {display_name}: {val:.4f} [{bar}]")

    # 保存详细报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": sample_size,
        "judge_model": JUDGE_MODEL,
        "scores": scores,
        "samples_detail": [],
    }

    for i, sample in enumerate(samples):
        report["samples_detail"].append({
            "question": sample.user_input,
            "response": sample.response,
            "reference": sample.reference,
        })

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n详细报告已保存: {OUTPUT_PATH}")

    # 解读
    print("\n--- 结果解读 ---")
    faith = scores.get('faithfulness', 0)
    if faith >= 0.9:
        print(f"  忠实度 {faith:.2f}: 优秀，回答基本忠于检索内容，编造风险低")
    elif faith >= 0.7:
        print(f"  忠实度 {faith:.2f}: 良好，但仍有编造风险，需要关注低分样本")
    else:
        print(f"  忠实度 {faith:.2f}: 需改进，回答中存在较多编造内容，需优化prompt")

    relevancy = scores.get('response_relevancy', 0)
    if relevancy >= 0.8:
        print(f"  相关性 {relevancy:.2f}: 回答切题")
    else:
        print(f"  相关性 {relevancy:.2f}: 回答偏离问题，需优化")

    precision = scores.get('llm_context_precision_with_reference', 0)
    if precision >= 0.8:
        print(f"  检索精度 {precision:.2f}: 检索结果精准")
    else:
        print(f"  检索精度 {precision:.2f}: 检索噪音较多，可能需要调整检索策略")

    factual = scores.get('factual_correctness', 0)
    if factual >= 0.8:
        print(f"  事实正确性 {factual:.2f}: 事实准确")
    else:
        print(f"  事实正确性 {factual:.2f}: 存在事实错误，需优化知识库或prompt")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS评估工具")
    parser.add_argument("--sample_size", type=int, default=20, help="评估样本数（默认20）")
    args = parser.parse_args()

    if not API_KEY:
        print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    run_evaluation(sample_size=args.sample_size)
