"""
AI客服问答系统脚本

功能：
1. 加载QA知识库
2. 接入大模型(Qwen3-8B)
3. 模拟客户提问，返回正确回答
4. 支持多种匹配方式：关键词匹配、相似度匹配、大模型增强

使用方式：
  python scripts/ai_qa_system.py --test              # 测试模式，预设问题
  python scripts/ai_qa_system.py --interactive        # 交互模式，手动输入问题
  python scripts/ai_qa_system.py --batch questions.txt # 批量测试
"""

import os
import sys
import json
import requests
import argparse
import pymysql
import pandas as pd
from collections import defaultdict

# ==================== 配置 ====================

# API配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"

# 数据库配置
DB_CONFIG = {
    'host': 'REMOVED_DB_HOST',
    'port': 3308,
    'user': 'REMOVED_DB_USER',
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': 'ai_customer_service',
    'charset': 'utf8mb4'
}

# 知识库文件路径
QA_JSON_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_优化版.json'

# ==================== 知识库加载 ====================

class QAKnowledgeBase:
    """QA知识库类"""

    def __init__(self, json_path=None, use_db=False):
        self.qa_data = []
        self.category_index = defaultdict(list)  # 按分类索引
        self.keyword_index = defaultdict(list)   # 按关键词索引

        if json_path:
            self.load_from_json(json_path)
        if use_db:
            self.load_from_db()

        self.build_indexes()

    def load_from_json(self, json_path):
        """从JSON文件加载知识库"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
        print(f"从JSON加载知识库: {len(self.qa_data)} 条")

    def load_from_db(self):
        """从数据库加载知识库"""
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                    SELECT primary_category, secondary_category, std_question,
                           keywords, answer, priority, source
                    FROM qa_knowledge
                """)
                for row in cursor.fetchall():
                    self.qa_data.append({
                        'primary_category': row['primary_category'],
                        'secondary_category': row['secondary_category'],
                        'std_question': row['std_question'],
                        'keywords': row['keywords'] or '',
                        'answer': row['answer'],
                        'priority': row['priority'] or 2,
                        'source': row['source']
                    })
        finally:
            conn.close()
        print(f"从数据库加载知识库: {len(self.qa_data)} 条")

    def build_indexes(self):
        """构建索引加速检索"""
        # 按分类索引
        for qa in self.qa_data:
            cat_key = f"{qa['primary_category']}|{qa['secondary_category']}"
            self.category_index[cat_key].append(qa)

        # 按关键词索引
        for qa in self.qa_data:
            keywords = qa['keywords'].split(',') if qa['keywords'] else []
            for kw in keywords:
                if kw.strip():
                    self.keyword_index[kw.strip()].append(qa)

    def search_by_keywords(self, question, top_k=5):
        """关键词匹配检索 - 改进版，支持特殊问题优先匹配"""

        # 特殊问题优先匹配规则（这些问题优先返回特定答案）
        special_rules = {
            # 免赔额金额问题
            '免赔额是多少': ['免赔额具体金额', '免赔额是多少'],
            '免赔额金额': ['免赔额具体金额'],
            # 产品介绍问题（使用关键词组合）
            '产品是什么': ['什么是', '宁惠保'],
            '这款产品是什么': ['什么是', '宁惠保'],
            '什么是宁惠保': ['什么是', '宁惠保'],
            '南京宁惠保是什么': ['什么是', '宁惠保'],
            '宁惠保': ['什么是', '宁惠保'],
            # 客服电话
            '客服电话': ['人工客服电话'],
            '客服电话是多少': ['人工客服电话'],
            # 理赔流程
            '怎么申请理赔': ['理赔流程是什么', '如何申请理赔'],
            '理赔流程': ['理赔流程是什么'],
            # 理赔材料
            '理赔需要什么材料': ['理赔申请需要哪些材料', '理赔需要提交哪些材料'],
            '理赔材料': ['理赔申请需要哪些材料'],
        }

        # 检查是否命中特殊规则
        question_clean = question.replace('？', '').replace('吗', '').replace('呢', '').strip()
        for key, targets in special_rules.items():
            if key in question or key in question_clean:
                # 特殊规则匹配：targets中任意一个目标匹配即可
                for qa in self.qa_data:
                    for target in targets:
                        if target in qa['std_question']:
                            return [qa], [(qa, 100)]  # 返回高分数表示精确匹配

        # 高权重关键词
        high_weight_keywords = [
            '免赔额', '既往症', '南京宁惠保', '宁惠保', '客服电话',
            '理赔流程', '理赔材料', '保障范围', '保费', '投保',
        ]

        # 一般关键词
        normal_keywords = [
            '理赔', '保单', '保险', '报销', '材料', '审核',
            '打款', '到账', '投保', '退保', '续保', '保障',
            '条款', '费用', '金额', '时效', '流程',
            '医保', '门诊', '住院', '赔付', '发票', '清单',
            '犹豫期', '客服', '电话', '参保', '咨询',
        ]

        # 统计每个QA的匹配得分
        scores = []
        for qa in self.qa_data:
            score = 0
            std_q = qa['std_question']
            keywords = qa.get('keywords', '')

            # 高权重关键词匹配
            for kw in high_weight_keywords:
                if kw in question:
                    if kw in std_q:
                        score += 10
                    elif kw in keywords:
                        score += 5

            # 一般关键词匹配
            for kw in normal_keywords:
                if kw in question:
                    if kw in std_q:
                        score += 3
                    elif kw in keywords:
                        score += 2

            # 问题文本相似度得分
            q_words = set(question.replace('？', '').replace('吗', '').replace('呢', '').split())
            std_words = set(std_q.replace('？', '').replace('吗', '').replace('呢', '').split())
            overlap = len(q_words & std_words)
            score += overlap * 2

            # 优先级加分
            score += qa['priority'] * 0.5

            if score > 0:
                scores.append((qa, score))

        # 按得分排序返回top_k
        scores.sort(key=lambda x: -x[1])
        return [qa for qa, score in scores[:top_k]], scores[:top_k]

    def search_by_category(self, question):
        """根据问题推断分类并检索"""
        # 分类关键词映射
        category_keywords = {
            '产品信息|产品介绍': ['什么是', '产品', '承保', '保险公司'],
            '产品信息|投保条件': ['投保', '购买', '年龄', '职业', '条件', '谁可以'],
            '产品信息|费率价格': ['多少钱', '保费', '价格', '费率', '赔付比例'],
            '产品信息|续保规则': ['续保', '自动续保'],
            '保障范围|保障内容': ['保障', '保什么', '保障范围', '报销', '住院', '门诊'],
            '保障范围|既往症保障': ['既往症', '既往病史'],
            '保障范围|除外责任': ['不赔', '除外', '不保障'],
            '理赔流程|理赔步骤': ['理赔', '申请理赔', '怎么理赔', '理赔流程'],
            '理赔流程|理赔时效': ['多久', '时间', '时效', '理赔多久'],
            '理赔流程|理赔条件': ['条件', '门槛', '免赔额', '多少费用'],
            '理赔材料|所需材料': ['材料', '发票', '病历', '清单'],
            '退保流程|退保流程': ['退保', '取消', '犹豫期'],
            '条款解释|条款含义': ['什么是', '意思', '定义', '名词'],
        }

        # 推断最可能的分类
        best_category = None
        best_score = 0
        for cat, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in question)
            if score > best_score:
                best_score = score
                best_category = cat

        # 返回该分类下的QA
        if best_category and best_category in self.category_index:
            return self.category_index[best_category], best_category

        return self.qa_data, None


# ==================== 大模型API ====================

class LLMClient:
    """大模型API客户端"""

    def __init__(self, api_url, api_key, model):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.total_tokens = 0

    def call(self, prompt, max_tokens=512, temperature=0.1):
        """调用API"""
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "enable_thinking": False,  # 关闭思考模式
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # 记录token使用
            usage = result.get("usage", {})
            self.total_tokens += usage.get("total_tokens", 0)

            return content
        except Exception as e:
            return f"API调用失败: {str(e)}"

    def generate_answer(self, question, knowledge_context):
        """基于知识库生成回答"""
        prompt = f"""你是南京宁惠保客服系统的AI助手。请根据以下知识库内容回答客户问题。

【知识库参考内容】
{knowledge_context}

【客户问题】
{question}

【回答要求】
1. 如果知识库中有完全匹配的问题，直接返回标准回答
2. 如果知识库有相关内容，基于知识库回答，不要添加知识库外的内容
3. 回答要简洁清晰，用"您好"开头
4. 如果知识库中没有相关内容，回复"您好，这个问题需要咨询人工客服，请拨打4000040181"

请直接输出回答内容："""

        return self.call(prompt, max_tokens=300)


# ==================== QA问答系统 ====================

class AIQASystem:
    """AI客服问答系统"""

    def __init__(self):
        self.knowledge_base = QAKnowledgeBase(json_path=QA_JSON_PATH)
        self.llm_client = LLMClient(API_URL, API_KEY, API_MODEL)

    def get_answer(self, question, use_llm=True):
        """获取问题回答"""
        print(f"\n{'='*60}")
        print(f"客户提问: {question}")
        print(f"{'='*60}")

        # 步骤1: 关键词匹配检索
        matched_qas, scores = self.knowledge_base.search_by_keywords(question, top_k=3)

        if not matched_qas:
            print("未找到匹配的知识库内容")
            if use_llm:
                return self.llm_client.generate_answer(question, "知识库中暂无相关内容")
            return "您好，这个问题需要咨询人工客服，请拨打4000040181"

        # 显示匹配结果
        print(f"\n匹配结果 (共{len(matched_qas)}条):")
        for i, (qa, score) in enumerate(scores[:3], 1):
            print(f"  {i}. [{score}分] {qa['std_question'][:50]}...")
            print(f"     分类: {qa['primary_category']}|{qa['secondary_category']}")

        # 步骤2: 获取最佳匹配
        best_match = matched_qas[0]
        best_score = scores[0][1]

        # 步骤3: 决定回答方式
        if best_score >= 5:  # 高匹配度，直接返回标准回答
            print(f"\n匹配度高({best_score}分)，直接返回标准回答")
            answer = best_match['answer']
            print(f"\nAI回复: {answer[:200]}...")
            return answer

        elif use_llm:  # 中等匹配度，使用大模型增强
            print(f"\n匹配度中等({best_score}分)，使用大模型增强回答")

            # 构建知识库上下文
            context = ""
            for qa in matched_qas[:2]:
                context += f"问题: {qa['std_question']}\n回答: {qa['answer'][:300]}\n\n"

            answer = self.llm_client.generate_answer(question, context)
            print(f"\nAI回复: {answer}")
            return answer

        else:
            # 不使用大模型，返回最佳匹配的回答
            print(f"\n返回最佳匹配回答")
            return best_match['answer']

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*60)
        print("AI客服问答系统 - 交互模式")
        print("="*60)
        print("输入问题进行测试，输入 'quit' 或 'exit' 退出")
        print("输入 'stats' 查看统计信息")
        print("="*60 + "\n")

        while True:
            try:
                question = input("\n请输入问题: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("退出系统")
                    break

                if question.lower() == 'stats':
                    self.show_stats()
                    continue

                if not question:
                    continue

                answer = self.get_answer(question)

            except KeyboardInterrupt:
                print("\n退出系统")
                break

    def test_mode(self):
        """测试模式 - 使用预设问题"""
        test_questions = [
            "理赔需要什么材料？",
            "免赔额是多少？",
            "这款产品保障什么？",
            "怎么申请理赔？",
            "既往症能理赔吗？",
            "保费多少钱？",
            "门诊费用能报销吗？",
            "住院费用怎么理赔？",
            "这款产品是什么？",
            "客服电话是多少？",
        ]

        print("\n" + "="*60)
        print("AI客服问答系统 - 测试模式")
        print("="*60)

        results = []
        for q in test_questions:
            answer = self.get_answer(q)
            results.append({
                'question': q,
                'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            })

        # 输出测试结果汇总
        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        for r in results:
            print(f"\n问: {r['question']}")
            print(f"答: {r['answer']}")

        print(f"\n总Token消耗: {self.llm_client.total_tokens}")

        return results

    def show_stats(self):
        """显示统计信息"""
        print("\n知识库统计:")
        print(f"  总条数: {len(self.knowledge_base.qa_data)}")

        # 分类统计
        cat_count = defaultdict(int)
        for qa in self.knowledge_base.qa_data:
            cat_count[qa['primary_category']] += 1

        print("\n分类分布:")
        for cat, cnt in sorted(cat_count.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt}条")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="AI客服问答系统")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--batch", type=str, help="批量测试文件路径")
    parser.add_argument("--no-llm", action="store_true", help="不使用大模型增强")
    args = parser.parse_args()

    # 创建系统
    system = AIQASystem()

    # 运行模式
    use_llm = not args.no_llm

    if args.interactive:
        system.interactive_mode()
    elif args.test:
        system.test_mode()
    elif args.batch:
        # 批量测试
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        print(f"批量测试 {len(questions)} 个问题")
        for q in questions:
            system.get_answer(q, use_llm=use_llm)
    else:
        # 默认测试模式
        system.test_mode()


if __name__ == "__main__":
    main()