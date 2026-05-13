"""
AI客服问答系统 - 向量化RAG
流程：
1. 知识库223个问题 → 本地Embedding向量化 → ChromaDB存储
2. 客户提问 → 本地Embedding向量 → ChromaDB检索Top-3
3. 检索结果 + 客户问题 → Qwen3-8B生成回答

注：Embedding使用本地模型(bge-large-zh-v1.5)，不依赖DashScope API。
    LLM调用仍需DashScope API Key（或应用级Token）。
"""

import json
import re
import requests
import chromadb
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ==================== 配置 ====================
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "REMOVED_API_KEY")
API_MODEL = "qwen3-32b"

# 本地Embedding模型路径（已通过ModelScope下载到本地）
LOCAL_EMBEDDING_MODEL = os.path.join(
    os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '06_models/embedding_model/AI-ModelScope/bge-large-zh-v1.5'
)

# 项目根目录：优先用环境变量，否则基于脚本位置推断
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QA_JSON_PATH = os.path.join(PROJECT_ROOT, '05_analyze/reports/知识库_优化版.json')
CHROMA_PATH = os.path.join(PROJECT_ROOT, '06_models/chroma_db')

# 全局Embedding模型实例（懒加载）
_embedding_model = None


def get_embedding_model():
    """懒加载本地Embedding模型"""
    global _embedding_model
    if _embedding_model is None:
        print(f"  加载本地Embedding模型: {LOCAL_EMBEDDING_MODEL}...")
        _embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"  模型加载完成，向量维度: {_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def get_embeddings(texts):
    """使用本地模型批量生成Embedding向量"""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 10)
    return embeddings.tolist()


def filter_pii(text):
    """脱敏用户输入中的隐私信息"""
    text = re.sub(r'\d{17}[\dXx]', '[身份证号已脱敏]', text)
    text = re.sub(r'1[3-9]\d{9}', '[手机号已脱敏]', text)
    return text


def scan_output_pii(text):
    """扫描AI输出中的隐私信息并脱敏"""
    text = re.sub(r'\d{17}[\dXx]', '[身份证号已脱敏]', text)
    text = re.sub(r'1[3-9]\d{9}', '[手机号已脱敏]', text)
    return text


def add_disclaimer(answer):
    """涉及赔付金额/比例时追加免责声明"""
    # 检测回答中是否包含金额、比例等关键数字
    if re.search(r'(免赔额|赔付比例|保费|保额|报销|理赔金额|赔付).*\d+', answer):
        if '仅供参考' not in answer:
            answer += '\n\n（以上信息仅供参考，具体以保险条款约定为准）'
    return answer


class ImprovedQASystem:
    """AI客服问答系统 - 向量化RAG"""

    # 数据库配置
    DB_HOST = "REMOVED_DB_HOST"
    DB_PORT = 3308
    DB_USER = "REMOVED_DB_USER"
    DB_PASS = "REMOVED_DB_PASSWORD"
    DB_NAME = "ai_customer_service"

    def __init__(self, rebuild=False):
        # 加载知识库
        with open(QA_JSON_PATH, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
        print(f"加载知识库: {len(self.qa_data)} 条")

        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        if rebuild:
            try:
                self.chroma_client.delete_collection("qa_knowledge")
            except:
                pass
            self._build_vector_db()
        else:
            try:
                self.collection = self.chroma_client.get_collection("qa_knowledge")
                count = self.collection.count()
                if count != len(self.qa_data):
                    print(f"向量库条数({count})与知识库({len(self.qa_data)})不一致，重建...")
                    self.chroma_client.delete_collection("qa_knowledge")
                    self._build_vector_db()
                else:
                    print(f"加载向量库: {count} 条")
            except Exception as e:
                print(f"向量库不存在，开始构建...")
                self._build_vector_db()

    def _build_vector_db(self):
        """构建向量数据库"""
        print("正在生成向量索引（首次运行约需1-2分钟）...")

        texts = []
        ids = []
        metadatas = []
        for i, qa in enumerate(self.qa_data):
            text = f"问题: {qa['std_question']} 回答: {qa['answer']}"
            texts.append(text)
            ids.append(str(i+1))
            metadatas.append({
                "std_question": qa['std_question'],
                "primary_category": qa.get('primary_category', ''),
            })

        print(f"  正在向量化 {len(texts)} 条数据...")
        embeddings = get_embeddings(texts)
        if embeddings is None:
            raise Exception("向量化失败，请检查API")

        self.collection = self.chroma_client.create_collection(
            name="qa_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        print(f"  向量库构建完成: {self.collection.count()} 条")

    def search_knowledge(self, question, top_k=3):
        """向量检索：客户问题 → 向量 → 余弦相似度 → 最相关的知识"""
        query_embedding = get_embeddings([question])
        if query_embedding is None:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        matched = []
        for i in range(len(results['ids'][0])):
            idx = int(results['ids'][0][i]) - 1
            score = 1 - results['distances'][0][i]  # cosine距离 → 相似度
            matched.append({
                'qa': self.qa_data[idx],
                'score': score,
                'index': idx,
            })

        return matched

    def _call_llm(self, prompt, history=None):
        """调用大模型生成回答（支持多轮对话）"""
        try:
            # 构建messages：先加入历史对话，再加入当前prompt
            messages = []
            if history:
                for h in history[-6:]:  # 最多保留最近3轮（每轮2条）
                    messages.append({"role": "user", "content": h["question"]})
                    messages.append({"role": "assistant", "content": h["answer"]})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": API_MODEL,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 400,
                    "enable_thinking": False,
                },
                timeout=60,
            )
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _save_log(self, question, answer, best_score, matched, confidence_level, response_time):
        """保存对话日志到MySQL"""
        try:
            import pymysql
            conn = pymysql.connect(
                host=self.DB_HOST, port=self.DB_PORT,
                user=self.DB_USER, password=self.DB_PASS,
                database=self.DB_NAME
            )
            cur = conn.cursor()
            top1_question = matched[0]['qa']['std_question'] if matched else ''
            cur.execute(
                "INSERT INTO ai_chat_log (question, answer, top1_score, top1_question, confidence_level, model_name, response_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (question, answer, best_score, top1_question, confidence_level, API_MODEL, response_time)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # 日志写入失败不影响主流程

    def _build_context(self, matched):
        """构建上下文（完整回答，不截断）"""
        context = ""
        for i, item in enumerate(matched, 1):
            qa = item['qa']
            context += f"【参考{i}】(相关度:{item['score']:.2f})\n问题: {qa['std_question']}\n回答: {qa['answer']}\n\n"
        return context

    def _build_prompt(self, question, context):
        """构建大模型prompt"""
        prompt = f"""你是南京宁惠保的AI客服，请【严格仅根据下方知识库内容】回答客户问题。

【禁止事项】
- 禁止编造知识库中没有的信息
- 禁止用自己的知识补充知识库没有的细节
- 禁止概括或改写知识库的具体流程、步骤、材料清单
- 禁止引用知识库中带有特定客户信息的内容

【回答原则】
- 只回答用户问的问题，不要主动扩展到用户未询问的话题
- 如果用户问简单问题（如"犹豫期多久"），给出简洁回答，不要展开
- 如果检索到多条相关信息，优先引用最直接回答用户问题的一条
- 控制回答在3-5句话以内，避免过度展开
- 如果知识库中有基础版和升级版两套数据，必须完整列出两版的数字
- 如果知识库中有既往症和非既往症的区分，必须说明两者的差异

【回答风格示例】
❌ 用户问"犹豫期多久"，AI回答"这款产品无犹豫期，等待期是...，保障期是..."
✓ 用户问"犹豫期多久"，AI回答"您好，这款产品无犹豫期"

❌ 用户问"保费多少钱"，AI回答"99元。此外宁惠保还提供双通道门诊用药报销..."
✓ 用户问"保费多少钱"，AI回答"您好，基础版保费99元，升级版保费150元"

**同义词对照**
- 门槛费 = 起付线 = 免赔额
- 材料 = 资料 = 文件
- 报销 = 理赔 = 赔付
- 取消保险 = 退保

{context}

客户问题: {question}

要求:
1. 只使用上方知识库中的内容回答
2. 用"您好"开头
3. 有具体数字的必须完整列出（免赔额金额、赔付比例、保费等）
4. 有多个版本/责任的区别时，用①②③逐条列出
5. 知识库没有的内容，回复"您好，这个问题需要咨询人工客服，请拨打4000040181"

直接回答:"""
        return prompt

    def get_answer(self, question, history=None):
        """获取回答：向量检索 + 置信度分层 + 大模型生成（支持多轮对话）"""
        import time
        start_time = time.time()

        # PII输入脱敏
        original_question = question
        question = filter_pii(question)

        # 第一步：向量检索
        matched = self.search_knowledge(question, top_k=3)

        # 低置信度：直接转人工
        if not matched or matched[0]['score'] < 0.5:
            answer = "您好，这个问题需要咨询人工客服获取准确解答。\n\n请拨打4000040181。"
            confidence_level = 'low'
            best_score = matched[0]['score'] if matched else 0.0
            elapsed = time.time() - start_time
            self._save_log(question, answer, best_score, matched, confidence_level, elapsed)
            return answer

        best_score = matched[0]['score']

        # 构建上下文和prompt
        context = self._build_context(matched)
        prompt = self._build_prompt(question, context)
        answer = self._call_llm(prompt, history=history)
        elapsed = time.time() - start_time

        if answer.startswith("ERROR:"):
            answer = "您好，系统暂时无法回答，请拨打人工客服4000040181。"
            self._save_log(question, answer, best_score, matched, 'error', elapsed)
            return answer

        # PII输出脱敏
        answer = scan_output_pii(answer)

        # 免责声明
        answer = add_disclaimer(answer)

        # 高置信度（≥0.7）：直接回答
        if best_score >= 0.7:
            confidence_level = 'high'
            self._save_log(question, answer, best_score, matched, confidence_level, elapsed)
            return answer

        # 中置信度（0.5-0.7）：回答 + 补充转人工提示
        confidence_level = 'medium'
        if "4000040181" not in answer:
            answer += "\n\n如需进一步帮助，请咨询人工客服：4000040181。"
        self._save_log(question, answer, best_score, matched, confidence_level, elapsed)
        return answer


if __name__ == "__main__":
    import sys
    rebuild = '--rebuild' in sys.argv
    system = ImprovedQASystem(rebuild=rebuild)
    print("\n南京宁惠保智能客服（向量化RAG）")
    print("输入问题获取回答，输入 q 退出，输入 --rebuild 重建向量库")
    print("-" * 50)
    while True:
        try:
            question = input("\n你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ('q', 'quit', 'exit'):
            print("再见！")
            break
        answer = system.get_answer(question)
        print(f"\n客服回复: {answer}")
