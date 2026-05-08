"""
AI客服问答系统 - 向量化RAG
流程：
1. 知识库222个问题 → Embedding向量化 → ChromaDB存储
2. 客户提问 → Embedding向量 → ChromaDB检索Top-3
3. 检索结果 + 客户问题 → Qwen3-8B生成回答

注：当前知识库222条，Embedding一次检索即可精准命中Top-3，无需Rerank精排。
    当知识库扩展到数千条时，可加Rerank（lte-rerank-v2）做两阶段检索。
"""

import json
import requests
import chromadb
import os
import time

# ==================== 配置 ====================
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
EMBEDDING_MODEL = "text-embedding-v3"

QA_JSON_PATH = '/home/REMOVED_DB_USER/customer-service/05_analyze/reports/知识库_优化版.json'
CHROMA_PATH = '/home/REMOVED_DB_USER/customer-service/06_models/chroma_db'


def get_embeddings(texts, batch_size=6):
    """批量调用Embedding API，返回向量列表，失败时逐条重试"""
    all_embeddings = [None] * len(texts)

    # 先批量处理
    failed_indices = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = requests.post(
                EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                },
                timeout=60,
            )
            result = response.json()
            if "data" in result:
                data = sorted(result["data"], key=lambda x: x["index"])
                for j, d in enumerate(data):
                    all_embeddings[i + j] = d["embedding"]
            else:
                for j in range(len(batch)):
                    failed_indices.append(i + j)
        except Exception as e:
            for j in range(len(batch)):
                failed_indices.append(i + j)

        if i + batch_size < len(texts):
            time.sleep(0.3)

    # 逐条重试失败的
    for idx in failed_indices:
        try:
            response = requests.post(
                EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": [texts[idx]],
                },
                timeout=30,
            )
            result = response.json()
            if "data" in result:
                all_embeddings[idx] = result["data"][0]["embedding"]
                print(f"  重试成功: 第{idx+1}条")
            else:
                print(f"  重试失败: 第{idx+1}条 - {str(result)[:100]}")
        except Exception as e:
            print(f"  重试失败: 第{idx+1}条 - {e}")
        time.sleep(0.3)

    if None in all_embeddings:
        return None
    return all_embeddings


class ImprovedQASystem:
    """AI客服问答系统 - 向量化RAG"""

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

    def _call_llm(self, prompt):
        """调用大模型生成回答"""
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": API_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 600,
                    "enable_thinking": False,
                },
                timeout=60,
            )
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def get_answer(self, question):
        """获取回答：向量检索 + 大模型生成"""
        # 第一步：向量检索
        matched = self.search_knowledge(question, top_k=3)

        if not matched or matched[0]['score'] < 0.3:
            return "您好，这个问题需要咨询人工客服，请拨打4000040181"

        # 第二步：构建上下文（完整回答，不截断）
        context = ""
        for i, item in enumerate(matched, 1):
            qa = item['qa']
            context += f"【参考{i}】(相关度:{item['score']:.2f})\n问题: {qa['std_question']}\n回答: {qa['answer']}\n\n"

        # 第三步：大模型生成
        prompt = f"""你是南京宁惠保的AI客服，请【严格仅根据下方知识库内容】回答客户问题。

【禁止事项】
- 禁止编造知识库中没有的信息
- 禁止用自己的知识补充知识库没有的细节
- 禁止概括或改写知识库的具体流程、步骤、材料清单，必须照原样列出

**同义词对照**
- 门槛费 = 起付线 = 免赔额
- 材料 = 资料 = 文件
- 报销 = 理赔 = 赔付
- 取消保险 = 退保

{context}

客户问题: {question}

要求:
1. 只使用上方知识库中的内容回答，一字不差地保留流程步骤、材料清单、具体数字
2. 用"您好"开头
3. 知识库中有具体数字的，必须完整列出（免赔额金额、赔付比例、保费、保额等），不能省略
4. 如果有多个版本/责任的区别，逐条列出
5. 如果有多条参考内容，综合回答
6. 知识库没有的内容，回复"您好，这个问题需要咨询人工客服，请拨打4000040181"

直接回答:"""

        answer = self._call_llm(prompt)
        if answer.startswith("ERROR:"):
            return "您好，系统暂时无法回答，请拨打人工客服4000040181"
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
