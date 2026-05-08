# 南京宁惠保 AI 客服系统

基于 RAG（检索增强生成）的智能客服系统，为南京宁惠保保险产品提供自动问答服务。

## 系统架构

```
客户提问 → Embedding向量化 → ChromaDB检索Top-3 → Qwen3-8B生成回答
```

- **知识库**: 222条标准QA，覆盖7个分类
- **向量检索**: text-embedding-v3 + ChromaDB
- **回答生成**: Qwen3-8B (DashScope API)
- **评估得分**: 9.6/10 (Qwen-Max评判)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入真实密钥
```

必须配置：
- `DASHSCOPE_API_KEY` - 阿里云DashScope API密钥

### 3. 运行AI客服

```bash
cd scripts
python ai_qa_system_v2.py
```

首次运行会自动构建向量库（约30秒），之后直接加载。

### 4. 运行评估

```bash
python rag_eval.py --sample_size 20
```

### 5. 生成客服评审表

```bash
python generate_review_sheet.py
```

## 项目结构

```
├── scripts/                    # 核心脚本
│   ├── ai_qa_system_v2.py      # RAG问答系统（主入口）
│   ├── rag_eval.py             # 评估工具
│   ├── generate_review_sheet.py # 评审表生成
│   ├── analyze.py              # 对话意图分析
│   ├── analyze_complaint.py    # 投诉类分析
│   ├── analyze_time_complaint.py # 理赔时效分析
│   ├── analyze_consultation.py # 咨询类分析
│   ├── consolidate_knowledge.py # 知识库整合
│   └── knowledge_checker.py    # 知识库自检
├── 05_analyze/
│   └── reports/                # 知识库和分析报告
│       ├── 知识库_优化版.json    # 知识库(222条)
│       └── 客服评审表.xlsx       # 评审表
├── docs/                       # 文档和参考数据
├── requirements.txt
└── .env.example
```

## 知识库分类

| 分类 | 数量 |
|------|------|
| 产品信息 | 75 |
| 理赔流程 | 55 |
| 保障范围 | 53 |
| 条款解释 | 16 |
| 其他问题 | 13 |
| 理赔材料 | 6 |
| 退保流程 | 4 |

## 技术要点

- **防幻觉**: Prompt中严格限制只使用知识库内容，禁止编造
- **同义词映射**: 门槛费=起付线=免赔额 等，提升检索命中率
- **向量库自动重建**: 首次运行或数据变更时自动构建
