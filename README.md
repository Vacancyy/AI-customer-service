# 南京宁惠保 AI 客服系统

基于 RAG（检索增强生成）的智能客服系统，为南京宁惠保保险产品提供自动问答服务。

## 系统架构

```
客户提问 → 本地Embedding向量化 → ChromaDB检索Top-3 → Qwen3-32B生成回答
```

- **知识库**: 223条标准QA，覆盖7个分类
- **向量检索**: 本地bge-large-zh-v1.5 + ChromaDB（无需API Key）
- **回答生成**: Qwen3-32B (DashScope API)
- **多轮对话**: 支持上下文记忆，连续提问
- **置信度分层**: ≥0.7直接回答，0.5-0.7答+转人工提示，<0.5转人工

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 设置环境变量（必须）
export DASHSCOPE_API_KEY="your-dashscope-api-key"  # 阿里云DashScope API密钥

# 数据库配置（可选，用于日志记录）
export DB_HOST="localhost"
export DB_PORT="3308"
export DB_USER="root"
export DB_PASS="your-password"
export DB_NAME="ai_customer_service"
```

或在 `.env` 文件中配置：
```
DASHSCOPE_API_KEY=your-dashscope-api-key
DB_HOST=localhost
DB_PORT=3308
DB_USER=root
DB_PASS=your-password
DB_NAME=ai_customer_service
```

### 3. 运行Web服务

```bash
python scripts/web_demo.py
```

访问 http://localhost:8000 即可使用。

### 4. 运行评估

```bash
python scripts/ragas_eval.py
python scripts/deepeval_eval.py
```

### 5. 知识库更新

```bash
python scripts/update_knowledge.py add --question "新问题" --answer "新答案"
python scripts/update_knowledge.py validate
python scripts/update_knowledge.py rebuild
```

## 项目结构

```
├── scripts/                    # 核心脚本
│   ├── ai_qa_system_v2.py      # RAG问答系统（主入口）
│   ├── web_demo.py             # Web界面服务
│   ├── update_knowledge.py     # 知识库更新工具
│   ├── ragas_eval.py           # RAGAS评估
│   ├── deepeval_eval.py        # DeepEval评估
│   ├── analyze.py              # 对话意图分析
│   ├── analyze_complaint.py    # 投诉类分析
│   ├── analyze_time_complaint.py # 理赔时效分析
│   ├── analyze_consultation.py # 咨询类分析
│   ├── consolidate_knowledge.py # 知识库整合
│   └── knowledge_checker.py    # 知识库自检
├── 05_analyze/
│   └── reports/                # 知识库和分析报告
│       ├── 知识库_优化版.json    # 知识库(223条)
│       ├── RAGAS评估报告.md
│       ├── AI客服综合评估报告.md
│       └── 客服评审表.xlsx       # 评审表
├── 06_models/
│   └── embedding_model/        # 本地Embedding模型
│       └── AI-ModelScope/bge-large-zh-v1.5/
├── docs/                       # 文档和参考数据
├── requirements.txt
└── README.md
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

**总计**: 223条

## 主要功能

- **多轮对话**: 支持上下文记忆，用户可连续追问
- **置信度分层**: 根据检索得分自动判断回答策略
- **PII防护**: 输入/输出/存储三层隐私信息脱敏
- **免责声明**: 涉及金额时自动追加免责提示
- **日志记录**: MySQL存储每次问答日志
- **知识库维护**: 提供add/validate/rebuild工具

## 安全说明

⚠️ **敏感信息已移除**: 所有API密钥和数据库密码从代码中移除，使用前必须配置环境变量。

请勿在代码中硬编码任何密钥或密码。
