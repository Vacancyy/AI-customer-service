# 客服对话分析项目

## 项目目标
处理客服数据，生成结构化 CSV 文件：
1. **电话录音**：256,186 条 → ASR 转录 + 对话分类
2. **在线对话**：183,207 条会话 → 直接提取对话内容

---

## 目录结构（2026年4月9日更新）

```
/home/REMOVED_DB_USER/customer-service/
│
├── 01_source/                   # 数据源阶段
│   └── heli.sqlite3             # 原始数据库
│
├── 02_download/                 # 下载阶段
│   └── audio/                   # 音频文件 (mp3) - 103,012 个
│
├── 03_process/                  # 处理阶段 (ASR+分离)
│   ├── cache/                   # 说话者分离缓存
│   └── temp/                    # 批处理临时目录
│
├── 04_output/                   # 输出阶段
│   ├── dialog_csv/              # 电话录音转录结果 - 38,123 个（处理中）
│   └── online_chat_csv/         # 在线对话数据 - 106,023 个
│
├── 05_analyze/                  # 分析阶段
│   ├── reports/                 # PDF分析报告
│   └── database/                # MySQL存储脚本
│
├── 06_models/                   # 模型目录
│   ├── asr/Qwen3-ASR-1.7B/      # ASR 模型
│   ├── aligner/Qwen3-ForcedAligner-0.6B/  # 时间对齐模型
│   └── local/                   # 其他本地模型
│
├── scripts/                     # 所有脚本
│   ├── config.py                # 路径配置（统一管理）
│   ├── tool_download_heli_audio.py  # 下载音频
│   ├── process_audio.py         # 音频处理
│   ├── batch_process_audio.py   # 批量处理
│   ├── process_online_chat.py   # 在线对话处理
│   ├── analyze.py               # 分析脚本
│   ├── import_to_db.py          # 数据导入数据库
│   └── diarize_worker.py        # 分离子进程
│
└── docs/                        # 文档
    ├── context.md
    ├── CHANGELOG.md
    └── PARAMETERS.md
```

---

## 最新进度（2026年4月13日）

### 数据处理进度

| 数据类型 | 数据量 | 处理状态 | 输出目录 |
|---------|--------|---------|---------|
| 电话录音音频 | **103,012 个已下载** | ✅ 完成 | `02_download/audio/` |
| 电话录音 CSV | **38,123 个** | 🔄 处理中 | `04_output/dialog_csv/` |
| 在线对话 CSV | **106,023 个** | ✅ 完成 | `04_output/online_chat_csv/` |
| 数据库导入 | 100 条测试 | ✅ 可用 | MySQL `dialog_analysis` 表 |

### 待处理任务

| 任务 | 数量 | 当前状态 |
|------|------|----------|
| 音频 ASR 转录 | ~64,889 个待处理 | 🔄 运行中（复用缓存模式） |
| 全量数据导入数据库 | ~11万条 | ⏳ 待 ASR 完成后执行 |

### 当前运行的进程

**运行指令**：
```bash
python scripts/process_audio.py ./02_download/audio 20 --reuse-diarization
```

**进程状态**（截至 2026-04-13）：
- 启动时间：09:48
- 已运行：约 5 小时
- 阶段：ASR 转录（阶段2）
- 缓存数据：已加载 101,343 个文件的分离数据
- GPU 使用：VLLM 22.3 GB + 用户进程 2.2 GB

### 分析报告功能（新增）

**脚本文件**: `analyze.py`

**功能**：
1. 意图分类：分析客户来电/咨询原因
2. 情感分析：识别客户情绪（满意/中立/不满）
3. 高频问题提取：统计客户最关心的问题 TOP N
4. PDF报告生成：包含可视化图表

**使用方式**：
```bash
# 分析电话录音
python analyze.py --dir dialog_csv --limit 200

# 分析在线对话
python analyze.py --dir online_chat_csv --limit 200

# 分析全部数据
python analyze.py --dir all
```

**输出报告**：
- 文件位置：`analysis_result/` 目录
- 文件名：根据选择的目录自动命名（电话录音分析报告.pdf / 在线对话分析报告.pdf）
- 页数：约9页
- 包含图表：7个可视化图表（饼图+柱状图）

### 报告内容结构

1. **数据概览**：基础统计、来源分布
2. **意图分类分析**：一级/二级分类分布
3. **情感分析**：客户情感分布、不满客户问题分析
4. **高频问题分析**：TOP 15问题、具体案例、类型细分
5. **核心观点与建议**：数据驱动的洞察和优化建议
6. **风险与机会分析**：风险点、改进机会
7. **监测指标建议**：舆情监测、服务质量指标
8. **总结与行动清单**：核心发现、立即行动项、预期效果

### 可视化图表（7个）

1. 关键指标对比图（柱状图）
2. 一级分类分布图（饼图）
3. 一级分类数量对比图（柱状图）
4. 二级分类 TOP 10 柱状图
5. 客户情感分布图（饼图）
6. 不满客户问题类型分布图（饼图）
7. 高频问题 TOP 10 分布图（柱状图）
8. 问题类型细分分布图（饼图）

### 意图分类体系（Plan A：按客户行为分类）

**设计原则**：一级分类统一为"客户行为"维度，避免业务领域混入导致重叠。

#### 一级分类：4个 + 其他

| 一级分类 | 定义 | 客户特征 |
|----------|------|----------|
| **咨询** | 了解信息，问"是什么、怎么做" | 初次了解，需要解释 |
| **查询** | 查询状态，问"到哪了、什么时候" | 已有业务，关注进度 |
| **办理** | 办理业务，要"提交、申请、操作" | 有明确办理需求 |
| **投诉** | 表达不满，有"抱怨、质疑" | 情绪负面，需安抚 |
| **其他** | 无法归类 | 回访、信息核实等 |

#### 二级分类：共22个

| 一级分类 | 二级分类 | 说明 | 示例对话 |
|----------|----------|------|----------|
| **咨询** | 产品了解 | 问产品保障内容 | "这个保险保什么？" |
| | 条款解释 | 问条款含义 | "既往症是什么意思？" |
| | 费率查询 | 问保费价格 | "一年多少钱？" |
| | 理赔流程了解 | 问理赔步骤 | "理赔怎么申请？" |
| | 理赔材料了解 | 问理赔材料 | "理赔需要什么材料？" |
| | 退保流程了解 | 问退保流程 | "怎么退保？" |
| | 保障范围了解 | 问保障范围 | "这个病能赔吗？" |
| | 其他了解 | 其他咨询 | 其他问题 |
| **查询** | 理赔进度查询 | 查理赔进度 | "理赔审核到哪了？" |
| | 理赔到账查询 | 查理赔到账 | "理赔款什么时候到？" |
| | 保单状态查询 | 查保单状态 | "我的保单有效吗？" |
| | 其他查询 | 其他查询 | 其他状态查询 |
| **办理** | 理赔申请 | 提交理赔 | "我要申请理赔" |
| | 事故报案 | 报告事故 | "我要报案" |
| | 新保投保 | 新买保险 | "我要买这个保险" |
| | 续保办理 | 续保操作 | "我要续保" |
| | 保单变更 | 变更信息 | "改一下联系地址" |
| | 退保办理 | 办理退保 | "帮我退保" |
| | 减保办理 | 减少保额 | "我要减保" |
| **投诉** | 理赔时效投诉 | 投诉理赔慢 | "理赔太慢了！" |
| | 拒赔投诉 | 投诉拒赔 | "为什么不赔？" |
| | 理赔金额异议 | 对金额不满 | "赔得太少了" |
| | 服务态度投诉 | 投诉服务 | "客服态度不好" |
| | 其他投诉 | 其他投诉 | 其他不满 |
| **其他** | 回访确认 | 回访确认 | 回访电话 |
| | 信息核实 | 核实信息 | 确认身份信息 |

#### 分类优势

- ✅ **无重叠**：一级分类维度统一（客户行为）
- ✅ **无歧义**：二级分类明确描述具体行为
- ✅ **易统计**：直接看出客户行为分布（咨询为主还是办理为主）

---

## 下载逻辑（已优化）

### 下载策略
| 数据时期 | 下载方式 | 说明 |
|---------|---------|------|
| 30天内 | 原地址 `a6alipbxsh16.7x24cc.com` | 直接下载 |
| 30天前 | 替换域名 `storage.7x24cc.com` | 存储服务器 |

### URL 替换示例
```
原始: https://a6alipbxsh16.7x24cc.com/monitor/1.102.16.105/20240929/xxx.mp3
替换: https://storage.7x24cc.com/storage-server/presigned/ss1/a6-online-ass-recorder/monitor/1.102.16.105/20240929/xxx.mp3
```

---

## 处理流程

### 电话录音处理（需要大模型）
```
音频文件 → 说话者分离 → ASR转录 → 对话分类（API） → CSV输出
```

### 在线对话处理（纯脚本，无需大模型）
```
数据库 Session/Message 表 → 规则匹配发送者身份 → CSV输出
```

### 各阶段耗时分析

| 阶段 | 耗时占比 | 运行位置 | 瓶颈原因 |
|------|---------|---------|---------|
| 说话者分离 | ~40% | 本地 GPU | **单线程，主要瓶颈** |
| ASR 转录 | ~40% | 本地 GPU | 批处理优化 |
| API 分类 | ~20% | 在线 API | 已多线程优化 |

---

## 数据库信息

### SQLite 原始数据库

- **数据库路径**: `/home/REMOVED_DB_USER/customer-service/01_source/heli.sqlite3`
- **总记录**: 256,186 条
- **时间范围**: 2024-09-29 ~ 2026-03-13

### MySQL 分析结果数据库

- **主机**: `REMOVED_DB_HOST:3308`
- **数据库名**: `ai_customer_service`
- **表名**: `dialog_analysis`

### 数据库表结构

#### Audio 表（heli_audio）- 通话记录

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 主键 |
| `CallTimeLength` | int | 通话时长（秒）|
| `CallNo` | varchar(32) | 主叫号码 |
| `CalledNo` | varchar(32) | 被叫号码 |
| `QueueTime` | datetime | 排队时间 |
| `MonitorFilename` | TextField | **录音URL（HTTP下载地址）** |
| `AudioFile` | FileField | **录音文件（OSS相对路径）** |

#### Session 表（heli_session）- 在线会话

| 字段 | 类型 | 说明 |
|------|------|------|
| `SessionID` | varchar(48) | 会话ID（唯一标识） |
| `VisitorId` | varchar(64) | **访客ID（客户标识）** |
| `AgentId` | varchar(64) | 坐席ID |
| `BeginTime` | datetime | 会话开始时间 |

#### Message 表（heli_message）- 在线消息

| 字段 | 类型 | 说明 |
|------|------|------|
| `SessionID` | varchar(48) | 会话ID（关联 Session 表） |
| `FromUserName` | varchar(48) | **发送者ID** |
| `ToUserName` | varchar(64) | **接收者ID** |
| `Content` | TextField | **消息内容** |
| `CreateTime` | datetime | 消息时间 |

#### dialog_analysis 表 - 分析结果（MySQL）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 主键 |
| `source_type` | varchar(20) | 数据来源：phone/online |
| `source_file` | varchar(255) | 原始文件路径 |
| `dialog_date` | date | 对话日期 |
| `primary_intent` | varchar(20) | 一级意图分类 |
| `secondary_intent` | varchar(50) | 二级意图分类 |
| `sentiment` | varchar(20) | 情感倾向：满意/中立/不满 |
| `issue_type` | varchar(100) | 问题类型 |
| `created_at` | datetime | 创建时间 |

---

## 脚本文件

| 文件 | 功能 | 说明 |
|-----|------|------|
| `main.py` | 完整处理流程（下载+处理） | 从数据库读取 |
| `tool_download_heli_audio.py` | 独立下载脚本 | 只下载不处理 |
| `process_audio.py` | 独立处理脚本 | 处理已下载音频，自动跳过已处理 |
| `batch_process_audio.py` | 分批处理脚本 | 失败隔离，每批独立 |
| `process_online_chat.py` | 在线对话提取 | 规则匹配 |
| `analyze.py` | 综合分析脚本 | 意图+情感+高频问题+PDF报告 |
| `import_to_db.py` | 数据导入数据库 | CSV → MySQL |
| `diarize_worker.py` | 说话者分离子进程 | pyannote，支持断点续传 |

---

## 使用方式

### 下载音频
```bash
# 下载 10,000 条（从最新开始）
python tool_download_heli_audio.py --success-limit 10000

# 下载到指定目录
python tool_download_heli_audio.py --success-limit 10000 --target-dir ./mp3
```

### 处理音频
```bash
# 分批处理（推荐，失败隔离）
python batch_process_audio.py ./mp3 500 20
# 参数：音频目录、每批文件数、API线程数

# 单次处理（测试用）
python process_audio.py ./mp3 20 --limit 100

# 复用分离缓存（ASR/API失败后）
python process_audio.py ./mp3 20 --reuse-diarization

# 跳过分离阶段（无角色标注，节省40%时间）
python batch_process_audio.py ./mp3 500 20 --skip-diarization
```

### 在线对话
```bash
# 生成 10 个有效 CSV
python process_online_chat.py 10

# 全量处理
python process_online_chat.py
```

### 分析 CSV（推荐使用新脚本）
```bash
# 分析电话录音
python analyze.py --dir dialog_csv --limit 200

# 分析在线对话
python analyze.py --dir online_chat_csv --limit 200

# 分析全部数据
python analyze.py --dir all

# 跳过某些分析
python analyze.py --dir dialog_csv --skip-intent    # 跳过意图分类
python analyze.py --dir dialog_csv --skip-sentiment # 跳过情感分析
python analyze.py --dir dialog_csv --skip-issue     # 跳过高频问题提取
```

### 导入数据库
```bash
# 导入在线对话（106,023条）
python import_to_db.py --type online

# 导入电话录音（10,121条）
python import_to_db.py --type phone

# 导入全部
python import_to_db.py --type all

# 限制数量测试
python import_to_db.py --type online --limit 100
```

### 完整处理流程
```bash
# 步骤1：下载音频（已完成 103,012 个）
python tool_download_heli_audio.py

# 步骤2：分批处理音频 → CSV
python batch_process_audio.py ./02_download/audio 500 20

# 步骤3：生成分析报告
python analyze.py --dir dialog_csv

# 步骤4：导入数据库
python import_to_db.py --type all
```

---

## 性能参数

### 关键参数说明

| 参数 | 文件 | 当前值 | 影响 | 说明 |
|------|------|--------|------|------|
| `ASR_BATCH_SIZE` | process_audio.py | 5 | 批处理文件数 | 已降低避免OOM |
| `max_inference_batch_size` | process_audio.py | 2 | GPU 推理并行度 | 已降低避免OOM |
| `max_model_len` | process_audio.py | 4096 | KV cache 大小 | 已降低避免OOM |
| `gpu_memory_utilization` | process_audio.py | 0.4 | 显存利用率 | 固定值 |
| `api_workers` | 命令行参数 | 20 | API 并发线程 | |

### 参数与 GPU 显存关系

| 参数 | 影响 GPU 显存 | 说明 |
|------|--------------|------|
| `max_inference_batch_size` | **直接影响** | vLLM 同时推理多少个请求，越大越占显存 |
| `max_model_len` | **直接影响** | KV cache 大小，决定能处理的文本长度 |
| `gpu_memory_utilization` | **直接影响** | 模型预留显存比例 |
| `ASR_BATCH_SIZE` | **间接影响** | Forced Aligner 同时处理的音频数，音频越长显存越大 |

### ASR_BATCH_SIZE 与 max_inference_batch_size 的区别

| 参数 | 作用位置 | 控制什么 |
|------|---------|---------|
| `ASR_BATCH_SIZE` | 外部调用 | 一次传多少个文件给 transcribe() |
| `max_inference_batch_size` | vLLM 内部 | vLLM 同时推理多少个请求 |

**示例：** `ASR_BATCH_SIZE=5` + `max_inference_batch_size=2`
- 传入 5 个文件，但 vLLM 每次只处理 2 个
- vLLM 显存按 2 个计算，不是 5 个

### 长音频处理

| 类型 | 阈值 | 处理方式 |
|------|------|---------|
| 正常音频 | <600秒 | 批次处理，每批5个 |
| 长音频 | >600秒 | 单独处理，每批1个 |

---

## 使用的模型

| 功能 | 模型 | 位置 | 说明 |
|------|------|------|------|
| 说话者分离 | pyannote/speaker-diarization-3.1 | 本地 GPU | ~300M 参数 |
| ASR 转录 | Qwen3-ASR-1.7B | 本地 GPU | 大模型，理解能力强 |
| 时间对齐 | Qwen3-ForcedAligner-0.6B | 本地 GPU | 精确时间戳 |
| **对话分类/情感分析** | **Qwen3-8B** | **API** | 多线程并发 |
| **高频问题提取** | **Qwen3-8B** | **API** | 多线程并发 |

### 各阶段 GPU 使用详解

#### 阶段1: 说话者分离（pyannote）

**为什么需要 GPU：** pyannote 是神经网络模型，需要 GPU 做推理计算

**计算过程：**
1. 音频特征提取：音频波形 → 梅尔频谱图 → 特征向量
2. 声纹嵌入提取：特征向量 → 神经网络 → 声纹向量（每个人的"声纹指纹"）
3. 说话者聚类：相似的声纹向量 → 归为同一说话者

**显存占用：** ~2-4 GiB

#### 阶段2: ASR 转录（Qwen3-ASR + vLLM）

**组成：**
- Qwen3-ASR-1.7B：语音转文字
- Forced Aligner 0.6B：时间对齐（给每个字标注时间戳）
- vLLM：推理加速框架

**为什么 Forced Aligner 需要 GPU：**
- 是神经网络模型（0.6B 参数）
- 需要计算音频特征与文字特征的注意力匹配
- 音频越长，特征矩阵越大，显存需求越高

**显存占用：**
- 模型权重：~4-5 GiB（ASR 1.7B + Aligner 0.6B）
- KV cache：由 `max_model_len` 决定
- Forced Aligner：随音频长度增长

#### 阶段3: API 分类

**不使用本地 GPU**，调用阿里云 Qwen3-8B API

### 参数可视化说明

详细参数说明见 `PARAMETERS.md`，包含：
- 显存占用关系图
- 处理流程与参数关系
- 参数对显存的影响
- 容错机制流程

---

## API 配置

```python
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"

# 注意: API 请求需要添加 enable_thinking: False
```

---

## OSS 配置

```python
OSS_ACCESS_KEY_ID = "${OSS_ACCESS_KEY_ID}"
OSS_ACCESS_KEY_SECRET = "${OSS_ACCESS_KEY_SECRET}"
OSS_BUCKET_NAME = "njzhyl-insurance-claim"
OSS_ENDPOINT = "oss-cn-beijing-internal.aliyuncs.com"
OSS_MEDIA_PREFIX = "invoiceRecogMedia/prod"
```

---

## 优化方向

### 已完成优化
- ✅ 下载逻辑优化（30天内外不同域名）
- ✅ 分离下载和处理脚本
- ✅ API 多线程并发
- ✅ 批处理 ASR 转录
- ✅ 分离断点续传（每文件保存）
- ✅ 分离结果缓存（可复用）
- ✅ 分批处理脚本（失败隔离）
- ✅ API 重试机制（3次重试）
- ✅ 长音频单独处理
- ✅ **综合分析脚本 analyze.py（意图+情感+高频问题+PDF报告）**
- ✅ **可视化图表（matplotlib，支持中文）**
- ✅ **动态观点生成（基于数据自动生成洞察）**

### 容错保障机制

| 保障 | 说明 |
|------|------|
| 分离断点续传 | 每个文件处理完就保存到缓存 |
| 分离缓存 | 保存到 `diarization_cache/`，可复用 |
| `--reuse-diarization` | ASR/API失败后跳过分离阶段 |
| CSV已存在跳过 | 不会重复处理已完成的文件 |
| API重试 | 失败自动重试3次，每次间隔2秒 |
| 分批处理 | 每批独立，失败只影响该批 |

### 可进一步优化
- 跳过说话者分离（节省 ~40% 时间，但无法区分客服/客户）
- 增大 `ASR_BATCH_SIZE`（需要更多显存）

---

## 文件结构

```
/home/REMOVED_DB_USER/customer-service/
├── 01_source/
│   └── heli.sqlite3               # SQLite 原始数据库
├── 02_download/
│   └── audio/                     # 音频文件 (103,012 个 mp3)
├── 03_process/
│   └── cache/                     # 说话者分离缓存
├── 04_output/
│   ├── dialog_csv/                # 电话录音 CSV (10,121 个)
│   └── online_chat_csv/           # 在线对话 CSV (106,023 个)
├── 05_analyze/
│   ├── reports/                   # PDF 分析报告
│   └── database/
│       └── db_store.py            # MySQL 存储模块
├── 06_models/
│   ├── asr/Qwen3-ASR-1.7B/        # ASR 模型
│   └── aligner/Qwen3-ForcedAligner-0.6B/  # 时间对齐模型
├── scripts/
│   ├── config.py                  # 路径配置
│   ├── main.py                    # 完整处理流程
│   ├── tool_download_heli_audio.py  # 下载脚本
│   ├── process_audio.py           # 音频处理
│   ├── batch_process_audio.py     # 分批处理
│   ├── process_online_chat.py     # 在线对话处理
│   ├── analyze.py                 # 分析报告生成
│   ├── import_to_db.py            # 数据导入数据库
│   └── diarize_worker.py          # 说话者分离
└── docs/
    ├── context.md                 # 项目上下文（本文档）
    ├── CHANGELOG.md               # 改动记录
    └── PARAMETERS.md              # 参数说明
```

---

## 历史记录

### 2026年3月25日
- 对话分类改用 API（Qwen3-8B），多线程并发
- 成功率：API 100% vs 本地 10%
- 添加 vLLM 加速框架

### 2026年3月27日
- 实现 `--from-db` 从数据库直接读取
- 发现 2026-03-01 之前数据已失效
- 更新下载逻辑：优先 OSS，回退 HTTP

### 2026年3月30日（上午）
- 查看 Django 模型文件 `/home/REMOVED_DB_USER/asr/heli/models.py`
- 理解 AudioFile 字段为相对路径的原因（Django upload_to）
- 确认数据有效性：仅 2026-03-01 之后可下载
- 修改 main.py，移除中间 TXT 文件输出，直接生成 CSV

### 2026年3月30日（下午）
- 新增 `process_online_chat.py` 处理在线对话
- 实现规则匹配判断发送者身份（无需大模型）
- 处理 Session 表 183,207 条会话
- 处理 Message 表 1,796,321 条消息
- 过滤系统噪音和无效对话（机器人问候无客户参与）
- 特殊处理：识别客户点击选项触发的问题（问句形式）
- 按时间倒序输出，确保生成指定数量的有效 CSV

### 2026年3月31日
- 修改下载逻辑：30天内直接下载，30天前替换域名
- 创建 `process_audio.py` 独立处理脚本（分离下载和处理）
- 创建 `analyze_dialog.py` 情感分析+高频问题提取脚本
- 测试 API 调用，修复 `enable_thinking: False` 问题
- 修复 `find_speaker` 函数兼容列表格式 segments

### 2026年4月1日
- 下载 10,121 个音频文件到 `mp3/` 目录
- 运行 `process_audio.py` 处理音频
- 分析各阶段耗时：说话者分离(40%) + ASR(40%) + API(20%)
- 讨论参数调优：`ASR_BATCH_SIZE`、`max_inference_batch_size`
- 确认瓶颈：说话者分离是单线程，无法充分利用 GPU
- 确认只能使用一块 GPU（不能用多 GPU 并行）

### 2026年4月1日-2日（重大更新）
- **新增 `batch_process_audio.py`**：分批处理脚本，失败隔离
- **分离断点续传**：`diarize_worker.py` 每处理一个文件就保存结果
- **分离结果缓存**：保存到 `diarization_cache/`，可复用
- **新增命令行参数**：
  - `--limit N`：限制处理文件数
  - `--skip-diarization`：跳过分离阶段
  - `--reuse-diarization`：复用分离缓存
- **参数调整（避免OOM）**：
  - `ASR_BATCH_SIZE`: 20 → 5
  - `max_inference_batch_size`: 6 → 2
  - `max_model_len`: 16384 → 4096
  - `gpu_memory_utilization`: 动态 → 固定0.4
- **长音频处理**：不跳过，改为单独处理（每批1个）
- **API重试机制**：失败重试3次，超时增加到120秒
- **Bug修复**：
  - 修复 `--limit` 时路径不匹配问题
  - 修复 `file_paths` 变量未初始化问题
- **创建文档**：
  - `CHANGELOG.md`：记录改动历史
  - `PARAMETERS.md`：参数可视化说明
- **技术文档更新**：
  - 添加 GPU 使用详解（分离、ASR、Forced Aligner）
  - 添加参数对比说明

### 2026年4月3日（分析报告功能）
- **新增 `analyze.py` 综合分析脚本**：
  - 整合意图分类、情感分析、高频问题提取
  - 支持命令行选择分析目录（dialog_csv/online_chat_csv/all）
  - 输出单个PDF报告，标题匹配所选目录
- **PDF报告功能**：
  - 使用 reportlab 生成PDF
  - 支持中文字体（NotoSansCJK）
  - 8个章节：数据概览、意图分类、情感分析、高频问题、核心观点、风险机会、监测指标、行动清单
- **可视化图表**：
  - 使用 matplotlib 生成图表（支持中文）
  - 7个图表：饼图+柱状图
  - 图表包括：关键指标对比、一级分类分布、二级分类TOP10、情感分布、不满客户问题、高频问题TOP10、问题类型细分
- **动态洞察生成**：
  - 所有观点和建议基于数据自动生成
  - 例如：理赔占比超40%时自动生成理赔服务优化建议
  - 不满比例超15%时自动生成不满客户分析建议
- **高频问题标准化**：
  - 21个标准化问题类型，支持有效聚合
  - 每个高频问题展示3个具体案例

### 2026年4月7日（分类体系优化）
- **意图分类体系重构（Plan A）**：
  - 一级分类改为"客户行为"维度：咨询、查询、办理、投诉、其他
  - 删除原有业务维度分类：理赔、退保、报案（改为二级分类）
  - 解决原分类重叠问题（理赔类别混杂咨询、查询、办理多种行为）
- **分类体系变更对比**：
  - 修改前：理赔、咨询、退保、投诉、办理、报案、其他（7个一级分类）
  - 修改后：咨询、查询、办理、投诉、其他（5个一级分类）
  - 二级分类：共22个，明确行为定义
- **核心改动**：
  - `INTENT_CATEGORIES`：更新分类定义
  - `PRIMARY_CATEGORIES`：更新一级分类列表
  - `INTENT_PROMPT`：更新分类提示词，增加说明
  - PDF报告生成：更新统计逻辑和观点生成

### 2026年4月9日（数据库导入与批量下载）
- **音频下载完成**：
  - 下载 103,012 个音频文件到 `02_download/audio/`
  - 只下载有 AudioFile 字段的记录（102,912条有效）
- **数据库导入模块**：
  - 新增 `import_to_db.py` 脚本
  - 支持导入电话录音和在线对话到 MySQL
  - 表结构：`dialog_analysis`（意图、情感、问题类型）
  - 测试导入 100 条在线对话成功
- **脚本优化**：
  - `tool_download_heli_audio.py`：只筛选有 AudioFile 的记录
  - `main.py`：添加 `keep_temp` 参数，修复函数签名
  - `process_audio.py`：添加 `SCRIPT_DIR` 变量，修复路径问题
  - `import_to_db.py`：按时间排序获取最新文件
- **项目迁移**：
  - 项目从 `/home/REMOVED_DB_USER/asr/` 迁移到 `/home/REMOVED_DB_USER/customer-service/`
  - 目录结构统一按 01-06 编号

### 2026年4月13日（ASR处理进度）
- **使用缓存复用模式处理剩余音频**：
  - 运行指令：`python scripts/process_audio.py ./02_download/audio 20 --reuse-diarization`
  - 加载 `audio_all.json` 缓存（101,343 个文件的分离数据）
  - 跳过说话者分离阶段，直接进行 ASR 转录 + API 分类
- **处理进度**：
  - 音频文件总数：103,012 个
  - 已生成 CSV：38,123 个（截至 4月11日）
  - 待处理：约 64,889 个
  - 进程状态：ASR 转录阶段运行中（已运行约5小时）
- **GPU 使用情况**：
  - VLLM Engine: 22.3 GB 显存
  - 用户进程: 2.2 GB 显存
  - 共使用 GPU 0-3

### 2026年4月14日（HTML投诉深度分析）

- **新增 analyze_complaint.py 投诉类深度分析脚本（HTML输出）**：
  - 专门分析投诉类对话，生成详细的 HTML 分析报告
  - 使用 CPU + API 模式（不占用 GPU）
  - 输出 HTML 格式（比 PDF 对图表更友好）
- **报告功能**：
  - 投诉类型分类：理赔时效投诉、拒赔投诉、理赔金额异议、服务态度投诉、流程复杂投诉
  - 投诉阶段分析：理赔申请阶段、审核阶段、打款阶段、拒赔阶段
  - 严重程度评估：严重、中等、轻微
  - 根本原因提取：自动识别投诉根源（TOP10）
  - 处理效果跟踪：已解决、部分解决、未解决、升级处理
  - 高频投诉问题分析：TOP10问题表格，含类型、阶段、客户诉求摘要、客服回复摘要
  - 自动优化建议：基于数据生成针对性改进方案
- **HTML特性**：
  - 导航链接：可链接到其他分类报告（咨询类、查询类、办理类、总览汇总）
  - 图表嵌入：使用base64编码嵌入，不依赖外部文件
  - CSS 样式：红色主题配色，专业外观
  - 统计卡片：动态显示关键指标（投诉总数、严重投诉数、未解决数等）
- **使用方式**：
  ```bash
  python scripts/analyze_complaint.py --dir online_chat_csv --limit 1000
  python scripts/analyze_complaint.py --dir online_chat_csv  # 全量分析
  ```
- **问题修复**：
  - CSS花括号问题：f-string 中 CSS 的 `{}` 被解析为表达式，改为普通字符串变量
  - 严重程度标签颜色：添加 `color:white;` 使文字可见
  - clean_text函数：保留换行符，避免对话被压缩成一行
  - CSV标题行过滤：添加"对话者"到过滤列表
  - 图表加载问题：改用base64嵌入图片，HTML自包含不依赖外部文件
  - 高频问题分析：改为表格形式，展示TOP10问题及简短的诉求/回复摘要
- **新增严重程度判断标准说明**：
  - 在HTML报告中添加判断标准框，明确三个级别的定义
  - 更新API Prompt，让模型按标准分类（严重/中等/轻微）
  - 标准：严重-情绪激动/威胁投诉/金额超1万；中等-有明显不满/需跟进；轻微-咨询性质/情绪平和

### 2026年4月15日（进度更新）

- **音频转录进度**：
  - ASR批次：11832/19853（约59.6%）
  - 已处理文件：约59,160个
  - 剩余文件：约40,105个
  - 已运行时间：2天1小时
  - 预计剩余时间：约1.5天完成ASR + 3-4小时API分类
- **投诉分析脚本优化**：
  - 移除报告中的饼图和柱状图（单一分类不需要图表）
  - 修复进度计数器重复增加问题（3900/2668 → 正确显示）
  - 文件大小从446KB减少到27KB
- **报告功能说明**：
  - 9个章节：数据概览、类型分布、发生阶段、严重程度、根本原因TOP10、处理效果、高频问题TOP10、优化方案、总体建议
  - 高频问题表格：含问题、次数、类型、阶段、客户诉求摘要、客服回复摘要
  - 自动生成针对性优化建议

### 2026年4月16日（理赔时效投诉深度分析）

- **新增 analyze_time_complaint.py 脚本**：
  - 从数据库筛选理赔时效投诉案例进行深度分析
  - 分析维度：慢的环节、慢的根本原因、客户核心不满、客服回复问题、改进建议
- **分析结果**（30条案例）：
  - 慢的环节：审核阶段43.3%、立案阶段40%、打款阶段6.7%
  - 慢的原因：60%是信息沟通不畅（客户不知道进度）
  - 客服回复问题：只说标准时效，不查具体进度
  - 解决状态：100%未解决
- **改进建议具体化**：
  - 立案环节：缩短到3个工作日，超时自动提醒
  - 审核环节：材料不全一次性告知，避免反复驳回
  - 信息沟通：增加理赔进度实时查询功能
  - 客服培训：先查具体进度再回复，而不是只说标准时效
- **创建校对问题清单**：
  - 整理需要与审核部门确认的问题
  - 包含：数据范围、流程标准、环节定义、审核人员配置、客服查询权限等

### 2026年4月17日（数据校验与进度更新）

- **音频转录进度**：
  - 正常音频批次：已完成
  - 长音频单独处理：3552/3748（**94.7%**）
  - 进程运行时间：3天23小时20分
  - 预计：1-2小时完成转录，之后API分类3-4小时
- **投诉数据校验发现**：
  - 理赔时效投诉实际数量：**265条**（不是之前分析的804条）
  - 投诉总数：2282条，占总对话106,023条的**2.2%**
  - 数据差异原因：AI分类可能存在误判，"查询理赔进度"被误判为"理赔时效投诉"
- **投诉二级分类实际数据**：
  - 其他投诉：929条
  - 拒赔投诉：610条
  - 服务态度投诉：300条
  - 理赔时效投诉：265条
  - 理赔金额异议：168条
- **与客服部门沟通**：
  - 客服反馈实际投诉数量少于分析数据
  - 需要校验AI分类准确性
  - 需要确认审核流程各环节标准时间、人员配置、客服查询权限

---

最后更新: 2026-04-17