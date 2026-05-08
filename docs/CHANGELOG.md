# 音频处理流水线改动记录

## 2026年4月1日-2日改动

---

### 一、新增文件

#### batch_process_audio.py
分批处理脚本，将大量音频文件分成小批次处理。

**功能：**
- 每批独立处理，失败隔离
- 自动生成失败批次的重跑命令
- 保留失败批次目录，方便手动重试

**使用方式：**
```bash
python batch_process_audio.py ./mp3 500 20
# 参数：音频目录、每批文件数、API线程数
```

---

### 二、process_audio.py 改动

#### 1. 新增命令行参数

| 参数 | 说明 |
|------|------|
| `--limit N` | 只处理前N个文件（测试用） |
| `--skip-diarization` | 跳过说话者分离阶段 |
| `--reuse-diarization` | 复用已有的分离结果缓存 |

#### 2. 新增配置

| 配置 | 位置 | 说明 |
|------|------|------|
| `DIAR_CACHE_DIR` | `diarization_cache/` | 分离结果缓存目录 |
| `LONG_AUDIO_THRESHOLD` | 600秒 | 长音频阈值 |

#### 3. 参数调整（避免OOM）

| 参数 | 之前 | 现在 | 说明 |
|------|------|------|------|
| `ASR_BATCH_SIZE` | 20 | 5 | 每批处理文件数 |
| `max_inference_batch_size` | 6 | 2 | vLLM推理并行度 |
| `max_model_len` | 16384 | 4096 | KV cache大小 |
| `max_new_tokens` | 4096 | 2048 | 输出token限制 |
| `gpu_memory_utilization` | 动态(0.70) | 固定0.4 | 显存利用率 |
| `API timeout` | 60秒 | 120秒 | API超时时间 |

#### 4. 长音频处理逻辑

不再跳过长音频，改为分类处理：
- 正常音频（<600秒）：批次处理，每批5个
- 长音频（>600秒）：单独处理，每批1个，处理完清理显存

#### 5. API重试机制

- 失败自动重试3次
- 每次重试间隔2秒
- 失败记录错误信息，继续处理下一个

#### 6. Bug修复

- 修复 `--limit` 时分离和ASR路径不匹配的问题
- 修复分离结果保存到缓存目录的问题
- 修复 `file_paths` 变量未初始化的问题

---

### 三、diarize_worker.py 改动

#### 断点续传功能

| 函数 | 说明 |
|------|------|
| `save_results()` | 保存结果到JSON |
| `load_existing_results()` | 加载已有结果 |

**工作方式：**
- 每处理完一个文件就保存结果
- 启动时检查已有结果，跳过已处理的文件
- 即使出错也保存当前结果

---

### 四、使用方式变化

#### 之前
```bash
python process_audio.py ./mp3 20   # 一次性处理，失败需重来
```

#### 现在
```bash
# 测试
python process_audio.py ./mp3 20 --limit 100

# 复用分离缓存（ASR/API失败后）
python process_audio.py ./mp3 20 --reuse-diarization

# 分批处理全部（推荐）
python batch_process_audio.py ./mp3 500 20
```

---

### 五、文件结构变化

```
新增目录：
  diarization_cache/    # 分离结果缓存
  batch_temp/           # 分批处理临时目录

新增文件：
  batch_process_audio.py
```

---

### 六、容错保障机制

| 保障 | 说明 |
|------|------|
| 分离断点续传 | 每个文件处理完就保存到缓存 |
| 分离缓存 | 保存到 `diarization_cache/`，可复用 |
| `--reuse-diarization` | ASR/API失败后跳过分离阶段 |
| CSV已存在跳过 | 不会重复处理已完成的文件 |
| API重试 | 失败自动重试3次，每次间隔2秒 |
| API超时增加 | 120秒超时 |
| 长音频单独处理 | 失败不影响其他文件 |
| 分批处理 | 每批独立，失败只影响该批 |

---

### 七、时间预估

**10,000个文件处理时间：**

| 阶段 | 耗时 |
|------|------|
| 说话者分离 | ~5小时 |
| ASR转录 | ~5小时 |
| API分类 | ~2.5小时 |
| **总计** | **~12-14小时** |

---

### 八、报错恢复方法

```bash
# 1. 清理残留进程
nvidia-smi
pkill -9 -f python

# 2. 查看已完成数量
ls dialog_csv/*.csv | wc -l

# 3. 重新运行（自动跳过已完成）
python batch_process_audio.py ./mp3 500 20

# 4. 如果ASR失败但分离已完成
python process_audio.py ./mp3 20 --reuse-diarization
```