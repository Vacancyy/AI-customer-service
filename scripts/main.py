"""
ASR 转录 + 说话者分离 + 对话分类（API多线程）+ 保存 CSV
混合架构：本地处理音频，API处理对话分类

使用方式:
  # 从本地目录处理
  python main.py /path/to/audio
  python main.py /path/to/audio 20     # 指定API并发线程数

  # 从数据库直接读取（不保存音频到本地）
  python main.py --from-db 100         # 处理最早100条
  python main.py --from-db 1000 20     # 处理1000条，20线程并发
"""

import os
import sys
import time
import json
import subprocess
import csv
import re
import sqlite3
import tempfile
import shutil
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse, unquote

os.environ["HF_HUB_OFFLINE"] = "1"

# OSS 签名配置（用于 2026-03-01 之前的数据）
try:
    import oss2
    OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    OSS_BUCKET_NAME = "njzhyl-insurance-claim"
    OSS_ENDPOINT = "oss-cn-beijing-internal.aliyuncs.com"
    OSS_URL_BASE = "oss-cn-beijing.aliyuncs.com"
    OSS_EXPIRE_TIME = 3600 * 24
    OSS_MEDIA_PREFIX = "invoiceRecogMedia/prod"

    OSS_AUTH = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    OSS_BUCKET = oss2.Bucket(OSS_AUTH, OSS_ENDPOINT, OSS_BUCKET_NAME)
    OSS_AVAILABLE = True
except ImportError:
    print("警告: oss2 未安装，OSS 签名下载功能不可用")
    OSS_AVAILABLE = False

# 日期分界点（2026-03-01 之后用 HTTP，之前用 OSS）
DATE_CUTOFF = "2026-03-01"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(_k, None)

import warnings
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import torch
import librosa
from qwen_asr import Qwen3ASRModel

# ==================== 配置 ====================
DEFAULT_INPUT = os.path.join(os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "02_download/audio")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
ASR_BATCH_SIZE = 10  # 减小批量大小避免内存溢出

# 导入路径配置
from config import DB_PATH, AUDIO_DIR, DIALOG_CSV_DIR, CACHE_DIR, ASR_MODEL_DIR, ALIGNER_MODEL_DIR

# OSS 配置
try:
    import oss2
    OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    OSS_BUCKET_NAME = "njzhyl-insurance-claim"
    OSS_ENDPOINT = "oss-cn-beijing-internal.aliyuncs.com"
    OSS_URL_BASE = "oss-cn-beijing.aliyuncs.com"
    OSS_EXPIRE_TIME = 3600 * 24
    OSS_MEDIA_PREFIX = "invoiceRecogMedia/prod"

    OSS_AUTH = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    OSS_BUCKET = oss2.Bucket(OSS_AUTH, OSS_ENDPOINT, OSS_BUCKET_NAME)
    OSS_AVAILABLE = True
except ImportError:
    print("警告: oss2 未安装，OSS 下载功能不可用", file=sys.stderr)
    OSS_AVAILABLE = False

# API 配置（对话分类使用 API 多线程）
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10  # API 并发线程数

# 计费标准（Qwen3-8B）
INPUT_PRICE = 0.0005   # ¥/千Token
OUTPUT_PRICE = 0.002   # ¥/千Token

# Prompt 模板
PROMPT_TEMPLATE = """请分析下面这段客服电话转录文本，识别每一句话是"客服"说的还是"客户"说的。

转录文本：
{transcript}

要求：
1. 输出每一句对话，格式为：对话者|内容
2. 对话者只能是"客服"或"客户"
3. 去除语气词（嗯、啊、呃等无意义内容）
4. 不要解释，直接输出对话列表

示例输出格式：
客服|您好，请问有什么可以帮助您
客户|我想咨询一下理赔的问题
客服|好的，请问您的保单号是多少
"""


# ==================== 数据库相关函数 ====================

def fetch_audio_urls_from_db(limit=1000, start_date=None, end_date=None):
    """从数据库获取音频 URL（去重）

    Args:
        limit: 下载数量
        start_date: 开始日期（>=），用于指定范围
        end_date: 截止日期（<=），默认从该日期往前下载
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 构建查询条件（只筛选有 AudioFile 的记录）
    where_conditions = [
        "id IS NOT NULL",
        "CallTimeLength IS NOT NULL",
        "CallTimeLength <> 0",
        "QueueTime IS NOT NULL",
        "MonitorFilename IS NOT NULL AND MonitorFilename != ''",
        "AudioFile IS NOT NULL AND AudioFile != '' AND AudioFile != 'normal'"
    ]

    if start_date:
        where_conditions.append("date(QueueTime) >= date(?)")
    if end_date:
        where_conditions.append("date(QueueTime) <= date(?)")

    where_clause = " AND ".join(where_conditions)

    # 从最新日期往前下载（DESC 倒序）
    query = f"""
        SELECT id, MonitorFilename, AudioFile, CallTimeLength, QueueTime
        FROM heli_audio
        WHERE {where_clause}
        ORDER BY QueueTime DESC
        LIMIT ?
    """

    # 构建参数
    params = []
    if start_date:
        params.append(start_date)
    if end_date:
        params.append(end_date)
    params.append(limit * 2)

    cur.execute(query, params)

    rows = cur.fetchall()
    conn.close()

    # 去重
    seen = set()
    results = []
    for audio_id, monitor_url, audio_file, duration, queue_time in rows:
        key = monitor_url or audio_file or f"id_{audio_id}"
        if key in seen:
            continue
        seen.add(key)

        results.append({
            'id': audio_id,
            'monitor_url': monitor_url,
            'audio_file': audio_file,
            'duration': duration,
            'queue_time': queue_time,
        })

        if len(results) >= limit:
            break

    return results


def is_oss_path(path: str) -> bool:
    """判断是否是 OSS 路径"""
    if not path:
        return False
    oss_keywords = ['oss', 'invoiceRecogMedia', 'aliyuncs.com']
    return any(kw in path.lower() for kw in oss_keywords)


def build_oss_signed_url(key: str) -> str:
    """生成 OSS 签名 URL（尝试两种前缀）"""
    if not OSS_AVAILABLE:
        return None

    raw_key = key.lstrip("/")
    if not raw_key:
        return None

    # 尝试不同的 key 前缀
    key_candidates = [raw_key]
    if not raw_key.startswith("invoiceRecogMedia/"):
        key_candidates.insert(0, f"{OSS_MEDIA_PREFIX}/{raw_key}")

    for try_key in key_candidates:
        try:
            signed_url = OSS_BUCKET.sign_url("GET", try_key, expires=OSS_EXPIRE_TIME, slash_safe=True)
            return signed_url.replace(OSS_ENDPOINT, OSS_URL_BASE)
        except Exception:
            continue

    return None


def download_audio_to_temp(url: str, temp_dir: str, filename: str) -> str:
    """下载音频到临时目录，返回文件路径"""
    temp_path = os.path.join(temp_dir, filename)

    # 下载
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    # 检查是否是有效的音频文件（至少 1KB）
    content_length = int(resp.headers.get('Content-Length', 0))
    if content_length > 0 and content_length < 1024:
        raise ValueError(f"文件太小 ({content_length} bytes)，可能不是有效音频")

    with open(temp_path, 'wb') as f:
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    # 再次检查文件大小
    actual_size = os.path.getsize(temp_path)
    if actual_size < 1024:
        os.remove(temp_path)
        raise ValueError(f"文件太小 ({actual_size} bytes)，可能不是有效音频")

    return temp_path


def get_download_url(record: dict) -> tuple:
    """获取可下载的 URL 和来源

    返回: (url, source) 或 (None, None)

    优先级（与 tool_download_heli_audio.py 一致）：
    1. AudioFile（OSS）- 存在就尝试
    2. MonitorFilename（HTTP 直连）
    """
    audio_file = record.get('audio_file', '')
    monitor_url = record.get('monitor_url', '')

    # 1) 优先尝试 AudioFile（OSS）
    # AudioFile 可能是 OSS 路径或相对路径，都尝试 OSS 签名
    if audio_file and OSS_AVAILABLE:
        oss_url = build_oss_signed_url(audio_file)
        if oss_url:
            return oss_url, 'oss'

    # 2) 回退 MonitorFilename（HTTP 直连）
    if monitor_url:
        # 如果是 OSS URL，尝试签名
        if "aliyuncs.com" in monitor_url and OSS_AVAILABLE:
            signed = build_oss_signed_url(urlparse(monitor_url).path)
            if signed:
                return signed, 'oss'
        return monitor_url, 'http'

    return None, None


# ==================== 工具函数 ====================

def get_audio_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        if not files:
            print(f"目录 {input_path} 中没有找到音频文件")
        return files
    else:
        print(f"路径不存在: {input_path}")
        return []


def get_audio_duration(file_path):
    """获取音频时长（秒）"""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        return 0.0


def format_duration(seconds):
    """格式化时长显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.2f}小时"


def find_speaker(start, end, segments):
    """根据时间戳匹配说话者"""
    best_speaker = None
    best_overlap = -1.0
    for seg_start, seg_end, speaker in segments:
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    if best_speaker is None:
        return "UNKNOWN"
    if isinstance(best_speaker, str):
        return best_speaker
    return f"SPEAKER_{int(best_speaker):02d}"


def call_api_for_dialog(text):
    """调用 API 进行对话分类"""
    prompt = PROMPT_TEMPLATE.format(transcript=text) + " /no_think"

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "enable_thinking": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    content = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    dialog_list = parse_dialog_content(content)
    return dialog_list, input_tokens, output_tokens


def parse_dialog_content(content: str) -> list:
    """解析对话内容"""
    dialog_list = []
    content = re.sub(r'海淀区.*?海淀区', '', content, flags=re.DOTALL).strip()

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("对话内容") or line.startswith("对话："):
            continue

        if "|" in line:
            parts = line.split("|", 1)
        elif "：" in line:
            parts = line.split("：", 1)
        elif ":" in line:
            parts = line.split(":", 1)
        else:
            continue

        if len(parts) == 2:
            speaker = parts[0].strip()
            content_text = parts[1].strip()

            if "客户" in speaker:
                speaker = "客户"
            elif "客服" in speaker:
                speaker = "客服"
            else:
                continue
            dialog_list.append((speaker, content_text))
    return dialog_list


# 线程安全的进度计数
progress_lock = Lock()
progress_counter = 0
total_tokens_input = 0
total_tokens_output = 0


def process_single_dialog(audio_name, transcript_text, csv_dir, total_files):
    """处理单个文件的对话分类"""
    global progress_counter, total_tokens_input, total_tokens_output

    try:
        dialog_list, input_tokens, output_tokens = call_api_for_dialog(transcript_text)

        with progress_lock:
            progress_counter += 1
            total_tokens_input += input_tokens
            total_tokens_output += output_tokens
            current = progress_counter
            status = "✓" if dialog_list else "✗"
            print(f"  [{current}/{total_files}] {status} {audio_name[:35]}... | 对话:{len(dialog_list)}")

        if dialog_list:
            csv_path = os.path.join(csv_dir, f"{audio_name}.csv")
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["对话者", "内容"])
                for speaker, content in dialog_list:
                    writer.writerow([speaker, content])
            return True
        return False

    except Exception as e:
        with progress_lock:
            progress_counter += 1
            current = progress_counter
            print(f"  [{current}/{total_files}] ✗ {audio_name[:35]}... | 错误: {e}")
        return False


# ==================== 主程序 ====================

def process_from_db(limit, api_workers, start_date=None, end_date=None, keep_temp=False):
    """从数据库读取并处理

    Args:
        limit: 处理数量上限
        api_workers: API并发线程数
        start_date: 开始日期
        end_date: 结束日期
        keep_temp: 是否保留临时音频文件
    """
    print(f"\n{'='*60}")
    print(f"ASR 处理流水线（从数据库读取）")
    print(f"{'='*60}")

    # 获取音频记录
    date_info = f"（从 {start_date} 开始）" if start_date else ""
    print(f"正在从数据库获取 {limit} 条记录{date_info}...")
    records = fetch_audio_urls_from_db(limit, start_date, end_date)

    if not records:
        print("没有找到有效的音频记录")
        return

    total_records = len(records)
    print(f"获取到 {total_records} 条记录")
    print(f"时间范围: {records[0]['queue_time']} ~ {records[-1]['queue_time']}")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="asr_audio_")
    print(f"临时目录: {temp_dir}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = DIALOG_CSV_DIR  # 使用 config.py 中的配置
    os.makedirs(csv_dir, exist_ok=True)

    diar_json = os.path.join(script_dir, ".diarization_cache.json")

    try:
        t_start = time.time()

        # === 阶段1：下载音频 ===
        print(f"\n[阶段1/4] 下载音频...")
        print(f"下载策略: 30天内直接下载，30天前替换域名重试")
        audio_files = []
        download_failed = []
        skipped_csv = 0

        # 30天前的日期阈值
        threshold_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        for i, record in enumerate(records, 1):
            audio_file = record.get('audio_file', '')
            monitor_url = record.get('monitor_url', '')
            queue_time = record.get('queue_time', '')

            # 构建文件名
            filename = f"audio_{record['id']}.mp3"
            if monitor_url:
                parsed = urlparse(monitor_url)
                filename = os.path.basename(unquote(parsed.path))
                if not filename.endswith('.mp3'):
                    filename += '.mp3'

            # 检查 CSV 是否已存在（跳过已处理的文件）
            csv_name = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(csv_dir, csv_name)
            if os.path.exists(csv_path):
                skipped_csv += 1
                continue

            downloaded = False

            # 判断日期，决定下载方式
            is_recent = False
            if queue_time:
                try:
                    date_str = queue_time[:10] if len(queue_time) >= 10 else queue_time
                    is_recent = date_str >= threshold_date
                except:
                    is_recent = True  # 日期解析失败，按最近处理

            if is_recent:
                # 30天内：直接用 MonitorFileName
                if monitor_url:
                    print(f"  [{i}/{total_records}] 下载: {filename[:40]}... (http)", end="", flush=True)
                    try:
                        temp_path = download_audio_to_temp(monitor_url, temp_dir, filename)
                        audio_files.append((temp_path, record))
                        downloaded = True
                        print(" ✓")
                    except Exception as e:
                        print(f" ✗ 失败")
            else:
                # 30天前：替换域名后下载
                if monitor_url:
                    # 替换域名
                    # 原始: https://a6alipbxsh16.7x24cc.com/monitor/...
                    # 替换: https://storage.7x24cc.com/storage-server/presigned/ss1/a6-online-ass-recorder/monitor/...
                    new_url = monitor_url.replace(
                        "https://a6alipbxsh16.7x24cc.com/",
                        "https://storage.7x24cc.com/storage-server/presigned/ss1/a6-online-ass-recorder/"
                    )
                    print(f"  [{i}/{total_records}] 下载: {filename[:40]}... (storage)", end="", flush=True)
                    try:
                        temp_path = download_audio_to_temp(new_url, temp_dir, filename)
                        audio_files.append((temp_path, record))
                        downloaded = True
                        print(" ✓")
                    except Exception as e:
                        print(f" ✗ 失败")

            if not downloaded:
                download_failed.append(record['id'])

        if skipped_csv > 0:
            print(f"跳过已处理: {skipped_csv} 个文件")
        print(f"下载完成: {len(audio_files)} 成功, {len(download_failed)} 失败")

        if not audio_files:
            print("没有需要处理的新文件（全部已处理或下载失败）")
            return

        # === 阶段2：说话者分离 ===
        print(f"\n[阶段2/4] 说话者分离（本地 GPU）...")
        t_diar_start = time.time()

        diarize_script = os.path.join(script_dir, "diarize_worker.py")
        proc = subprocess.run(
            [sys.executable, diarize_script, temp_dir, diar_json],
            text=True,
        )
        if proc.returncode != 0:
            print("说话者分离失败")
            return

        t_diar_end = time.time()
        print(f"[阶段2/4] 完成，耗时 {t_diar_end - t_diar_start:.1f}s")

        with open(diar_json, encoding="utf-8") as f:
            all_speaker_segments = json.load(f)

        # === 阶段3：ASR 转录 ===
        print(f"\n[阶段3/4] ASR 转录（本地 GPU）...")

        free_mem, total_mem = torch.cuda.mem_get_info()
        safe_util = max(0.30, min(0.70, (free_mem - 8*1024**3) / total_mem))

        t_asr_load = time.time()
        asr_model = Qwen3ASRModel.LLM(
            model=ASR_MODEL_DIR,
            gpu_memory_utilization=safe_util,
            max_inference_batch_size=4,  # 减小推理批次避免内存溢出
            max_new_tokens=4096,
            forced_aligner=ALIGNER_MODEL_DIR,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
        )
        print(f"[模型加载] ASR 模型加载完成，耗时 {time.time() - t_asr_load:.1f}s")

        t_asr_start = time.time()
        all_transcripts = {}
        file_paths = [f[0] for f in audio_files]
        total_batches = (len(file_paths) + ASR_BATCH_SIZE - 1) // ASR_BATCH_SIZE

        for batch_idx, batch_start in enumerate(range(0, len(file_paths), ASR_BATCH_SIZE), 1):
            batch_files = file_paths[batch_start: batch_start + ASR_BATCH_SIZE]
            print(f"  批次 [{batch_idx}/{total_batches}] {len(batch_files)} 个文件...", flush=True)

            batch_results = asr_model.transcribe(
                audio=batch_files,
                language=["Chinese"] * len(batch_files),
                return_time_stamps=True,
            )

            for audio_file, r in zip(batch_files, batch_results):
                segments = all_speaker_segments.get(audio_file, [])
                audio_name = os.path.splitext(os.path.basename(audio_file))[0]

                lines = [f"语言: {r.language}"]
                current_speaker = None
                current_text = []
                for item in r.time_stamps:
                    speaker = find_speaker(item.start_time, item.end_time, segments)
                    if speaker != current_speaker:
                        if current_speaker and current_text:
                            lines.append(f"[{current_speaker}]: {''.join(current_text)}")
                        current_speaker = speaker
                        current_text = [item.text]
                    else:
                        current_text.append(item.text)
                if current_speaker and current_text:
                    lines.append(f"[{current_speaker}]: {''.join(current_text)}")

                transcript_text = "\n".join(lines)
                all_transcripts[audio_name] = transcript_text

        t_asr_end = time.time()
        print(f"[阶段3/4] ASR 完成，耗时 {t_asr_end - t_asr_start:.1f}s")

        # 释放 ASR 模型
        del asr_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # === 阶段4：对话分类 ===
        print(f"\n[阶段4/4] 对话分类（API {api_workers}线程并发）...")
        t_api_start = time.time()

        audio_names = list(all_transcripts.keys())
        total_csv = 0
        failed_files = []

        progress_counter = 0
        total_tokens_input = 0
        total_tokens_output = 0

        with ThreadPoolExecutor(max_workers=api_workers) as executor:
            futures = {
                executor.submit(process_single_dialog, name, all_transcripts[name], csv_dir, len(audio_names)): name
                for name in audio_names
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        total_csv += 1
                    else:
                        failed_files.append(futures[future])
                except:
                    failed_files.append(futures[future])

        t_api_end = time.time()

        # 统计
        total_elapsed = time.time() - t_start
        total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE

        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"处理记录:         {len(audio_files)} 条")
        print(f"下载失败:         {len(download_failed)} 条")
        print(f"-" * 60)
        print(f"总耗时:           {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
        print(f"CSV 文件:         {total_csv} 个 → {csv_dir}")
        print(f"API 成本:         ¥{total_cost:.4f}")
        print(f"{'='*60}")

    finally:
        # 清理临时目录
        if keep_temp:
            print(f"\n保留临时目录: {temp_dir}")
        else:
            print(f"\n清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_from_local(input_path, api_workers):
    """从本地目录处理"""
    audio_files = get_audio_files(input_path)
    if not audio_files:
        return

    total_files = len(audio_files)
    print(f"\n{'='*60}")
    print(f"ASR 处理流水线（本地目录）")
    print(f"{'='*60}")
    print(f"输入文件: {total_files} 个")
    print(f"API 并发: {api_workers} 线程")
    print(f"{'='*60}\n")

    # 统计音频总时长
    print("正在统计音频时长...")
    total_audio_duration = 0.0
    for i, f in enumerate(audio_files, 1):
        dur = get_audio_duration(f)
        total_audio_duration += dur
        if i % 50 == 0 or i == total_files:
            print(f"  已统计 {i}/{total_files} 个文件", end="\r")
    print()

    avg_duration = total_audio_duration / total_files if total_files > 0 else 0
    print(f"音频总时长: {format_duration(total_audio_duration)}")
    print(f"平均时长:   {format_duration(avg_duration)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = DIALOG_CSV_DIR  # 使用 config.py 中的配置
    os.makedirs(csv_dir, exist_ok=True)

    diar_json = os.path.join(script_dir, ".diarization_cache.json")
    t_start = time.time()

    # 阶段1：说话者分离
    print(f"\n[阶段1/3] 说话者分离...")
    t_diar_start = time.time()
    diarize_script = os.path.join(script_dir, "diarize_worker.py")
    subprocess.run([sys.executable, diarize_script, input_path, diar_json], text=True, check=True)
    t_diar_end = time.time()
    print(f"[阶段1/3] 完成，耗时 {t_diar_end - t_diar_start:.1f}s")

    with open(diar_json, encoding="utf-8") as f:
        all_speaker_segments = json.load(f)

    # 阶段2：ASR 转录
    print(f"\n[阶段2/3] ASR 转录...")
    free_mem, total_mem = torch.cuda.mem_get_info()
    safe_util = max(0.30, min(0.70, (free_mem - 8*1024**3) / total_mem))

    t_asr_load = time.time()
    asr_model = Qwen3ASRModel.LLM(
        model=ASR_MODEL_DIR,
        gpu_memory_utilization=safe_util,
        max_inference_batch_size=8,
        max_new_tokens=4096,
        forced_aligner=ALIGNER_MODEL_DIR,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    print(f"[模型加载] ASR 模型加载完成，耗时 {time.time() - t_asr_load:.1f}s")

    t_asr_start = time.time()
    all_transcripts = {}
    total_batches = (total_files + ASR_BATCH_SIZE - 1) // ASR_BATCH_SIZE

    for batch_idx, batch_start in enumerate(range(0, total_files, ASR_BATCH_SIZE), 1):
        batch_files = audio_files[batch_start: batch_start + ASR_BATCH_SIZE]
        print(f"  批次 [{batch_idx}/{total_batches}] {len(batch_files)} 个文件...", flush=True)

        batch_results = asr_model.transcribe(
            audio=batch_files, language=["Chinese"] * len(batch_files), return_time_stamps=True,
        )

        for audio_file, r in zip(batch_files, batch_results):
            segments = all_speaker_segments.get(audio_file, [])
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]

            lines = [f"语言: {r.language}"]
            current_speaker = None
            current_text = []
            for item in r.time_stamps:
                speaker = find_speaker(item.start_time, item.end_time, segments)
                if speaker != current_speaker:
                    if current_speaker and current_text:
                        lines.append(f"[{current_speaker}]: {''.join(current_text)}")
                    current_speaker = speaker
                    current_text = [item.text]
                else:
                    current_text.append(item.text)
            if current_speaker and current_text:
                lines.append(f"[{current_speaker}]: {''.join(current_text)}")

            transcript_text = "\n".join(lines)
            all_transcripts[audio_name] = transcript_text

    t_asr_end = time.time()
    print(f"[阶段2/3] ASR 完成，耗时 {t_asr_end - t_asr_start:.1f}s")

    del asr_model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 阶段3：对话分类
    print(f"\n[阶段3/3] 对话分类（API {api_workers}线程并发）...")
    t_api_start = time.time()

    audio_names = list(all_transcripts.keys())
    total_csv = 0
    failed_files = []

    global progress_counter, total_tokens_input, total_tokens_output
    progress_counter = 0
    total_tokens_input = 0
    total_tokens_output = 0

    with ThreadPoolExecutor(max_workers=api_workers) as executor:
        futures = {
            executor.submit(process_single_dialog, name, all_transcripts[name], csv_dir, len(audio_names)): name
            for name in audio_names
        }
        for future in as_completed(futures):
            try:
                if future.result():
                    total_csv += 1
                else:
                    failed_files.append(futures[future])
            except:
                failed_files.append(futures[future])

    t_api_end = time.time()

    # 统计
    total_elapsed = time.time() - t_start
    total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE

    print(f"\n{'='*60}")
    print("处理完成统计")
    print(f"{'='*60}")
    print(f"文件数:           {total_files}")
    print(f"音频总时长:       {format_duration(total_audio_duration)}")
    print(f"总耗时:           {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
    print(f"CSV 文件:         {total_csv} 个 → {csv_dir}")
    print(f"API 成本:         ¥{total_cost:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ASR 处理流水线')
    parser.add_argument('--from-db', type=int, metavar='LIMIT',
                        help='从数据库读取指定数量的记录')
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)，默认从最早开始')
    parser.add_argument('--end-date', type=str, default=None,
                        help='截止日期 (YYYY-MM-DD)，下载此日期之前的数据')
    parser.add_argument('--api-workers', type=int, default=API_MAX_WORKERS,
                        help=f'API 并发线程数，默认 {API_MAX_WORKERS}')
    parser.add_argument('--keep-temp', action='store_true',
                        help='保留临时目录（不自动清理），便于重新处理')
    parser.add_argument('input_path', nargs='?', default=DEFAULT_INPUT,
                        help=f'本地音频目录，默认 {DEFAULT_INPUT}')

    args = parser.parse_args()

    if args.from_db:
        # 从数据库读取
        process_from_db(args.from_db, args.api_workers, args.start_date, args.end_date, args.keep_temp)
    else:
        # 从本地目录读取
        process_from_local(args.input_path, args.api_workers)