"""
音频处理脚本（不含下载）

用于处理已下载的音频文件：
1. 说话者分离
2. ASR 转录
3. 对话分类（API）
4. 保存 CSV

使用方式:
  python process_audio.py /path/to/audio        # 处理指定目录
  python process_audio.py /path/to/audio 20     # 指定API并发线程数
"""

import os
import sys
import time
import json
import subprocess
import csv
import re
import tempfile
import shutil
import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse, unquote
from glob import glob

os.environ["HF_HUB_OFFLINE"] = "1"

# ==================== 配置 ====================
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
ASR_BATCH_SIZE = 5  # 批量大小，降低以减少显存压力

# 导入路径配置
from config import CACHE_DIR, DIALOG_CSV_DIR, ASR_MODEL_DIR, ALIGNER_MODEL_DIR

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"

# 计费标准（Qwen3-8B）
INPUT_PRICE = 0.0005   # ¥/千Token
OUTPUT_PRICE = 0.002   # ¥/千Token

# 输出目录（已从config导入：DIALOG_CSV_DIR, CACHE_DIR）
CSV_DIR = DIALOG_CSV_DIR
DIAR_CACHE_DIR = CACHE_DIR

# 脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== Prompt ====================
SYSTEM_PROMPT = """下面是一段客服电话录音的转录内容，已标注说话者（如 [客服]、[客户]）。
请将对话按角色分类，输出为 CSV 格式，包含两列：角色、内容。

规则：
1. 角色只能是"客服"或"客户"
2. 内容是该角色说的完整句子
3. 每行一句，保持原始对话顺序
4. 如果无法判断角色，根据上下文推断

只输出 CSV 内容，不要解释。第一行为表头：角色,内容"""

# ==================== 辅助函数 ====================

def find_speaker(start, end, segments):
    """根据时间戳找说话者

    segments 格式: [[start, end, speaker], ...]
    """
    if not segments:
        return "客服"  # 默认

    max_overlap = 0
    speaker = "客服"  # 默认
    for seg in segments:
        # seg 格式: [start, end, speaker]
        if isinstance(seg, dict):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_speaker = seg.get("speaker", "客服")
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            seg_start, seg_end, seg_speaker = seg[0], seg[1], seg[2]
        else:
            continue

        overlap = max(0, min(end, seg_end) - max(start, seg_start))
        if overlap > max_overlap:
            max_overlap = overlap
            speaker = seg_speaker

    # 转换 speaker 标签
    # SPEAKER_00 -> 客服, SPEAKER_01 -> 客户 (假设第一个说话者是客服)
    if speaker.startswith("SPEAKER_"):
        try:
            idx = int(speaker.split("_")[1])
            return "客服" if idx == 0 else "客户"
        except:
            return "客服"

    return speaker


# ==================== API 调用 ====================

progress_lock = Lock()
progress_counter = 0
total_tokens_input = 0
total_tokens_output = 0


def process_single_dialog(audio_name, transcript, csv_dir, total_files):
    """处理单个对话"""
    global progress_counter, total_tokens_input, total_tokens_output

    csv_path = os.path.join(csv_dir, f"{audio_name}.csv")

    # 已存在则跳过
    if os.path.exists(csv_path):
        with progress_lock:
            progress_counter += 1
        return True

    prompt = f"{SYSTEM_PROMPT}\n\n对话内容：\n{transcript}\n\n请输出 CSV 格式："

    # API重试机制
    max_retries = 3
    last_error = None
    for retry in range(max_retries):
        try:
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
                    "max_tokens": 4096,
                    "enable_thinking": False,  # 必须禁用 thinking 模式
                },
                timeout=120,  # 增加超时时间到120秒
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            with progress_lock:
                progress_counter += 1
                total_tokens_input += usage.get("prompt_tokens", 0)
                total_tokens_output += usage.get("completion_tokens", 0)

            # 解析 CSV
            lines = content.strip().split("\n")
            csv_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("```"):
                    continue
                if line.startswith("角色") or line.startswith("说话者"):
                    continue
                # 清理行号
                line = re.sub(r"^\d+[\.\、\s]*", "", line)
                if line:
                    csv_lines.append(line)

            if not csv_lines:
                return False

            # 保存 CSV
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                f.write("角色,内容\n")
                for line in csv_lines:
                    f.write(line + "\n")

            print(f"  [{progress_counter}/{total_files}] ✓ {audio_name[:40]}... | 对话:{len(csv_lines)}")
            return True

        except Exception as e:
            last_error = e
            if retry < max_retries - 1:
                print(f"  [{audio_name[:30]}] 重试 {retry+2}/{max_retries}...", flush=True)
                time.sleep(2)  # 等待2秒后重试
            continue

    # 所有重试都失败
    with progress_lock:
        progress_counter += 1
    print(f"  [{progress_counter}/{total_files}] ✗ {audio_name[:40]}... | 错误: {str(last_error)[:30]}")
    return False


# ==================== 主流程 ====================

def process_audio_dir(audio_dir, api_workers=10, skip_diarization=False, limit=None, reuse_diarization=False):
    """处理音频目录

    Args:
        audio_dir: 音频文件目录
        api_workers: API 并发线程数
        skip_diarization: 是否跳过说话者分离阶段
        limit: 限制处理文件数量（用于测试）
        reuse_diarization: 是否复用已有的分离结果
    """
    print(f"\n{'='*60}")
    print("音频处理流水线")
    if skip_diarization:
        print("⚠️  已跳过说话者分离阶段，输出无角色标注")
    if limit:
        print(f"⚠️  限制处理文件数量: {limit} 个（测试模式）")
    if reuse_diarization:
        print("⚠️  复用已有的分离结果")
    print(f"{'='*60}")

    # 收集音频文件
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(glob(os.path.join(audio_dir, f"*{ext}")))
        audio_files.extend(glob(os.path.join(audio_dir, f"*{ext.upper()}")))

    if not audio_files:
        print(f"目录 {audio_dir} 中没有找到音频文件")
        return

    audio_files = sorted(audio_files)

    # 限制文件数量
    if limit and limit > 0:
        audio_files = audio_files[:limit]

    total_files = len(audio_files)
    print(f"音频文件: {total_files} 个")
    print(f"API 并发: {api_workers} 线程")
    print(f"{'='*60}\n")

    # 临时目录
    temp_dir = tempfile.mkdtemp(prefix="asr_process_")
    diar_json = os.path.join(temp_dir, "diarization.json")

    # 如果有 limit，创建临时音频目录（用符号链接）
    work_audio_dir = audio_dir
    work_audio_files = audio_files  # 用于 ASR 处理的文件列表
    if limit and limit > 0 and not skip_diarization:
        work_audio_dir = os.path.join(temp_dir, "audio_links")
        os.makedirs(work_audio_dir, exist_ok=True)
        work_audio_files = []  # 符号链接路径列表
        for f in audio_files:
            src = os.path.abspath(f)
            dst = os.path.join(work_audio_dir, os.path.basename(f))
            os.symlink(src, dst)
            work_audio_files.append(dst)
        print(f"创建临时目录: {work_audio_dir} ({len(work_audio_files)} 个符号链接)")

    try:
        t_start = time.time()

        # === 阶段1：说话者分离 ===
        if skip_diarization:
            print(f"[阶段1/3] 已跳过说话者分离")
            all_speaker_segments = {}  # 空字典，后续使用默认角色
        else:
            # 分离结果缓存文件路径
            os.makedirs(DIAR_CACHE_DIR, exist_ok=True)
            audio_dir_name = os.path.basename(os.path.normpath(audio_dir))
            cache_key = f"{audio_dir_name}_{limit if limit else 'all'}"
            diar_cache_file = os.path.join(DIAR_CACHE_DIR, f"{cache_key}.json")

            # 检查是否已有缓存
            if reuse_diarization and os.path.exists(diar_cache_file):
                print(f"[阶段1/3] 复用已有分离结果: {diar_cache_file}")
                with open(diar_cache_file, encoding="utf-8") as f:
                    all_speaker_segments = json.load(f)
                print(f"[阶段1/3] 已加载 {len(all_speaker_segments)} 个文件的分离数据")
            else:
                print(f"[阶段1/3] 说话者分离（本地 GPU）...")
                t_diar_start = time.time()

                # 直接把缓存文件作为输出，支持断点续传
                diarize_script = os.path.join(SCRIPT_DIR, "diarize_worker.py")
                proc = subprocess.run(
                    [sys.executable, diarize_script, work_audio_dir, diar_cache_file],  # 直接输出到缓存文件
                    text=True,
                )
                if proc.returncode != 0:
                    print("说话者分离失败")
                    return

                t_diar_end = time.time()
                print(f"[阶段1/3] 完成，耗时 {t_diar_end - t_diar_start:.1f}s")

                with open(diar_cache_file, encoding="utf-8") as f:
                    all_speaker_segments = json.load(f)
                print(f"[阶段1/3] 分离结果已保存: {diar_cache_file}")

        # === 阶段2：ASR 转录 ===
        print(f"\n[阶段2/3] ASR 转录（本地 GPU）...")

        # 强制清理 GPU 显存（确保分离模型完全释放）
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 动态加载模型
        from qwen_asr import Qwen3ASRModel

        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"可用显存: {free_mem/(1024**3):.2f} GiB / {total_mem/(1024**3):.2f} GiB")

        # 更保守的显存利用率
        safe_util = max(0.30, min(0.50, (free_mem - 12*1024**3) / total_mem))

        t_asr_load = time.time()
        asr_model = Qwen3ASRModel.LLM(
            model=ASR_MODEL_DIR,
            gpu_memory_utilization=0.4,  # 更保守的显存利用率
            max_inference_batch_size=2,  # 降低到 2，最小批次
            max_new_tokens=2048,  # 降低输出长度
            max_model_len=4096,  # 降低到 4096，最小 KV cache
            forced_aligner=ALIGNER_MODEL_DIR,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
        )
        print(f"[模型加载] ASR 模型加载完成，耗时 {time.time() - t_asr_load:.1f}s")

        # 使用工作目录的文件列表
        file_paths = work_audio_files

        # 分类音频文件：短音频正常批次，长音频单独处理
        print(f"[音频分类] 检查音频时长...")
        normal_files = []  # 正常时长音频
        long_files = []    # 长音频（>600秒）
        LONG_AUDIO_THRESHOLD = 600  # 长音频阈值（秒）

        import librosa
        for f in file_paths:
            try:
                duration = librosa.get_duration(filename=f)
                if duration > LONG_AUDIO_THRESHOLD:
                    long_files.append(f)
                else:
                    normal_files.append(f)
            except:
                normal_files.append(f)  # 无法读取时长则归为正常

        print(f"[音频分类] 正常音频: {len(normal_files)} 个")
        print(f"[音频分类] 长音频(>{LONG_AUDIO_THRESHOLD}s): {len(long_files)} 个")
        if long_files:
            print(f"[音频分类] 长音频将单独处理（批次大小=1）")

        t_asr_start = time.time()
        all_transcripts = {}

        # 先处理正常音频（批次处理）
        if normal_files:
            normal_batches = (len(normal_files) + ASR_BATCH_SIZE - 1) // ASR_BATCH_SIZE
            print(f"\n[ASR] 处理正常音频，共 {len(normal_files)} 个，分 {normal_batches} 批")

            for batch_idx, batch_start in enumerate(range(0, len(normal_files), ASR_BATCH_SIZE), 1):
                batch_files = normal_files[batch_start: batch_start + ASR_BATCH_SIZE]
                print(f"  批次 [{batch_idx}/{normal_batches}] {len(batch_files)} 个文件...", flush=True)

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

                # 每批次后清理显存碎片
                torch.cuda.empty_cache()
                if batch_idx % 5 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"    显存状态: {free_mem/(1024**3):.1f} GiB 可用", flush=True)

        # 再处理长音频（单独处理，批次大小=1）
        if long_files:
            print(f"\n[ASR] 处理长音频，共 {len(long_files)} 个，单独处理")
            for i, audio_file in enumerate(long_files, 1):
                print(f"  长音频 [{i}/{len(long_files)}] {os.path.basename(audio_file)[:40]}...", flush=True)

                try:
                    result = asr_model.transcribe(
                        audio=[audio_file],
                        language=["Chinese"],
                        return_time_stamps=True,
                    )
                    r = result[0]
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

                except Exception as e:
                    print(f"    处理失败: {str(e)[:50]}", flush=True)
                    # 长音频失败不影响其他文件，继续处理下一个

                # 每个长音频后清理显存
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        t_asr_end = time.time()
        print(f"[阶段2/3] ASR 完成，耗时 {t_asr_end - t_asr_start:.1f}s")

        # 释放 ASR 模型
        del asr_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # === 阶段3：对话分类 ===
        print(f"\n[阶段3/3] 对话分类（API {api_workers}线程并发）...")
        t_api_start = time.time()

        os.makedirs(CSV_DIR, exist_ok=True)

        audio_names = list(all_transcripts.keys())
        total_csv = 0
        failed_files = []

        # 重置计数器
        progress_counter = 0
        total_tokens_input = 0
        total_tokens_output = 0

        with ThreadPoolExecutor(max_workers=api_workers) as executor:
            futures = {
                executor.submit(process_single_dialog, name, all_transcripts[name], CSV_DIR, len(audio_names)): name
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
        total_cost = (total_tokens_input / 1000) * INPUT_PRICE + (total_tokens_output / 1000) * OUTPUT_PRICE
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"音频文件:       {total_files} 个")
        print(f"CSV 成功:       {total_csv} 个")
        print(f"失败:           {len(failed_files)} 个")
        print(f"总耗时:         {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
        print(f"CSV 目录:       {CSV_DIR}")
        print(f"API 成本:       ¥{total_cost:.2f}")
        print(f"{'='*60}")

    finally:
        # 清理临时目录
        print(f"\n清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="音频处理脚本")
    parser.add_argument("audio_dir", help="音频文件目录")
    parser.add_argument("api_workers", type=int, nargs="?", default=10, help="API并发线程数")
    parser.add_argument("--skip-diarization", action="store_true", help="跳过说话者分离阶段")
    parser.add_argument("--limit", type=int, default=None, help="限制处理文件数量（测试用）")
    parser.add_argument("--reuse-diarization", action="store_true", help="复用已有的分离结果缓存")
    args = parser.parse_args()

    if not os.path.isdir(args.audio_dir):
        print(f"目录不存在: {args.audio_dir}")
        sys.exit(1)

    process_audio_dir(args.audio_dir, args.api_workers, args.skip_diarization, args.limit, args.reuse_diarization)