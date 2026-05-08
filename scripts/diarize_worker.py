"""
独立子进程：用 pyannote 在 GPU 上做说话者分离，结果保存为 JSON，退出后 VRAM 完全释放。
用法: python diarize_worker.py <audio_dir_or_file> <output_json>

支持断点续传：
- 每处理完一个文件就保存结果
- 启动时检查已有缓存，跳过已处理的文件
"""
import os
import sys
import json
import time
import queue
import threading

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for _k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(_k, None)

import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed")

import torch
import numpy as np
import librosa
from pyannote.audio import Pipeline

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_audio_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    return [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]


def save_results(results, output_json):
    """保存结果到 JSON 文件"""
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def load_existing_results(output_json):
    """加载已有的结果（用于断点续传）"""
    if os.path.exists(output_json):
        try:
            with open(output_json, encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def main():
    input_path = sys.argv[1]
    output_json = sys.argv[2]

    audio_files = get_audio_files(input_path)
    total = len(audio_files)
    print(f"[diarize] 共 {total} 个文件", flush=True)

    # 加载已有结果（断点续传）
    results = load_existing_results(output_json)
    already_done = set(results.keys())
    pending_files = [f for f in audio_files if f not in already_done]

    if already_done:
        print(f"[diarize] 已有 {len(already_done)} 个文件结果，跳过", flush=True)
        print(f"[diarize] 待处理 {len(pending_files)} 个文件", flush=True)

    if not pending_files:
        print(f"[diarize] 全部文件已处理完成，无需重新处理", flush=True)
        return

    print("[diarize] 加载 pyannote 说话者分离模型（GPU）...", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    pipeline.to(torch.device("cuda:0"))

    # 用线程预加载音频，掩盖 librosa IO 等待
    load_queue = queue.Queue(maxsize=4)

    def audio_loader():
        for f in pending_files:
            audio, _ = librosa.load(f, sr=16000, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            waveform = torch.from_numpy(audio)
            load_queue.put((f, waveform))
        load_queue.put(None)

    loader_thread = threading.Thread(target=audio_loader, daemon=True)
    loader_thread.start()

    t_start = time.time()
    i = 0
    pending_total = len(pending_files)

    while True:
        item = load_queue.get()
        if item is None:
            break
        audio_file, waveform = item
        i += 1

        try:
            # 先尝试自动识别
            diar_result = pipeline({"waveform": waveform, "sample_rate": 16000})
            annotation = diar_result.speaker_diarization if hasattr(diar_result, 'speaker_diarization') else diar_result
            segments = [[round(t.start, 3), round(t.end, 3), spk]
                        for t, _, spk in annotation.itertracks(yield_label=True)]

            # 如果只识别出1个说话者，强制识别2个
            num_speakers = len(set(s[2] for s in segments))
            if num_speakers == 1:
                # 使用正确的参数格式强制指定说话者数量
                diar_result = pipeline({"waveform": waveform, "sample_rate": 16000}, num_speakers=2)
                annotation = diar_result.speaker_diarization if hasattr(diar_result, 'speaker_diarization') else diar_result
                new_segments = [[round(t.start, 3), round(t.end, 3), spk]
                            for t, _, spk in annotation.itertracks(yield_label=True)]
                # 只有当强制识别确实产生了2个说话者时才更新
                if len(set(s[2] for s in new_segments)) >= 2:
                    segments = new_segments

            results[audio_file] = segments

            # 每处理完一个文件就保存（断点续传）
            save_results(results, output_json)

            elapsed = time.time() - t_start
            eta = elapsed / i * (pending_total - i)
            num_spk = len(set(s[2] for s in segments))
            print(f"  [{len(already_done) + i}/{total}] {os.path.basename(audio_file)[:40]}  "
                  f"说话者:{num_spk}  已用:{elapsed:.0f}s  预计剩余:{eta:.0f}s", flush=True)

        except Exception as e:
            print(f"  [{len(already_done) + i}/{total}] {os.path.basename(audio_file)[:40]}  "
                  f"错误: {str(e)[:30]}", flush=True)
            # 即使出错也保存当前结果
            save_results(results, output_json)

    total_time = time.time() - t_start
    print(f"[diarize] 完成，耗时 {total_time:.1f}s，本次处理 {pending_total} 个文件", flush=True)
    print(f"[diarize] 总计 {len(results)} 个文件结果已保存到 {output_json}", flush=True)


if __name__ == '__main__':
    main()
