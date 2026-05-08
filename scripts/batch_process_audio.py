#!/usr/bin/env python3
"""分批处理音频文件

将大量音频文件分成小批次处理，避免一次性处理失败导致进度丢失。
每批处理完成后生成CSV，出错只需重跑该批。

使用方式:
  python batch_process_audio.py ./mp3 500 20  # 每批500个，20线程
  python batch_process_audio.py ./mp3 500 20 --skip-diarization
"""

import os
import sys
import shutil
import glob
import argparse
from pathlib import Path

# 导入路径配置
from config import TEMP_DIR

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def get_audio_files(audio_dir):
    """获取音频文件列表"""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(audio_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(audio_dir, f"*{ext.upper()}")))
    return sorted(files)


def run_batch(batch_dir, api_workers, skip_diarization):
    """运行单批处理"""
    import subprocess
    process_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "process_audio.py")

    cmd = [sys.executable, process_script, batch_dir, str(api_workers)]
    if skip_diarization:
        cmd.append("--skip-diarization")

    # 捕获输出，显示错误信息
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="分批处理音频文件")
    parser.add_argument("audio_dir", help="音频文件目录")
    parser.add_argument("batch_size", type=int, default=500, help="每批文件数")
    parser.add_argument("api_workers", type=int, default=20, help="API并发线程数")
    parser.add_argument("--skip-diarization", action="store_true", help="跳过说话者分离")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(f"目录不存在: {audio_dir}")
        sys.exit(1)

    # 获取所有音频文件
    all_files = get_audio_files(audio_dir)
    total = len(all_files)

    if total == 0:
        print("没有找到音频文件")
        return

    batch_size = args.batch_size
    total_batches = (total + batch_size - 1) // batch_size

    print(f"\n{'='*60}")
    print("分批音频处理")
    print(f"{'='*60}")
    print(f"总文件数: {total}")
    print(f"每批大小: {batch_size}")
    print(f"总批次数: {total_batches}")
    print(f"跳过分离: {'是' if args.skip_diarization else '否'}")
    print(f"{'='*60}\n")

    # 创建临时批次目录
    batch_base_dir = TEMP_DIR

    success_batches = 0
    failed_batches = []

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_files = all_files[batch_start:batch_end]

        batch_dir = os.path.join(batch_base_dir, f"batch_{batch_idx:04d}")

        print(f"\n[批次 {batch_idx+1}/{total_batches}] 文件 {batch_start+1}-{batch_end}")

        # 创建批次目录并复制文件（用符号链接节省空间）
        os.makedirs(batch_dir, exist_ok=True)
        for f in batch_files:
            src = os.path.abspath(f)
            dst = os.path.join(batch_dir, os.path.basename(f))
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        # 运行处理
        success = run_batch(batch_dir, args.api_workers, args.skip_diarization)

        if success:
            success_batches += 1
            print(f"[批次 {batch_idx+1}/{total_batches}] ✓ 完成")
            # 清理批次目录
            shutil.rmtree(batch_dir, ignore_errors=True)
        else:
            failed_batches.append(batch_idx + 1)
            print(f"[批次 {batch_idx+1}/{total_batches}] ✗ 失败")
            # 保留失败批次目录，方便手动重跑

    # 统计
    print(f"\n{'='*60}")
    print("分批处理完成")
    print(f"{'='*60}")
    print(f"成功批次: {success_batches}/{total_batches}")
    if failed_batches:
        print(f"失败批次: {failed_batches}")
        print(f"\n重跑失败批次命令:")
        for batch_idx in failed_batches:
            batch_dir = os.path.join(batch_base_dir, f"batch_{batch_idx-1:04d}")
            cmd = f"python process_audio.py {batch_dir} {args.api_workers}"
            if args.skip_diarization:
                cmd += " --skip-diarization"
            print(f"  {cmd}")
    else:
        # 清理临时目录
        shutil.rmtree(batch_base_dir, ignore_errors=True)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()