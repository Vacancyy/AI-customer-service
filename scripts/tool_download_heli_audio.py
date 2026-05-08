#!/usr/bin/env python3
"""独立下载 heli 录音文件的脚本。

这个脚本不依赖项目里的 Django 配置和模型，直接连接本地 dev
环境的 SQLite 数据库读取 heli_audio 表中的录音 URL 并下载到本地目录。

下载策略：
- 30天内：直接用 MonitorFileName 原地址下载
- 30天前：替换域名为 storage.7x24cc.com 下载
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

import requests

# OSS 配置
OSS_ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET_NAME = "njzhyl-insurance-claim"
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"  # 公网地址
OSS_MEDIA_PREFIX = "invoiceRecogMedia/prod"

# 尝试导入 oss2
try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False


TABLE_NAME = "heli_audio"
DOWNLOAD_TIMEOUT = 60
DEFAULT_TARGET_DIR = "02_download/audio"

# 导入路径配置
try:
    from config import DB_PATH, AUDIO_DIR
except ImportError:
    # 如果config.py不存在，使用默认路径
    _ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH = os.path.join(_ROOT, "01_source/heli.sqlite3")
    AUDIO_DIR = os.path.join(_ROOT, "02_download/audio")

# 新旧域名映射
OLD_DOMAIN = "https://a6alipbxsh16.7x24cc.com/"
NEW_DOMAIN = "https://storage.7x24cc.com/storage-server/presigned/ss1/a6-online-ass-recorder/"


@dataclass
class AudioRow:
    audio_id: int
    monitor_filename: str
    audio_file: str
    call_time_length: int
    queue_time: str


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从本地 heli_audio 表下载录音文件到本地目录。"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=DEFAULT_TARGET_DIR,
        help=f"录音文件保存目录，默认：{DEFAULT_TARGET_DIR}",
    )
    parser.add_argument(
        "--success-limit",
        type=int,
        default=None,
        help="期望成功下载的条数，用于抽样预估；不传则尽量下载全部。",
    )
    parser.add_argument(
        "--use-oss",
        action="store_true",
        help="优先使用 AudioFile OSS 路径下载（需安装 oss2 库）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期（含），格式 YYYY-MM-DD，基于 QueueTime 字段",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期（含），格式 YYYY-MM-DD，基于 QueueTime 字段",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库文件不存在：{DB_PATH}")
    return sqlite3.connect(DB_PATH, timeout=30)


def fetch_audio_rows(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    order_desc: bool = True
) -> list[AudioRow]:
    """获取音频记录，按 URL 去重

    Args:
        start_date: 开始日期
        end_date: 结束日期
        order_desc: 是否按时间倒序（最新优先）
    """

    order_clause = "ORDER BY QueueTime DESC" if order_desc else "ORDER BY QueueTime ASC"

    sql = f"""
        SELECT id, MonitorFilename, AudioFile, CallTimeLength, QueueTime
        FROM {TABLE_NAME}
        WHERE id IS NOT NULL
          AND CallTimeLength IS NOT NULL
          AND CallTimeLength <> 0
          AND MonitorFilename IS NOT NULL AND MonitorFilename != ''
          AND AudioFile IS NOT NULL AND AudioFile != '' AND AudioFile != 'normal'
          AND QueueTime IS NOT NULL
          AND (
              :start_date IS NULL
              OR date(QueueTime) >= date(:start_date)
          )
          AND (
              :end_date IS NULL
              OR date(QueueTime) <= date(:end_date)
          )
        {order_clause}
    """

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, {"start_date": start_date, "end_date": end_date})
        rows = cursor.fetchall()
    finally:
        conn.close()

    # 去重：优先用 MonitorFilename 去重，没有则用 AudioFile
    seen_urls = set()
    result: list[AudioRow] = []

    for audio_id, monitor_filename, audio_file, call_time_length, queue_time in rows:
        # 构建唯一键用于去重
        dedup_key = monitor_filename or audio_file or f"id_{audio_id}"

        if dedup_key in seen_urls:
            continue
        seen_urls.add(dedup_key)

        result.append(
            AudioRow(
                audio_id=int(audio_id),
                monitor_filename=str(monitor_filename or "").strip(),
                audio_file=str(audio_file or "").strip(),
                call_time_length=int(call_time_length),
                queue_time=queue_time or "",
            )
        )

    return result


def build_filename(audio: AudioRow) -> str:
    """构建保存的文件名"""
    filename = ""

    # 优先从 AudioFile 提取
    if audio.audio_file:
        filename = os.path.basename(unquote(audio.audio_file))

    # 其次从 MonitorFilename 提取
    if not filename and audio.monitor_filename:
        raw_url_path = urlparse(audio.monitor_filename).path
        filename = os.path.basename(unquote(raw_url_path))

    # 最后用 id
    if not filename:
        filename = f"audio_{audio.audio_id}.mp3"

    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".mp3"
    return f"{name}{ext}"


def is_oss_path(path: str) -> bool:
    """判断是否是 OSS 路径"""
    if not path:
        return False
    # OSS 路径特征
    oss_keywords = ['oss', 'invoiceRecogMedia', 'aliyuncs.com']
    return any(kw in path.lower() for kw in oss_keywords)


def build_download_url(monitor_url: str, queue_time: str) -> str:
    """根据日期构建下载 URL

    - 30天内：直接用原地址
    - 30天前：替换域名
    """
    # 计算30天前的日期
    threshold_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 解析日期
    if queue_time and len(queue_time) >= 10:
        record_date = queue_time[:10]
    else:
        record_date = threshold_date  # 默认按最近处理

    if record_date >= threshold_date:
        # 30天内：直接用原地址
        return monitor_url
    else:
        # 30天前：替换域名
        return monitor_url.replace(OLD_DOMAIN, NEW_DOMAIN)


def download_file(session: requests.Session, url: str, dst_path: str) -> int:
    """下载文件，返回字节数"""
    with session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        total = 0
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
        return total


def get_oss_bucket():
    """获取 OSS Bucket 对象"""
    if not OSS_AVAILABLE:
        raise ImportError("oss2 库未安装，请运行: pip install oss2")

    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
    return bucket


def download_from_oss(audio_file: str, dst_path: str) -> int:
    """从 OSS 下载文件，返回字节数

    Args:
        audio_file: OSS 相对路径，如 'invoiceRecogMedia/prod/xxx.mp3'
        dst_path: 本地保存路径
    """
    bucket = get_oss_bucket()

    # 处理路径：移除可能的前缀
    oss_key = audio_file
    if oss_key.startswith('/'):
        oss_key = oss_key[1:]

    # 如果路径不包含 invoiceRecogMedia，添加前缀
    if not oss_key.startswith(OSS_MEDIA_PREFIX):
        oss_key = f"{OSS_MEDIA_PREFIX}/{oss_key}"

    bucket.get_object_to_file(oss_key, dst_path)

    # 返回文件大小
    return os.path.getsize(dst_path)


def download_audios(
    target_dir: str,
    success_limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_oss: bool = False,
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    print("正在查询数据库...")
    audio_rows = fetch_audio_rows(start_date=start_date, end_date=end_date, order_desc=True)
    total_eligible = len(audio_rows)

    if total_eligible == 0:
        print("没有可下载的录音记录")
        return

    # 统计有 AudioFile 的记录数
    oss_count = sum(1 for a in audio_rows if a.audio_file)
    if use_oss:
        print(f"共有 {total_eligible} 条记录，其中 {oss_count} 条有 OSS 路径")

    if success_limit is not None and success_limit > 0:
        print(
            f"共有 {total_eligible} 条满足条件的录音记录（已去重），"
            f"本次目标成功下载 {success_limit} 条"
        )
    else:
        print(f"共有 {total_eligible} 条满足条件的录音记录（已去重），计划全部下载")

    print(f"保存目录: {target_dir}")
    if start_date or end_date:
        print(f"日期筛选：{start_date or '-∞'} ~ {end_date or '+∞'}（基于 QueueTime）")

    # 显示下载策略
    threshold_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if use_oss:
        print(f"下载策略：优先 OSS，失败则回退 HTTP（30天内原地址，30天前替换域名）")
    else:
        print(f"下载策略：HTTP（30天内用原地址，30天前替换域名）")

    success = 0
    skipped = 0
    failed = 0
    oss_success = 0
    http_success = 0
    downloaded_bytes = 0
    t_start = time.time()

    with requests.Session() as session:
        for idx, audio in enumerate(audio_rows, 1):
            if success_limit is not None and success >= success_limit:
                break

            filename = build_filename(audio)
            dst_path = os.path.join(target_dir, filename)

            # 跳过已存在文件
            if os.path.exists(dst_path):
                skipped += 1
                continue

            downloaded = False

            try:
                # 优先使用 OSS 下载（如果指定且有 OSS 路径）
                if use_oss and audio.audio_file:
                    try:
                        content_size = download_from_oss(audio.audio_file, dst_path)
                        downloaded_bytes += content_size
                        success += 1
                        oss_success += 1
                        downloaded = True
                        print(f"[{idx}/{total_eligible}] ✓ oss: {filename[:50]}...")
                    except Exception as e:
                        print(f"[{idx}] OSS下载失败: {e}", file=sys.stderr)

                # OSS 失败或未指定，回退 HTTP 下载
                if not downloaded and audio.monitor_filename:
                    url = build_download_url(audio.monitor_filename, audio.queue_time)

                    try:
                        content_size = download_file(session, url, dst_path)
                        downloaded_bytes += content_size
                        success += 1
                        http_success += 1
                        downloaded = True

                        # 判断用的哪种方式
                        if audio.queue_time and audio.queue_time[:10] < threshold_date:
                            method = "storage"
                        else:
                            method = "http"
                        print(f"[{idx}/{total_eligible}] ✓ {method}: {filename[:50]}...")
                    except Exception as e:
                        print(f"[{idx}] HTTP下载失败: {e}", file=sys.stderr)

                if not downloaded:
                    failed += 1
                    print(
                        f"[{idx}/{total_eligible}] ✗ 下载失败: id={audio.audio_id}",
                        file=sys.stderr
                    )

            except Exception as exc:
                failed += 1
                print(f"[{idx}] 异常: {exc}", file=sys.stderr)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"下载完成")
    print(f"{'='*60}")
    print(f"成功: {success} | 跳过: {skipped} | 失败: {failed}")
    if use_oss:
        print(f"  OSS成功: {oss_success} | HTTP成功: {http_success}")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"保存目录: {target_dir}")
    print(f"{'='*60}")

    if success > 0:
        avg_size_mb = downloaded_bytes / success / (1024 * 1024)
        avg_time = elapsed / success

        print(f"\n统计信息:")
        print(f"  平均大小: {avg_size_mb:.2f} MB/文件")
        print(f"  平均耗时: {avg_time:.2f} 秒/文件")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    download_audios(
        args.target_dir,
        args.success_limit,
        args.start_date,
        args.end_date,
        args.use_oss
    )


if __name__ == "__main__":
    main()