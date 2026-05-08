"""
将在线对话CSV数据导入数据库
读取CSV文件，调用API分类，存入dialog_analysis表
"""

import os
import sys
import csv
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from glob import glob

# 添加database目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "05_analyze", "database"))

from db_store import (
    create_table,
    batch_insert,
    check_file_exists,
    get_connection
)

# ==================== 配置 ====================

# 导入路径配置
from config import DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR

# API 配置
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_MODEL = "qwen3-8b"
API_MAX_WORKERS = 10  # 并发线程数

# ==================== Prompt 定义 ====================

INTENT_PROMPT = """下面是一段保险客服电话或在线对话的内容，请分析客户的来电原因。

对话内容：
{dialog}

请从以下分类体系中选择最符合的原因，输出格式为"一级分类-二级分类"：

【判断规则】
- 如果客户问"是什么、怎么做、多少钱" → 咨询（了解信息）
- 如果客户问"到哪了、什么时候、有没有、在不在" → 查询（查状态）
- 如果客户说"我要、帮我、申请" → 办理（办业务）
- 如果客户有抱怨、不满、质疑 → 投诉

【咨询】客户询问了解信息（问知识、流程）
- 产品了解：问产品保障内容
- 条款解释：问条款含义、保障范围
- 费率查询：问保费价格
- 理赔流程了解：问理赔步骤、怎么理赔
- 理赔材料了解：问理赔需要什么材料
- 退保流程了解：问退保怎么操作
- 保障范围了解：问某种情况能否理赔
- 其他了解：其他咨询问题

【查询】客户查询具体状态（问进度、状态）
- 理赔进度查询：查理赔审核到哪了
- 理赔到账查询：查理赔款什么时候到账
- 保单状态查询：查保单是否有效、什么时候生成
- 其他查询：其他状态查询

【办理】客户要办理业务
- 理赔申请：提交理赔
- 事故报案：报告事故
- 新保投保：新买保险
- 续保办理：续保操作
- 保单变更：变更保单信息
- 退保办理：办理退保
- 减保办理：减少保额

【投诉】客户有不满投诉
- 理赔时效投诉：投诉理赔太慢
- 拒赔投诉：投诉拒赔决定
- 理赔金额异议：对赔付金额不满
- 服务态度投诉：投诉客服态度
- 其他投诉：其他不满

【其他】回访确认、信息核实

只输出一个分类（如：查询-保单状态查询），不要解释。"""

SENTIMENT_PROMPT = """分析以下客服对话中客户的情绪状态。

对话内容：
{dialog}

请判断客户的情绪倾向，只输出一个标签：
- 满意：客户问题得到解决，态度积极
- 中立：客户情绪平稳，无明显倾向
- 不满：客户有抱怨、质疑、投诉倾向

只输出一个词（满意/中立/不满），不要解释。"""

ISSUE_PROMPT = """分析以下客服对话，判断客户咨询的核心问题类型。

对话内容：
{dialog}

请从以下选项中选择唯一一个最符合的问题类型，只输出类型名称：

理赔进度查询、理赔款项到账查询、理赔时效咨询、理赔材料准备、理赔资料补交、理赔金额计算、理赔金额异议、报销比例咨询、保障范围咨询、理赔条件确认、既往症咨询、退保流程咨询、退保金额计算、犹豫期退保、保单查询、保单变更、信息修改、续保咨询、产品条款解释、保费计算、投保条件咨询

只输出一个类型名称（如：理赔进度查询），不要输出多个，不要解释。"""


# ==================== 函数定义 ====================

def call_api(prompt, max_retries=3):
    """调用 API，带重试机制"""
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
                    "max_tokens": 256,
                    "enable_thinking": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def read_dialog_csv(csv_path):
    """读取 CSV 文件，返回对话内容字符串"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) < 2:
                return None

            # 检查是否有标题行
            first_row = rows[0]
            if first_row[0] in ['角色', '发送者', 'sender']:
                rows = rows[1:]

            # 拼接对话内容
            dialog_lines = []
            for row in rows:
                if len(row) >= 2:
                    role = row[0]
                    content = row[1]
                    dialog_lines.append(f"{role}: {content}")

            return '\n'.join(dialog_lines)
    except Exception as e:
        print(f"读取文件失败: {csv_path}, {e}")
        return None


def parse_intent(result):
    """解析意图分类结果"""
    if not result:
        return "其他", "其他了解"

    # 定义有效的一级分类
    valid_primary = ["咨询", "查询", "办理", "投诉", "其他"]

    # 尝试解析 "一级分类-二级分类" 格式
    if '-' in result:
        parts = result.split('-', 1)
        primary = parts[0].strip()
        secondary = parts[1].strip() if len(parts) > 1 else "其他"

        # 如果一级分类无效，尝试从二级分类推断
        if primary not in valid_primary:
            # 常见的二级分类关键词映射
            if any(k in primary for k in ['了解', '咨询', '解释']):
                primary = "咨询"
            elif any(k in primary for k in ['查询', '进度', '状态']):
                primary = "查询"
            elif any(k in primary for k in ['申请', '办理', '报案']):
                primary = "办理"
            elif any(k in primary for k in ['投诉', '不满']):
                primary = "投诉"
            else:
                primary = "其他"

            # 二级分类设为原一级分类的内容
            secondary = parts[0].strip()

        return primary, secondary

    return "其他", result


def process_single_file(csv_path, source_type):
    """处理单个文件，返回分析结果"""
    filename = os.path.basename(csv_path)

    # 检查是否已处理
    if check_file_exists(csv_path):
        return None

    # 读取对话内容
    dialog = read_dialog_csv(csv_path)
    if not dialog:
        return None

    # 调用API分类
    intent_result = call_api(INTENT_PROMPT.format(dialog=dialog))
    sentiment_result = call_api(SENTIMENT_PROMPT.format(dialog=dialog))
    issue_result = call_api(ISSUE_PROMPT.format(dialog=dialog))

    # 解析结果
    primary_intent, secondary_intent = parse_intent(intent_result)

    return {
        'source_type': source_type,
        'source_file': csv_path,
        'primary_intent': primary_intent,
        'secondary_intent': secondary_intent,
        'sentiment': sentiment_result or '中立',
        'issue_type': issue_result
    }


# 全局计数器
progress_lock = Lock()
processed_count = 0
skipped_count = 0
failed_count = 0


def import_online_chat(limit=None, batch_size=100):
    """导入在线对话数据"""
    global processed_count, skipped_count, failed_count

    # 获取文件列表，按文件名排序（最新的在前）
    csv_files = sorted(glob(os.path.join(ONLINE_CHAT_CSV_DIR, "*.csv")), reverse=True)

    if limit:
        csv_files = csv_files[:limit]

    total_files = len(csv_files)
    print(f"找到 {total_files} 个在线对话文件（已按时间倒序排列）")

    # 确保表存在
    create_table()

    # 批量处理
    batch_results = []

    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_file, f, 'online'): f
            for f in csv_files
        }

        for future in as_completed(futures):
            result = future.result()

            with progress_lock:
                if result is None:
                    skipped_count += 1
                else:
                    batch_results.append(result)
                    processed_count += 1

                # 进度显示
                done = processed_count + skipped_count + failed_count
                if done % 50 == 0 or done == total_files:
                    print(f"进度: {done}/{total_files} - 已处理 {processed_count}, 跳过 {skipped_count}, 失败 {failed_count}")

            # 批量写入数据库
            if len(batch_results) >= batch_size:
                try:
                    batch_insert(batch_results)
                    batch_results.clear()
                except Exception as e:
                    print(f"批量插入失败: {e}")
                    failed_count += len(batch_results)
                    batch_results.clear()

    # 写入剩余数据
    if batch_results:
        try:
            batch_insert(batch_results)
        except Exception as e:
            print(f"最后批次插入失败: {e}")
            failed_count += len(batch_results)

    print(f"\n完成！已处理 {processed_count}, 跳过 {skipped_count}, 失败 {failed_count}")


def import_dialog_csv(limit=None, batch_size=100):
    """导入电话录音数据"""
    global processed_count, skipped_count, failed_count

    # 重置计数器
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # 获取文件列表
    csv_files = glob(os.path.join(DIALOG_CSV_DIR, "*.csv"))

    if limit:
        csv_files = csv_files[:limit]

    total_files = len(csv_files)
    print(f"找到 {total_files} 个电话录音文件")

    # 确保表存在
    create_table()

    # 批量处理
    batch_results = []

    with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_file, f, 'phone'): f
            for f in csv_files
        }

        for future in as_completed(futures):
            result = future.result()

            with progress_lock:
                if result is None:
                    skipped_count += 1
                else:
                    batch_results.append(result)
                    processed_count += 1

                # 进度显示
                done = processed_count + skipped_count + failed_count
                if done % 50 == 0 or done == total_files:
                    print(f"进度: {done}/{total_files} - 已处理 {processed_count}, 跳过 {skipped_count}, 失败 {failed_count}")

            # 批量写入数据库
            if len(batch_results) >= batch_size:
                try:
                    batch_insert(batch_results)
                    batch_results.clear()
                except Exception as e:
                    print(f"批量插入失败: {e}")
                    failed_count += len(batch_results)
                    batch_results.clear()

    # 写入剩余数据
    if batch_results:
        try:
            batch_insert(batch_results)
        except Exception as e:
            print(f"最后批次插入失败: {e}")
            failed_count += len(batch_results)

    print(f"\n完成！已处理 {processed_count}, 跳过 {skipped_count}, 失败 {failed_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导入CSV数据到数据库')
    parser.add_argument('--type', choices=['online', 'phone', 'all'], default='online',
                        help='数据类型：online(在线对话), phone(电话录音), all(全部)')
    parser.add_argument('--limit', type=int, default=None, help='限制处理文件数')
    parser.add_argument('--batch-size', type=int, default=100, help='批量写入大小')

    args = parser.parse_args()

    if args.type == 'online':
        import_online_chat(limit=args.limit, batch_size=args.batch_size)
    elif args.type == 'phone':
        import_dialog_csv(limit=args.limit, batch_size=args.batch_size)
    elif args.type == 'all':
        import_online_chat(limit=args.limit, batch_size=args.batch_size)
        import_dialog_csv(limit=args.limit, batch_size=args.batch_size)