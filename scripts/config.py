"""
项目路径配置

所有脚本统一使用此配置文件，便于维护
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 各阶段目录 ====================

# 01_source - 数据源
SOURCE_DIR = os.path.join(PROJECT_ROOT, "01_source")
DB_PATH = os.path.join(SOURCE_DIR, "heli.sqlite3")

# 02_download - 下载阶段
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "02_download")
AUDIO_DIR = os.path.join(DOWNLOAD_DIR, "audio")

# 03_process - 处理阶段
PROCESS_DIR = os.path.join(PROJECT_ROOT, "03_process")
CACHE_DIR = os.path.join(PROCESS_DIR, "cache")  # 说话者分离缓存
TEMP_DIR = os.path.join(PROCESS_DIR, "temp")    # 批处理临时目录

# 04_output - 输出阶段
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "04_output")
DIALOG_CSV_DIR = os.path.join(OUTPUT_DIR, "dialog_csv")      # 电话录音CSV
ONLINE_CHAT_CSV_DIR = os.path.join(OUTPUT_DIR, "online_chat_csv")  # 在线对话CSV

# 05_analyze - 分析阶段
ANALYZE_DIR = os.path.join(PROJECT_ROOT, "05_analyze")
REPORTS_DIR = os.path.join(ANALYZE_DIR, "reports")  # PDF报告
DATABASE_DIR = os.path.join(ANALYZE_DIR, "database")  # 数据库存储脚本

# 06_models - 模型目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "06_models")
ASR_MODEL_DIR = os.path.join(MODELS_DIR, "asr", "Qwen3-ASR-1.7B")
ALIGNER_MODEL_DIR = os.path.join(MODELS_DIR, "aligner", "Qwen3-ForcedAligner-0.6B")
LOCAL_MODELS_DIR = os.path.join(MODELS_DIR, "local")

# ==================== 验证路径存在 ====================

def ensure_dirs_exist():
    """确保所有目录存在"""
    dirs = [
        SOURCE_DIR, AUDIO_DIR, CACHE_DIR, TEMP_DIR,
        DIALOG_CSV_DIR, ONLINE_CHAT_CSV_DIR,
        REPORTS_DIR, DATABASE_DIR,
        ASR_MODEL_DIR, ALIGNER_MODEL_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    # 测试路径配置
    print("项目根目录:", PROJECT_ROOT)
    print("\n各阶段目录:")
    print("  01_source:", SOURCE_DIR)
    print("  数据库:", DB_PATH, "- 存在:", os.path.exists(DB_PATH))
    print("  02_download/audio:", AUDIO_DIR)
    print("  03_process/cache:", CACHE_DIR)
    print("  04_output/dialog_csv:", DIALOG_CSV_DIR)
    print("  04_output/online_chat_csv:", ONLINE_CHAT_CSV_DIR)
    print("  05_analyze/reports:", REPORTS_DIR)
    print("  06_models/asr:", ASR_MODEL_DIR, "- 存在:", os.path.exists(ASR_MODEL_DIR))
    print("  06_models/aligner:", ALIGNER_MODEL_DIR, "- 存在:", os.path.exists(ALIGNER_MODEL_DIR))