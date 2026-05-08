#!/bin/bash
ls -la /home/REMOVED_DB_USER/asr/
echo ""
echo "=== 数据统计 ==="
echo "02_download/audio: $(ls /home/REMOVED_DB_USER/asr/02_download/audio/ 2>/dev/null | wc -l)"
echo "04_output/dialog_csv: $(ls /home/REMOVED_DB_USER/asr/04_output/dialog_csv/ 2>/dev/null | wc -l)"
echo "04_output/online_chat_csv: $(ls /home/REMOVED_DB_USER/asr/04_output/online_chat_csv/ 2>/dev/null | wc -l)"
echo "scripts: $(ls /home/REMOVED_DB_USER/asr/scripts/*.py 2>/dev/null | wc -l)"