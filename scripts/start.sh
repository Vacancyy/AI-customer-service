#!/bin/bash
# 南京宁惠保 AI客服系统启动脚本

# 设置环境变量（请根据实际情况修改）
export DASHSCOPE_API_KEY="your-api-key-here"
export DB_HOST="localhost"
export DB_PORT="3308"
export DB_USER="root"
export DB_PASS=""
export DB_NAME="ai_customer_service"

# 启动Web服务
echo "正在启动AI客服系统..."
cd "$(dirname "$0")/.."
python scripts/web_demo.py

# 或者后台运行：
# nohup python scripts/web_demo.py > logs/web_demo.log 2>&1 &