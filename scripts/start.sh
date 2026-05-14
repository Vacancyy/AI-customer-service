#!/bin/bash
# 南京宁惠保 AI客服系统启动脚本

# 从.env文件加载环境变量
cd "$(dirname "$0")/.."
if [ -f ".env" ]; then
    echo "加载环境变量配置..."
    export $(grep -v '^#' .env | xargs)
else
    echo "错误：未找到.env配置文件，请先创建.env文件"
    echo "参考.env.example模板"
    exit 1
fi

# 启动Web服务
echo "正在启动AI客服系统..."
python scripts/web_demo.py

# 或者后台运行：
# nohup python scripts/web_demo.py > logs/web_demo.log 2>&1 &