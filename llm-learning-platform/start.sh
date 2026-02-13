#!/bin/bash

# LLM 学习平台一键启动脚本

set -e

echo "🚀 LLM 学习平台 - 一键启动"
echo "================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未检测到 Python 3,请先安装 Python 3.8+"
    echo "macOS: brew install python@3.11"
    echo "访问: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"

# 创建虚拟环境 (如果不存在)
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
if [ ! -f "venv/.installed" ]; then
    echo "📥 安装依赖 (首次运行可能需要几分钟)..."
    echo "正在升级 pip..."
    pip install --upgrade pip
    echo ""
    echo "正在安装依赖包 (使用国内镜像加速)..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    touch venv/.installed
    echo "✅ 依赖安装完成"
else
    echo "✅ 依赖已安装"
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  未检测到 .env 文件,正在创建..."
    cp .env.example .env
    echo "📝 请编辑 .env 文件,填入你的 API Key"
    echo "然后重新运行此脚本"
    exit 0
fi

# 创建数据目录
mkdir -p data

echo ""
echo "✅ 环境检查完成!"
echo "🌐 启动 Streamlit 应用..."
echo "浏览器将自动打开,如未打开请访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止服务"
echo "================================"

# 启动应用
streamlit run app.py
