# 快速开始指南

## ⚡ 一键启动 (推荐)

```bash
cd ~/llm-learning-platform
./start.sh
```

脚本会自动完成:
1. ✅ 检查 Python 环境
2. ✅ 创建虚拟环境
3. ✅ 安装依赖
4. ✅ 启动 Web 平台

## 📁 项目结构

```
llm-learning-platform/
├── app.py                    # Streamlit 学习平台主界面
├── start.sh                  # 一键启动脚本
├── requirements.txt          # Python 依赖
├── README.md                 # 环境配置详细指南
├── .env.example             # API Key 配置模板
│
├── data/                    # 数据目录
│   └── progress.json        # 学习进度 (自动生成)
│
├── exercises/               # 代码练习模板
│   ├── week1_translator.py  # Week1: 翻译助手
│   ├── week2_rag.py         # Week2: RAG 问答
│   └── week7_react_agent.py # Week7: ReAct Agent
│
└── utils/                   # 工具函数
    └── progress_tracker.py  # 进度统计工具
```

## 🎯 使用流程

### 1️⃣ 首次使用

```bash
# 配置 API Key
cp .env.example .env
nano .env  # 填入你的 OpenAI/DeepSeek API Key

# 启动平台
./start.sh
```

### 2️⃣ 开始学习

1. 在 Web 界面设置学习开始日期
2. 按照每个阶段的任务进行学习
3. 点击资源链接阅读教程
4. 在 `exercises/` 目录运行代码练习
5. 在平台上做笔记和打卡

### 3️⃣ 运行代码练习

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行 Week1 翻译助手
python exercises/week1_translator.py

# 运行 Week2 RAG 问答
python exercises/week2_rag.py

# 运行 Week7 ReAct Agent
python exercises/week7_react_agent.py
```

## 🔧 常用命令

```bash
# 启动平台
./start.sh

# 导出学习笔记
python utils/progress_tracker.py

# 查看进度报告
python -c "from utils.progress_tracker import generate_progress_report; print(generate_progress_report())"

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## 📝 学习建议

1. **严格遵守 DDL**: 不要跨阶段学习,基础不牢地动山摇
2. **动手实践**: 每个练习都要运行并修改代码
3. **做好笔记**: 在平台上记录学习心得和问题
4. **定期复盘**: 每周日查看进度报告,补齐未完成任务
5. **构建作品集**: 将完成的项目上传到 GitHub

祝你学习顺利! 🚀
