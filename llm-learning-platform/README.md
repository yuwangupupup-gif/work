# 🚀 LLM 学习平台

一个基于 Streamlit 的大模型学习平台，帮助零基础学习者系统掌握 LLM 技术。

## ✨ 功能特点

- 📅 **8周系统学习路径**：从 Prompt Engineering 到模型微调
- 📝 **学习笔记系统**：每个任务都可以记录学习笔记
- ✅ **进度追踪**：实时追踪学习进度，可视化完成情况
- ⏰ **DDL 提醒**：每个任务都有截止日期倒计时
- 💻 **代码练习模板**：提供实战练习代码框架

## 🎯 学习路线

### 第一阶段：认知与基础 (Week 1)
- Transformer 架构理解
- Prompt Engineering 技巧
- OpenAI API 调用

### 第二阶段：RAG 开发 (Week 2-3)
- 向量数据库（ChromaDB）
- RAG 流程实现
- 进阶 RAG 优化

### 第三阶段：模型微调 (Week 4-6)
- LoRA/QLoRA 微调
- 数据集准备
- 模型评估

### 第四阶段：Agent 开发 (Week 7-8)
- ReAct 框架
- Function Calling
- 多 Agent 协作

## 🚀 快速开始

### 在线访问
访问部署好的应用：[你的应用URL]

### 本地运行

```bash
# 克隆项目
git clone <your-repo-url>
cd llm-learning-platform

# 一键启动
./start.sh
```

浏览器会自动打开 http://localhost:8501

## 📋 环境要求

- Python 3.9+
- 依赖包见 `requirements.txt`

## 🔑 配置 API Key（可选）

如需使用 OpenAI API 功能：

1. 复制 `.env.example` 为 `.env`
2. 填入你的 OpenAI API Key

```bash
OPENAI_API_KEY=sk-your-key-here
```

## 📚 学习资源

所有学习资源链接都在平台内提供，包括：
- 吴恩达课程
- Hugging Face 文档
- LangChain 教程
- 等等...

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License
