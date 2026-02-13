import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="大模型学习平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 学习计划数据
LEARNING_PLAN = {
    "第一阶段：认知与基础 (Week 1-2)": {
        "tasks": [
            {
                "name": "Day 1-2: LLM 基础概念",
                "ddl": 2,
                "resources": [
                    {"title": "🇨🇳 李沐 - 大语言模型原理（中文）", "url": "https://www.bilibili.com/video/BV1TD4y137mP"},
                    {"title": "🇨🇳 ChatGPT 工作原理（中文图解）", "url": "https://zhuanlan.zhihu.com/p/619490922"},
                    {"title": "🇨🇳 什么是 Token（中文）", "url": "https://platform.openai.com/tokenizer"},
                    {"title": "🇬🇧 3Blue1Brown - Attention（可开字幕）", "url": "https://www.youtube.com/watch?v=eMlx5fFNoYc"}
                ],
                "exercise": "用自己的话解释：Tokenization、Embedding、Attention、Transformer 四个概念",
                "hint": "思考：为什么 GPT 不能直接理解文字？Token 是什么？Attention 在做什么计算？"
            },
            {
                "name": "Day 3-4: Transformer 架构深入",
                "ddl": 4,
                "resources": [
                    {"title": "🇨🇳 李沐论文精读 - Transformer（中文）", "url": "https://www.bilibili.com/video/BV1pu411o7BE"},
                    {"title": "🇨🇳 图解 Transformer（中文翻译）", "url": "https://blog.csdn.net/qq_41664845/article/details/84969266"},
                    {"title": "🇨🇳 Attention 机制详解（中文）", "url": "https://zhuanlan.zhihu.com/p/47282410"},
                    {"title": "🇬🇧 Illustrated Transformer（可翻译）", "url": "https://jalammar.github.io/illustrated-transformer/"}
                ],
                "exercise": "绘制 Transformer 完整架构图，手动计算一次 Self-Attention（3个词的例子）",
                "hint": "Q=WQ*X, K=WK*X, V=WV*X, Attention(Q,K,V) = softmax(QK^T/√d_k)V，重点理解 Multi-Head"
            },
            {
                "name": "Day 5-6: Prompt Engineering 基础",
                "ddl": 6,
                "resources": [
                    {"title": "🇨🇳 Prompt Engineering 中文指南", "url": "https://www.promptingguide.ai/zh"},
                    {"title": "🇨🇳 吴恩达课程（B站中文字幕）", "url": "https://www.bilibili.com/video/BV1Bo4y1A7FU"},
                    {"title": "🇨🇳 Prompt 技巧大全（中文）", "url": "https://github.com/f/awesome-chatgpt-prompts/blob/main/README-cn.md"},
                    {"title": "🇨🇳 OpenAI Prompt 最佳实践（中文）", "url": "https://cookbook.openai.com/"}
                ],
                "exercise": "掌握 6 种 Prompt 技巧：Zero-shot、Few-shot、CoT、Self-Consistency、ToT、ReAct",
                "hint": "实践：写一个旅游规划 Prompt，要求输出 JSON 格式，包含景点、预算、时间安排"
            },
            {
                "name": "Day 7-8: Prompt 进阶技巧",
                "ddl": 8,
                "resources": [
                    {"title": "🇨🇳 Prompt 注入攻防（中文）", "url": "https://learnprompting.org/zh-Hans/docs/prompt_hacking/injection"},
                    {"title": "🇨🇳 提示词工程指南（中文）", "url": "https://github.com/dair-ai/Prompt-Engineering-Guide/tree/main/guides/prompts-intro.zh.md"},
                    {"title": "🇨🇳 常用 Prompt 模板（中文）", "url": "https://github.com/PlexPt/awesome-chatgpt-prompts-zh"}
                ],
                "exercise": "实现 3 个角色 Prompt：Linux 终端、Python 解释器、面试官",
                "hint": "用 System Message 定义角色，用 Few-shot 示例约束输出格式"
            },
            {
                "name": "Day 9-10: OpenAI API 实战",
                "ddl": 10,
                "resources": [
                    {"title": "🇨🇳 OpenAI API 中文文档", "url": "https://platform.openai.com/docs/quickstart"},
                    {"title": "🇨🇳 Python 调用 ChatGPT 教程", "url": "https://www.bilibili.com/video/BV1M24y1h78T"},
                    {"title": "🇨🇳 API 成本优化技巧（中文）", "url": "https://zhuanlan.zhihu.com/p/620626490"}
                ],
                "exercise": "实现一个多轮对话翻译助手，支持上下文记忆、流式输出、Token 统计",
                "hint": "temperature、top_p、max_tokens、frequency_penalty 参数的作用，如何计算成本"
            },
            {
                "name": "Day 11-12: LangChain 框架入门",
                "ddl": 12,
                "resources": [
                    {"title": "🇨🇳 LangChain 中文教程", "url": "https://www.langchain.com.cn/"},
                    {"title": "🇨🇳 LangChain 实战（B站）", "url": "https://www.bilibili.com/video/BV1XX4y1K7X4"},
                    {"title": "🇨🇳 LangChain 中文文档", "url": "https://python.langchain.com.cn/docs/get_started/introduction"}
                ],
                "exercise": "用 LangChain 实现：PromptTemplate + LLM + OutputParser 的完整链路",
                "hint": "掌握 Chain、Memory、Agent 三大核心组件"
            },
            {
                "name": "Day 13-14: 模型评估与测试",
                "ddl": 14,
                "resources": [
                    {"title": "🇨🇳 如何评估大模型（中文）", "url": "https://zhuanlan.zhihu.com/p/642908437"},
                    {"title": "🇨🇳 C-Eval 中文评测基准", "url": "https://cevalbenchmark.com/"},
                    {"title": "🇨🇳 SuperCLUE 中文榜单", "url": "https://www.superclueai.com/"}
                ],
                "exercise": "对比 GPT-3.5 和 GPT-4 在同一任务上的表现差异（准确率、速度、成本）",
                "hint": "使用 5-10 个测试样例，记录输出质量、响应时间、Token 消耗"
            }
        ]
    },
    "第二阶段：RAG 开发 (Week 3)": {
        "tasks": [
            {
                "name": "Day 15-16: Embedding 与向量检索",
                "ddl": 16,
                "resources": [
                    {"title": "🇨🇳 Embedding 原理详解（中文）", "url": "https://zhuanlan.zhihu.com/p/647710447"},
                    {"title": "🇨🇳 向量相似度计算（中文）", "url": "https://www.cnblogs.com/wuyongqiang/p/15467234.html"},
                    {"title": "🇨🇳 OpenAI Embedding API 使用", "url": "https://platform.openai.com/docs/guides/embeddings"}
                ],
                "exercise": "理解 Cosine Similarity、Euclidean Distance、Dot Product 的区别，手动计算示例",
                "hint": "为什么 Embedding 能捕捉语义？768 维向量代表什么？归一化的作用？"
            },
            {
                "name": "Day 17-18: ChromaDB 实战",
                "ddl": 18,
                "resources": [
                    {"title": "🇨🇳 ChromaDB 中文教程", "url": "https://docs.trychroma.com/getting-started"},
                    {"title": "🇨🇳 向量数据库入门（中文）", "url": "https://zhuanlan.zhihu.com/p/639277854"},
                    {"title": "🇨🇳 Milvus vs Chroma 对比", "url": "https://zhuanlan.zhihu.com/p/635839939"}
                ],
                "exercise": "实现文档切片 → Embedding → 存储 → 语义搜索完整流程",
                "hint": "RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)"
            },
            {
                "name": "Day 19-20: RAG 核心流程",
                "ddl": 20,
                "resources": [
                    {"title": "🇨🇳 RAG 原理与实践（中文）", "url": "https://zhuanlan.zhihu.com/p/651857654"},
                    {"title": "🇨🇳 LangChain RAG 教程", "url": "https://www.langchain.com.cn/use_cases/question_answering"},
                    {"title": "🇨🇳 知识库问答实战（B站）", "url": "https://www.bilibili.com/video/BV1sN411n7cc"}
                ],
                "exercise": "构建个人知识库问答系统（支持 PDF/Markdown 导入）",
                "hint": "Retriever → Prompt → LLM → Answer，注意 Context 长度控制"
            },
            {
                "name": "Day 21: 进阶 RAG 优化",
                "ddl": 21,
                "resources": [
                    {"title": "🇨🇳 RAG 进阶技巧（中文）", "url": "https://zhuanlan.zhihu.com/p/667626118"},
                    {"title": "🇨🇳 Reranking 技术详解", "url": "https://zhuanlan.zhihu.com/p/641080888"},
                    {"title": "🇨🇳 混合检索策略（中文）", "url": "https://blog.csdn.net/weixin_43334693/article/details/134099766"}
                ],
                "exercise": "实现 Hybrid Search（BM25 + Vector）+ Reranking + 引用来源标注",
                "hint": "检索 Top-20 → Rerank → 取 Top-5 → 注入 Prompt，输出 [来源1][来源2]"
            }
        ]
    },
    "第三阶段：模型微调 (Week 4-7)": {
        "tasks": [
            {
                "name": "Day 22-24: 微调理论基础",
                "ddl": 24,
                "resources": [
                    {"title": "🇨🇳 大模型微调入门（中文）", "url": "https://zhuanlan.zhihu.com/p/635152813"},
                    {"title": "🇨🇳 LoRA 原理详解（中文）", "url": "https://zhuanlan.zhihu.com/p/618894919"},
                    {"title": "🇨🇳 PEFT 技术对比（中文）", "url": "https://zhuanlan.zhihu.com/p/635686756"},
                    {"title": "🇨🇳 李沐 - LoRA 论文精读", "url": "https://www.bilibili.com/video/BV1Ld4y1L7L6"}
                ],
                "exercise": "理解 4 种微调方法：Full Fine-tuning、Adapter、Prefix Tuning、LoRA 的区别",
                "hint": "对比参数量、显存占用、训练速度、效果。为什么 LoRA 只训练 0.1% 参数却效果好？"
            },
            {
                "name": "Day 25-27: 环境搭建与模型加载",
                "ddl": 27,
                "resources": [
                    {"title": "🇨🇳 Transformers 中文教程", "url": "https://transformers.run/"},
                    {"title": "🇨🇳 模型量化详解（中文）", "url": "https://zhuanlan.zhihu.com/p/627436535"},
                    {"title": "🇨🇳 Colab 使用教程（中文）", "url": "https://www.bilibili.com/video/BV1Vt4y1K7HX"},
                    {"title": "🇨🇳 显存优化技巧（中文）", "url": "https://zhuanlan.zhihu.com/p/620885226"}
                ],
                "exercise": "在 Colab (T4 GPU) 加载 Qwen-7B-Chat，实现 4bit 量化推理",
                "hint": "使用 BitsAndBytesConfig + load_in_4bit=True 节省显存，from_pretrained 参数详解"
            },
            {
                "name": "Day 28-30: 数据集构建与处理",
                "ddl": 30,
                "resources": [
                    {"title": "🇨🇳 微调数据集构建指南", "url": "https://zhuanlan.zhihu.com/p/635686756"},
                    {"title": "🇨🇳 Alpaca 中文数据集", "url": "https://github.com/ymcui/Chinese-LLaMA-Alpaca"},
                    {"title": "🇨🇳 数据清洗与增强（中文）", "url": "https://zhuanlan.zhihu.com/p/629589593"},
                    {"title": "🇨🇳 指令微调数据格式", "url": "https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md"}
                ],
                "exercise": "构建 100 条高质量指令微调数据集（选择一个垂直领域：医疗/法律/编程/客服）",
                "hint": "格式：{instruction, input, output}。确保多样性：问答、总结、翻译、生成等"
            },
            {
                "name": "Day 31-33: LoRA 微调实战",
                "ddl": 33,
                "resources": [
                    {"title": "🇨🇳 LoRA 微调完整教程（中文）", "url": "https://www.bilibili.com/video/BV1LW4y1r7GC"},
                    {"title": "🇨🇳 PEFT 库使用指南", "url": "https://huggingface.co/docs/peft/index"},
                    {"title": "🇨🇳 ChatGLM-6B 微调实战", "url": "https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning"},
                    {"title": "🇨🇳 参数调优最佳实践", "url": "https://zhuanlan.zhihu.com/p/631535042"}
                ],
                "exercise": "使用 LoRA 微调 Qwen-7B，实现特定风格输出（例如：猫娘、古风、技术博主）",
                "hint": "重点参数：r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['q_proj','v_proj']"
            },
            {
                "name": "Day 34-36: QLoRA 与显存优化",
                "ddl": 36,
                "resources": [
                    {"title": "🇨🇳 QLoRA 原理与实践（中文）", "url": "https://zhuanlan.zhihu.com/p/636879908"},
                    {"title": "🇨🇳 4bit 量化详解（中文）", "url": "https://zhuanlan.zhihu.com/p/632426681"},
                    {"title": "🇨🇳 12GB 显卡微调 LLaMA", "url": "https://www.bilibili.com/video/BV1fd4y1Z7Y5"},
                    {"title": "🇨🇳 Gradient Checkpointing", "url": "https://zhuanlan.zhihu.com/p/599806898"}
                ],
                "exercise": "用 QLoRA 在 12GB 显卡上微调 13B 模型（对比 LoRA 的显存占用）",
                "hint": "4bit 量化 + NF4 数据类型 + double quantization，batch_size=1, gradient_accumulation_steps=4"
            },
            {
                "name": "Day 37-39: LLaMA-Factory 全流程",
                "ddl": 39,
                "resources": [
                    {"title": "🇨🇳 LLaMA-Factory 中文教程", "url": "https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md"},
                    {"title": "🇨🇳 WebUI 使用指南（B站）", "url": "https://www.bilibili.com/video/BV1LW4y1r7GC"},
                    {"title": "🇨🇳 配置文件详解（中文）", "url": "https://github.com/hiyouga/LLaMA-Factory/wiki/Chinese"},
                    {"title": "🇨🇳 常见问题解答", "url": "https://github.com/hiyouga/LLaMA-Factory/blob/main/FAQ_zh.md"}
                ],
                "exercise": "用 LLaMA-Factory 完成：数据准备 → 训练 → 评估 → 导出 → 部署完整流程",
                "hint": "llamafactory-cli train --stage sft --model_name_or_path qwen --dataset alpaca_zh"
            },
            {
                "name": "Day 40-42: 全参数微调 (SFT)",
                "ddl": 42,
                "resources": [
                    {"title": "🇨🇳 全参数微调 vs LoRA（中文）", "url": "https://zhuanlan.zhihu.com/p/635686756"},
                    {"title": "🇨🇳 DeepSpeed 使用教程", "url": "https://www.bilibili.com/video/BV1Td4y1Z7Y5"},
                    {"title": "🇨🇳 分布式训练入门（中文）", "url": "https://zhuanlan.zhihu.com/p/617133971"},
                    {"title": "🇨🇳 多卡训练配置指南", "url": "https://github.com/THUDM/ChatGLM-6B/blob/main/README.md"}
                ],
                "exercise": "理解全参数微调 vs LoRA 的适用场景，什么时候必须用全参数？",
                "hint": "领域知识注入、语言迁移需要全参数；风格调整、任务适配用 LoRA"
            },
            {
                "name": "Day 43-45: RLHF 与 DPO",
                "ddl": 45,
                "resources": [
                    {"title": "🇨🇳 RLHF 原理详解（中文）", "url": "https://zhuanlan.zhihu.com/p/622134699"},
                    {"title": "🇨🇳 DPO 算法解析（中文）", "url": "https://zhuanlan.zhihu.com/p/642569664"},
                    {"title": "🇨🇳 TRL 库使用教程", "url": "https://huggingface.co/docs/trl/index"},
                    {"title": "🇨🇳 偏好数据构建（中文）", "url": "https://zhuanlan.zhihu.com/p/638333362"}
                ],
                "exercise": "构建偏好数据集（chosen vs rejected），理解 PPO 训练流程",
                "hint": "RLHF 三阶段：SFT → Reward Model → PPO。DPO 直接优化，无需 RM"
            },
            {
                "name": "Day 46-48: 模型评估与部署",
                "ddl": 48,
                "resources": [
                    {"title": "🇨🇳 模型评估指标详解（中文）", "url": "https://zhuanlan.zhihu.com/p/642908437"},
                    {"title": "🇨🇳 vLLM 部署教程（中文）", "url": "https://www.bilibili.com/video/BV1RN411c7nc"},
                    {"title": "🇨🇳 llama.cpp 量化部署", "url": "https://zhuanlan.zhihu.com/p/635152813"},
                    {"title": "🇨🇳 FastChat 部署指南", "url": "https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md"}
                ],
                "exercise": "评估微调后模型：PPL、BLEU、人工评分，对比微调前后差异",
                "hint": "使用 vLLM 部署，对比推理速度（tokens/s）、显存占用、并发能力"
            },
            {
                "name": "Day 49-51: 持续学习与灾难遗忘",
                "ddl": 51,
                "resources": [
                    {"title": "🇨🇳 灾难性遗忘问题（中文）", "url": "https://zhuanlan.zhihu.com/p/640987937"},
                    {"title": "🇨🇳 持续学习策略（中文）", "url": "https://zhuanlan.zhihu.com/p/618894919"},
                    {"title": "🇨🇳 通用能力保持技巧", "url": "https://github.com/hiyouga/LLaMA-Factory/wiki/Chinese"}
                ],
                "exercise": "微调后测试通用能力是否下降（加法运算、常识问答），如何缓解？",
                "hint": "混合通用数据集、控制学习率、使用 EWC 正则化"
            }
        ]
    },
    "第四阶段：Agent 开发 (Week 8-9)": {
        "tasks": [
            {
                "name": "Day 52-54: Agent 基础与 ReAct",
                "ddl": 54,
                "resources": [
                    {"title": "🇨🇳 Agent 原理详解（中文）", "url": "https://zhuanlan.zhihu.com/p/643085881"},
                    {"title": "🇨🇳 ReAct 框架实战（中文）", "url": "https://www.bilibili.com/video/BV1Xu411z7d6"},
                    {"title": "🇨🇳 LangChain Agent 教程", "url": "https://www.langchain.com.cn/modules/agents"},
                    {"title": "🇬🇧 ReAct 论文（可翻译）", "url": "https://arxiv.org/abs/2210.03629"}
                ],
                "exercise": "实现 ReAct Agent：Question → Thought → Action → Observation 循环",
                "hint": "工具：Calculator、Wikipedia、Weather API，最多 5 轮循环"
            },
            {
                "name": "Day 55-57: Function Calling",
                "ddl": 57,
                "resources": [
                    {"title": "🇨🇳 Function Calling 详解（中文）", "url": "https://zhuanlan.zhihu.com/p/638318103"},
                    {"title": "🇨🇳 OpenAI 函数调用教程", "url": "https://www.bilibili.com/video/BV1vu411z7d6"},
                    {"title": "🇨🇳 工具定义最佳实践（中文）", "url": "https://platform.openai.com/docs/guides/function-calling"}
                ],
                "exercise": "构建智能助手：天气查询 + 日历管理 + 邮件发送（3 个 Function）",
                "hint": "定义 JSON Schema → 模型返回 function_call → 执行函数 → 返回结果"
            },
            {
                "name": "Day 58-60: 多 Agent 协作",
                "ddl": 60,
                "resources": [
                    {"title": "🇨🇳 AutoGen 中文教程", "url": "https://www.bilibili.com/video/BV1LN411E7cX"},
                    {"title": "🇨🇳 MetaGPT 实战指南", "url": "https://github.com/geekan/MetaGPT/blob/main/README_CN.md"},
                    {"title": "🇨🇳 多 Agent 协作模式（中文）", "url": "https://zhuanlan.zhihu.com/p/655439706"},
                    {"title": "🇨🇳 CrewAI 使用教程", "url": "https://www.bilibili.com/video/BV1Xu411z7d6"}
                ],
                "exercise": "实现双 Agent Code Review：Coder (写代码) + Reviewer (审查代码)",
                "hint": "UserProxy ↔ Assistant，最多 3 轮对话达成一致"
            },
            {
                "name": "Day 61-62: Memory 与上下文管理",
                "ddl": 62,
                "resources": [
                    {"title": "🇨🇳 LangChain Memory 详解", "url": "https://www.langchain.com.cn/modules/memory"},
                    {"title": "🇨🇳 上下文窗口优化（中文）", "url": "https://zhuanlan.zhihu.com/p/642018299"},
                    {"title": "🇨🇳 长对话管理策略（中文）", "url": "https://www.bilibili.com/video/BV1Vu411z7d6"}
                ],
                "exercise": "实现 ConversationBufferMemory、ConversationSummaryMemory 并对比",
                "hint": "超过 4k tokens 如何压缩？如何保留关键信息？"
            },
            {
                "name": "Day 63-65: 完整项目实战",
                "ddl": 65,
                "resources": [
                    {"title": "🇨🇳 智能客服系统实战（中文）", "url": "https://www.bilibili.com/video/BV1LN411E7cX"},
                    {"title": "🇨🇳 RAG + Agent 结合（中文）", "url": "https://zhuanlan.zhihu.com/p/655439706"},
                    {"title": "🇨🇳 Streamlit 部署教程", "url": "https://www.bilibili.com/video/BV1Vt4y1K7HX"},
                    {"title": "🇨🇳 项目完整代码示例", "url": "https://github.com/chatchat-space/Langchain-Chatchat"}
                ],
                "exercise": "综合项目：基于 RAG + Agent 的智能客服系统（知识库检索 + 工具调用 + 多轮对话）",
                "hint": "整合所有知识点，部署到 Streamlit Cloud，准备作品集展示"
            }
        ]
    }
}

# 初始化进度文件
PROGRESS_FILE = "data/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    os.makedirs('data', exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

# 侧边栏
st.sidebar.title("🚀 LLM 学习平台")
st.sidebar.markdown("---")

# 开始日期设置
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now()

start_date = st.sidebar.date_input(
    "设置学习开始日期",
    value=st.session_state.start_date
)
st.session_state.start_date = datetime.combine(start_date, datetime.min.time())

# 计算当前天数
current_day = (datetime.now() - st.session_state.start_date).days + 1
st.sidebar.metric("学习进度", f"第 {current_day} 天", "共 60 天")

# 加载进度
progress = load_progress()

# 主页面
st.title("🧠 大模型 (LLM) 从0到1 学习平台")
st.markdown("**核心理念**: Project-based Learning (PBL) - 边学边练")

# 显示阶段
tabs = st.tabs(list(LEARNING_PLAN.keys()))

for tab_idx, (stage_name, stage_data) in enumerate(LEARNING_PLAN.items()):
    with tabs[tab_idx]:
        st.header(stage_name)
        
        for task_idx, task in enumerate(stage_data["tasks"]):
            task_id = f"{tab_idx}_{task_idx}"
            task_status = progress.get(task_id, {"completed": False, "notes": ""})
            
            with st.expander(f"📌 {task['name']} (DDL: Day {task['ddl']})", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("**📚 学习资源**:")
                    for res in task["resources"]:
                        st.markdown(f"- [{res['title']}]({res['url']})")
                    
                    st.markdown(f"**✏️ 实战练习**: {task['exercise']}")
                    st.info(f"💡 提示: {task['hint']}")
                    
                    # 笔记区
                    notes = st.text_area(
                        "📝 学习笔记",
                        value=task_status.get("notes", ""),
                        key=f"notes_{task_id}",
                        height=100
                    )
                    
                    if st.button(f"💾 保存笔记", key=f"save_{task_id}"):
                        progress[task_id] = progress.get(task_id, {})
                        progress[task_id]["notes"] = notes
                        save_progress(progress)
                        st.success("✅ 笔记已保存")
                        st.rerun()
                
                with col2:
                    # DDL 倒计时
                    ddl_date = st.session_state.start_date + timedelta(days=task['ddl'])
                    days_left = (ddl_date - datetime.now()).days
                    
                    if days_left > 0:
                        st.metric("⏰ 剩余", f"{days_left} 天")
                    elif days_left == 0:
                        st.warning("⚠️ 今天截止")
                    else:
                        st.error(f"❌ 已超期 {abs(days_left)} 天")
                    
                    # 完成状态
                    checkbox_key = f"complete_{task_id}"
                    completed = st.checkbox(
                        "✅ 已完成",
                        value=task_status.get("completed", False),
                        key=checkbox_key
                    )
                    
                    # 检测状态变化并保存
                    if checkbox_key in st.session_state:
                        current_value = st.session_state[checkbox_key]
                        if current_value != task_status.get("completed", False):
                            progress[task_id] = progress.get(task_id, {})
                            progress[task_id]["completed"] = current_value
                            progress[task_id]["completed_date"] = datetime.now().strftime("%Y-%m-%d") if current_value else None
                            save_progress(progress)
                            if current_value:
                                st.balloons()

# 底部统计
st.markdown("---")
st.subheader("📊 学习统计")

col1, col2, col3 = st.columns(3)

total_tasks = sum(len(stage["tasks"]) for stage in LEARNING_PLAN.values())
completed_tasks = sum(1 for task in progress.values() if task.get("completed"))

with col1:
    st.metric("总任务数", total_tasks)
with col2:
    st.metric("已完成", completed_tasks)
with col3:
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    st.metric("完成率", f"{completion_rate:.1f}%")

# 进度条
st.progress(completion_rate / 100)
