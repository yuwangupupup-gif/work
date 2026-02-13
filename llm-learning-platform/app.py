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
                    {"title": "3Blue1Brown - Attention 机制", "url": "https://www.youtube.com/watch?v=eMlx5fFNoYc"},
                    {"title": "什么是大语言模型", "url": "https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/"},
                    {"title": "LLM 发展史", "url": "https://huggingface.co/blog/large-language-models"}
                ],
                "exercise": "用自己的话解释：Tokenization、Embedding、Attention、Transformer 四个概念",
                "hint": "思考：为什么 GPT 不能直接理解文字？Token 是什么？Attention 在做什么计算？"
            },
            {
                "name": "Day 3-4: Transformer 架构深入",
                "ddl": 4,
                "resources": [
                    {"title": "The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/"},
                    {"title": "Transformer 论文精读", "url": "https://www.youtube.com/watch?v=nzqlFIcCSWQ"},
                    {"title": "Let's build GPT (Karpathy)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY"}
                ],
                "exercise": "绘制 Transformer 完整架构图，手动计算一次 Self-Attention（3个词的例子）",
                "hint": "Q=WQ*X, K=WK*X, V=WV*X, Attention(Q,K,V) = softmax(QK^T/√d_k)V，重点理解 Multi-Head"
            },
            {
                "name": "Day 5-6: Prompt Engineering 基础",
                "ddl": 6,
                "resources": [
                    {"title": "吴恩达 Prompt 课程", "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"},
                    {"title": "Prompt Engineering Guide", "url": "https://www.promptingguide.ai/zh"},
                    {"title": "OpenAI Prompt 最佳实践", "url": "https://platform.openai.com/docs/guides/prompt-engineering"}
                ],
                "exercise": "掌握 6 种 Prompt 技巧：Zero-shot、Few-shot、CoT、Self-Consistency、ToT、ReAct",
                "hint": "实践：写一个旅游规划 Prompt，要求输出 JSON 格式，包含景点、预算、时间安排"
            },
            {
                "name": "Day 7-8: Prompt 进阶技巧",
                "ddl": 8,
                "resources": [
                    {"title": "Advanced Prompting", "url": "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"},
                    {"title": "Prompt 注入攻防", "url": "https://learnprompting.org/docs/prompt_hacking/injection"}
                ],
                "exercise": "实现 3 个角色 Prompt：Linux 终端、Python 解释器、面试官",
                "hint": "用 System Message 定义角色，用 Few-shot 示例约束输出格式"
            },
            {
                "name": "Day 9-10: OpenAI API 实战",
                "ddl": 10,
                "resources": [
                    {"title": "OpenAI API 文档", "url": "https://platform.openai.com/docs/quickstart"},
                    {"title": "API 参数详解", "url": "https://platform.openai.com/docs/api-reference/chat"},
                    {"title": "Token 计费规则", "url": "https://openai.com/pricing"}
                ],
                "exercise": "实现一个多轮对话翻译助手，支持上下文记忆、流式输出、Token 统计",
                "hint": "temperature、top_p、max_tokens、frequency_penalty 参数的作用，如何计算成本"
            },
            {
                "name": "Day 11-12: LangChain 框架入门",
                "ddl": 12,
                "resources": [
                    {"title": "LangChain 快速开始", "url": "https://python.langchain.com/docs/get_started/quickstart"},
                    {"title": "LangChain 核心概念", "url": "https://python.langchain.com/docs/modules/"},
                    {"title": "LCEL 表达式", "url": "https://python.langchain.com/docs/expression_language/"}
                ],
                "exercise": "用 LangChain 实现：PromptTemplate + LLM + OutputParser 的完整链路",
                "hint": "掌握 Chain、Memory、Agent 三大核心组件"
            },
            {
                "name": "Day 13-14: 模型评估与测试",
                "ddl": 14,
                "resources": [
                    {"title": "如何评估 LLM", "url": "https://huggingface.co/blog/evaluating-llm-chat-models"},
                    {"title": "MMLU/HellaSwag 基准", "url": "https://github.com/hendrycks/test"}
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
                    {"title": "Vector Embeddings 原理", "url": "https://www.pinecone.io/learn/vector-embeddings/"},
                    {"title": "text-embedding-ada-002", "url": "https://platform.openai.com/docs/guides/embeddings"},
                    {"title": "向量相似度计算", "url": "https://www.pinecone.io/learn/vector-similarity/"}
                ],
                "exercise": "理解 Cosine Similarity、Euclidean Distance、Dot Product 的区别，手动计算示例",
                "hint": "为什么 Embedding 能捕捉语义？768 维向量代表什么？归一化的作用？"
            },
            {
                "name": "Day 17-18: ChromaDB 实战",
                "ddl": 18,
                "resources": [
                    {"title": "ChromaDB 快速开始", "url": "https://docs.trychroma.com/getting-started"},
                    {"title": "向量数据库对比", "url": "https://github.com/qdrant/vector-db-benchmark"}
                ],
                "exercise": "实现文档切片 → Embedding → 存储 → 语义搜索完整流程",
                "hint": "RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)"
            },
            {
                "name": "Day 19-20: RAG 核心流程",
                "ddl": 20,
                "resources": [
                    {"title": "LangChain RAG", "url": "https://python.langchain.com/docs/use_cases/question_answering/"},
                    {"title": "RAG 论文", "url": "https://arxiv.org/abs/2005.11401"}
                ],
                "exercise": "构建个人知识库问答系统（支持 PDF/Markdown 导入）",
                "hint": "Retriever → Prompt → LLM → Answer，注意 Context 长度控制"
            },
            {
                "name": "Day 21: 进阶 RAG 优化",
                "ddl": 21,
                "resources": [
                    {"title": "Advanced RAG", "url": "https://www.pinecone.io/learn/advanced-rag/"},
                    {"title": "Reranking 技术", "url": "https://www.sbert.net/examples/applications/cross-encoder/README.html"}
                ],
                "exercise": "实现 Hybrid Search（BM25 + Vector）+ Reranking + 引用来源标注",
                "hint": "检索 Top-20 → Rerank → 取 Top-5 → 注入 Prompt，输出 [来源1][来源2]"
            }
        ]
    },
    "第三阶段：模型微调 (Week 4-7)": {
        "tasks": [
            {
                "name": "Day 15-17: 微调理论基础",
                "ddl": 17,
                "resources": [
                    {"title": "Fine-tuning 原理", "url": "https://huggingface.co/blog/fine-tune-llms"},
                    {"title": "LoRA 论文解读", "url": "https://arxiv.org/abs/2106.09685"},
                    {"title": "PEFT 文档", "url": "https://huggingface.co/docs/peft/index"}
                ],
                "exercise": "理解 4 种微调方法：Full Fine-tuning、Adapter、Prefix Tuning、LoRA 的区别",
                "hint": "对比参数量、显存占用、训练速度、效果。为什么 LoRA 只训练 0.1% 参数却效果好？"
            },
            {
                "name": "Day 18-20: 环境搭建与模型加载",
                "ddl": 20,
                "resources": [
                    {"title": "Transformers 快速开始", "url": "https://huggingface.co/docs/transformers/quicktour"},
                    {"title": "模型量化 (4bit/8bit)", "url": "https://huggingface.co/blog/4bit-transformers-bitsandbytes"},
                    {"title": "Accelerate 库", "url": "https://huggingface.co/docs/accelerate/index"}
                ],
                "exercise": "在 Colab (T4 GPU) 加载 Qwen-7B-Chat，实现 4bit 量化推理",
                "hint": "使用 BitsAndBytesConfig + load_in_4bit=True 节省显存，from_pretrained 参数详解"
            },
            {
                "name": "Day 21-23: 数据集构建与处理",
                "ddl": 23,
                "resources": [
                    {"title": "Alpaca 数据集", "url": "https://github.com/tatsu-lab/stanford_alpaca"},
                    {"title": "数据格式规范", "url": "https://huggingface.co/docs/datasets/about_dataset_load"},
                    {"title": "Tokenization 技巧", "url": "https://huggingface.co/docs/transformers/preprocessing"}
                ],
                "exercise": "构建 100 条高质量指令微调数据集（选择一个垂直领域：医疗/法律/编程/客服）",
                "hint": "格式：{instruction, input, output}。确保多样性：问答、总结、翻译、生成等"
            },
            {
                "name": "Day 24-26: LoRA 微调实战",
                "ddl": 26,
                "resources": [
                    {"title": "LoRA 官方代码", "url": "https://github.com/microsoft/LoRA"},
                    {"title": "PEFT + Transformers", "url": "https://huggingface.co/blog/peft"},
                    {"title": "训练参数调优", "url": "https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2"}
                ],
                "exercise": "使用 LoRA 微调 Qwen-7B，实现特定风格输出（例如：猫娘、古风、技术博主）",
                "hint": "重点参数：r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['q_proj','v_proj']"
            },
            {
                "name": "Day 27-29: QLoRA 与显存优化",
                "ddl": 29,
                "resources": [
                    {"title": "QLoRA 论文", "url": "https://arxiv.org/abs/2305.14314"},
                    {"title": "Gradient Checkpointing", "url": "https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing"},
                    {"title": "显存优化技巧", "url": "https://huggingface.co/docs/transformers/perf_train_gpu_one"}
                ],
                "exercise": "用 QLoRA 在 12GB 显卡上微调 13B 模型（对比 LoRA 的显存占用）",
                "hint": "4bit 量化 + NF4 数据类型 + double quantization，batch_size=1, gradient_accumulation_steps=4"
            },
            {
                "name": "Day 30-32: LLaMA-Factory 全流程",
                "ddl": 32,
                "resources": [
                    {"title": "LLaMA-Factory", "url": "https://github.com/hiyouga/LLaMA-Factory"},
                    {"title": "WebUI 使用教程", "url": "https://www.youtube.com/watch?v=your-tutorial"},
                    {"title": "配置文件详解", "url": "https://github.com/hiyouga/LLaMA-Factory/wiki"}
                ],
                "exercise": "用 LLaMA-Factory 完成：数据准备 → 训练 → 评估 → 导出 → 部署完整流程",
                "hint": "llamafactory-cli train --stage sft --model_name_or_path qwen --dataset alpaca_zh"
            },
            {
                "name": "Day 33-35: 全参数微调 (SFT)",
                "ddl": 35,
                "resources": [
                    {"title": "Supervised Fine-Tuning", "url": "https://huggingface.co/blog/llama2#how-to-prompt-llama-2"},
                    {"title": "DeepSpeed ZeRO", "url": "https://www.deepspeed.ai/tutorials/zero/"},
                    {"title": "FSDP 分布式训练", "url": "https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/"}
                ],
                "exercise": "理解全参数微调 vs LoRA 的适用场景，什么时候必须用全参数？",
                "hint": "领域知识注入、语言迁移需要全参数；风格调整、任务适配用 LoRA"
            },
            {
                "name": "Day 36-38: RLHF 与 DPO",
                "ddl": 38,
                "resources": [
                    {"title": "RLHF 原理", "url": "https://huggingface.co/blog/rlhf"},
                    {"title": "DPO 论文", "url": "https://arxiv.org/abs/2305.18290"},
                    {"title": "TRL 库", "url": "https://github.com/huggingface/trl"}
                ],
                "exercise": "构建偏好数据集（chosen vs rejected），理解 PPO 训练流程",
                "hint": "RLHF 三阶段：SFT → Reward Model → PPO。DPO 直接优化，无需 RM"
            },
            {
                "name": "Day 39-42: 模型评估与部署",
                "ddl": 42,
                "resources": [
                    {"title": "模型评估指标", "url": "https://huggingface.co/spaces/evaluate-metric/perplexity"},
                    {"title": "vLLM 高性能推理", "url": "https://github.com/vllm-project/vllm"},
                    {"title": "模型量化部署", "url": "https://github.com/ggerganov/llama.cpp"}
                ],
                "exercise": "评估微调后模型：PPL、BLEU、人工评分，对比微调前后差异",
                "hint": "使用 vLLM 部署，对比推理速度（tokens/s）、显存占用、并发能力"
            },
            {
                "name": "Day 43-45: 持续学习与灾难遗忘",
                "ddl": 45,
                "resources": [
                    {"title": "Catastrophic Forgetting", "url": "https://arxiv.org/abs/2002.06305"},
                    {"title": "Elastic Weight Consolidation", "url": "https://arxiv.org/abs/1612.00796"}
                ],
                "exercise": "微调后测试通用能力是否下降（加法运算、常识问答），如何缓解？",
                "hint": "混合通用数据集、控制学习率、使用 EWC 正则化"
            }
        ]
    },
    "第四阶段：Agent 开发 (Week 8)": {
        "tasks": [
            {
                "name": "Day 46-48: Agent 基础与 ReAct",
                "ddl": 48,
                "resources": [
                    {"title": "ReAct 论文", "url": "https://arxiv.org/abs/2210.03629"},
                    {"title": "LangChain Agent", "url": "https://python.langchain.com/docs/modules/agents/"},
                    {"title": "Agent 设计模式", "url": "https://lilianweng.github.io/posts/2023-06-23-agent/"}
                ],
                "exercise": "实现 ReAct Agent：Question → Thought → Action → Observation 循环",
                "hint": "工具：Calculator、Wikipedia、Weather API，最多 5 轮循环"
            },
            {
                "name": "Day 49-51: Function Calling",
                "ddl": 51,
                "resources": [
                    {"title": "OpenAI Function Calling", "url": "https://platform.openai.com/docs/guides/function-calling"},
                    {"title": "工具定义规范", "url": "https://json-schema.org/"}
                ],
                "exercise": "构建智能助手：天气查询 + 日历管理 + 邮件发送（3 个 Function）",
                "hint": "定义 JSON Schema → 模型返回 function_call → 执行函数 → 返回结果"
            },
            {
                "name": "Day 52-54: 多 Agent 协作",
                "ddl": 54,
                "resources": [
                    {"title": "AutoGen", "url": "https://github.com/microsoft/autogen"},
                    {"title": "MetaGPT", "url": "https://github.com/geekan/MetaGPT"},
                    {"title": "CrewAI", "url": "https://github.com/joaomdmoura/crewAI"}
                ],
                "exercise": "实现双 Agent Code Review：Coder (写代码) + Reviewer (审查代码)",
                "hint": "UserProxy ↔ Assistant，最多 3 轮对话达成一致"
            },
            {
                "name": "Day 55-56: Memory 与上下文管理",
                "ddl": 56,
                "resources": [
                    {"title": "LangChain Memory", "url": "https://python.langchain.com/docs/modules/memory/"},
                    {"title": "上下文窗口优化", "url": "https://github.com/hwchase17/chat-langchain"}
                ],
                "exercise": "实现 ConversationBufferMemory、ConversationSummaryMemory 并对比",
                "hint": "超过 4k tokens 如何压缩？如何保留关键信息？"
            },
            {
                "name": "Day 57-60: 完整项目实战",
                "ddl": 60,
                "resources": [
                    {"title": "LangChain 项目案例", "url": "https://github.com/langchain-ai/langchain/tree/master/templates"},
                    {"title": "Streamlit 部署", "url": "https://docs.streamlit.io/streamlit-community-cloud"}
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
