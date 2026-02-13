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
    "第一阶段：认知与基础 (Week 1)": {
        "tasks": [
            {
                "name": "理论基础",
                "ddl": 3,
                "resources": [
                    {"title": "The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/"},
                    {"title": "Let's build GPT (Karpathy)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY"}
                ],
                "exercise": "绘制 Transformer 架构图，标注 Self-Attention 的 Q, K, V 计算流程",
                "hint": "重点标注 Encoder/Decoder 堆叠、Multi-Head Attention、Feed Forward"
            },
            {
                "name": "Prompt Engineering",
                "ddl": 5,
                "resources": [
                    {"title": "吴恩达 Prompt 课程", "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"},
                    {"title": "Prompt 指南", "url": "https://www.promptingguide.ai/zh"}
                ],
                "exercise": "编写 Linux Terminal Prompt",
                "hint": "Prompt: I want you to act as a linux terminal..."
            },
            {
                "name": "API 调用",
                "ddl": 7,
                "resources": [
                    {"title": "OpenAI API", "url": "https://platform.openai.com/docs/quickstart"},
                    {"title": "LangChain", "url": "https://python.langchain.com/docs/get_started/quickstart"}
                ],
                "exercise": "命令行翻译助手",
                "hint": "使用 openai.ChatCompletion.create() 调用 API"
            }
        ]
    },
    "第二阶段：RAG 开发 (Week 2-3)": {
        "tasks": [
            {
                "name": "向量数据库",
                "ddl": 10,
                "resources": [
                    {"title": "Vector Embeddings", "url": "https://www.pinecone.io/learn/vector-embeddings/"},
                    {"title": "ChromaDB", "url": "https://docs.trychroma.com/getting-started"}
                ],
                "exercise": "文档切片 + Embedding + ChromaDB 语义搜索",
                "hint": "DocumentLoader -> text-embedding -> collection.add()"
            },
            {
                "name": "RAG 流程",
                "ddl": 14,
                "resources": [
                    {"title": "LangChain RAG", "url": "https://python.langchain.com/docs/use_cases/question_answering/"}
                ],
                "exercise": "个人知识库问答机器人",
                "hint": "retriever = vectorstore.as_retriever()"
            },
            {
                "name": "进阶 RAG",
                "ddl": 21,
                "resources": [
                    {"title": "Advanced RAG", "url": "https://www.pinecone.io/learn/advanced-rag/"}
                ],
                "exercise": "增加引用来源标注",
                "hint": "在 Prompt 中要求输出引用索引"
            }
        ]
    },
    "第三阶段：模型微调 (Week 4-6)": {
        "tasks": [
            {
                "name": "微调基础",
                "ddl": 28,
                "resources": [
                    {"title": "PEFT 文档", "url": "https://huggingface.co/docs/peft/index"}
                ],
                "exercise": "Colab 运行 Qwen-7B 推理",
                "hint": "AutoModelForCausalLM.from_pretrained()"
            },
            {
                "name": "数据准备",
                "ddl": 32,
                "resources": [
                    {"title": "Alpaca Dataset", "url": "https://github.com/tatsu-lab/stanford_alpaca"}
                ],
                "exercise": "构建 50-100 条微调数据集",
                "hint": "JSON 格式: instruction, input, output"
            },
            {
                "name": "LoRA 实战",
                "ddl": 42,
                "resources": [
                    {"title": "LLaMA-Factory", "url": "https://github.com/hiyouga/LLaMA-Factory"}
                ],
                "exercise": "微调猫娘/面试官风格模型",
                "hint": "llamafactory-cli train --stage sft"
            }
        ]
    },
    "第四阶段：Agent 与落地 (Week 7-8)": {
        "tasks": [
            {
                "name": "Agent 原理",
                "ddl": 49,
                "resources": [
                    {"title": "ReAct Paper", "url": "https://arxiv.org/abs/2210.03629"}
                ],
                "exercise": "ReAct 循环调用计算器",
                "hint": "Question -> Thought -> Action -> Action Input"
            },
            {
                "name": "多 Agent",
                "ddl": 53,
                "resources": [
                    {"title": "MetaGPT", "url": "https://github.com/geekan/MetaGPT"}
                ],
                "exercise": "双 Agent Code Review",
                "hint": "UserProxy + Assistant"
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
