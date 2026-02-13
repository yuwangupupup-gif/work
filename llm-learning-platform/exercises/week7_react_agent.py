"""
Week 7 - Day 49: ReAct Agent 实现
练习目标: 手写 ReAct 循环,让模型自主调用工具
"""

from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

# 定义工具
TOOLS = {
    "Calculator": {
        "description": "执行数学计算,输入为数学表达式,如: 23 * 45",
        "function": lambda expr: eval(expr)  # 注意:生产环境需要安全检查
    },
    "Search": {
        "description": "搜索知识,输入为搜索关键词",
        "function": lambda query: f"搜索结果: {query} 是一个很有趣的话题"
    }
}

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""

class ReActAgent:
    """ReAct Agent 实现"""
    
    def __init__(self, tools, max_iterations=5):
        self.tools = tools
        self.max_iterations = max_iterations
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _build_prompt(self, question):
        """构建 Prompt"""
        tool_descriptions = "\n".join([
            f"{name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
        tool_names = ", ".join(self.tools.keys())
        
        return REACT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            question=question
        )
    
    def _parse_action(self, text):
        """解析 Action 和 Action Input"""
        action_pattern = r"Action:\s*(\w+)"
        input_pattern = r"Action Input:\s*(.+?)(?:\n|$)"
        
        action_match = re.search(action_pattern, text)
        input_match = re.search(input_pattern, text)
        
        if action_match and input_match:
            return action_match.group(1), input_match.group(1).strip()
        return None, None
    
    def run(self, question):
        """运行 ReAct 循环"""
        prompt = self._build_prompt(question)
        agent_scratchpad = ""
        
        print(f"🤖 Agent 开始思考: {question}\n")
        print("=" * 60)
        
        for i in range(self.max_iterations):
            # 调用 LLM
            response = self.client.chat.completions.create(
                model="gpt-4",  # 建议用 GPT-4,GPT-3.5 推理能力较弱
                messages=[{"role": "user", "content": prompt + agent_scratchpad}],
                temperature=0,
                max_tokens=500
            )
            
            output = response.choices[0].message.content
            agent_scratchpad += output
            
            print(f"\n💭 Iteration {i+1}:")
            print(output)
            
            # 检查是否得到最终答案
            if "Final Answer:" in output:
                final_answer = output.split("Final Answer:")[-1].strip()
                print("\n" + "=" * 60)
                print(f"✅ 最终答案: {final_answer}")
                return final_answer
            
            # 解析并执行 Action
            action, action_input = self._parse_action(output)
            
            if action and action in self.tools:
                try:
                    observation = self.tools[action]["function"](action_input)
                    agent_scratchpad += f"\nObservation: {observation}\nThought:"
                    print(f"🔧 执行工具: {action}({action_input}) -> {observation}")
                except Exception as e:
                    agent_scratchpad += f"\nObservation: Error: {str(e)}\nThought:"
            else:
                # 如果没有正确格式,提示 Agent
                agent_scratchpad += "\nObservation: Invalid action format. Please use the correct format.\nThought:"
        
        return "达到最大迭代次数,未找到答案"

def main():
    """示例运行"""
    agent = ReActAgent(TOOLS)
    
    questions = [
        "123 乘以 456 等于多少?",
        "搜索一下机器学习是什么,然后计算 2 的 8 次方",
    ]
    
    for q in questions:
        print("\n" + "#" * 60)
        agent.run(q)
        print("\n")

if __name__ == "__main__":
    main()

# 运行: python exercises/week7_react_agent.py
