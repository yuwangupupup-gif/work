"""
Week 1 - Day 7: 命令行翻译助手
练习目标: 学会调用 OpenAI API 实现基础对话功能
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def create_translator():
    """创建翻译助手"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def translate(text: str) -> str:
        """
        翻译函数
        Args:
            text: 待翻译文本
        Returns:
            翻译结果
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a translator. Detect the source language and translate to Chinese if it's not Chinese, or translate to English if it's Chinese."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"翻译失败: {str(e)}"
    
    return translate

def main():
    """主程序"""
    print("🌍 命令行翻译助手 (输入 'quit' 退出)")
    print("-" * 50)
    
    translator = create_translator()
    
    while True:
        text = input("\n请输入要翻译的内容: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 再见!")
            break
        
        if not text:
            continue
        
        print("\n翻译中...")
        result = translator(text)
        print(f"📝 翻译结果: {result}")

if __name__ == "__main__":
    main()

# ============ 测试用例 ============
def test_translator():
    """单元测试"""
    translator = create_translator()
    
    # 测试英译中
    result1 = translator("Hello, world!")
    assert "你好" in result1 or "世界" in result1
    
    # 测试中译英
    result2 = translator("你好世界")
    assert "hello" in result2.lower() or "world" in result2.lower()
    
    print("✅ 所有测试通过!")

# 运行: python exercises/week1_translator.py
# 测试: pytest exercises/week1_translator.py::test_translator
