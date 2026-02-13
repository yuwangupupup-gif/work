"""
Week 2 - Day 14: 个人知识库问答机器人 (RAG)
练习目标: 掌握 RAG 全流程
"""

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBaseQA:
    """个人知识库问答系统"""
    
    def __init__(self, persist_directory="./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self, file_paths):
        """加载文档"""
        documents = []
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())
        return documents
    
    def create_vectorstore(self, documents):
        """创建向量数据库"""
        # 文档切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # 存入 ChromaDB
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print(f"✅ 已处理 {len(splits)} 个文档块")
    
    def build_qa_chain(self):
        """构建问答链"""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # 自定义 Prompt
        template = """使用以下上下文回答问题。如果不知道答案,就说不知道,不要编造答案。

上下文: {context}

问题: {question}

回答:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def query(self, question):
        """查询问答"""
        if not self.qa_chain:
            raise ValueError("请先调用 build_qa_chain()")
        
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

def main():
    """示例用法"""
    # 初始化
    kb = KnowledgeBaseQA()
    
    # 加载你的笔记文件 (支持 .txt, .pdf)
    documents = kb.load_documents([
        "data/my_notes.txt",  # 替换为你的文件路径
    ])
    
    # 构建向量库
    kb.create_vectorstore(documents)
    kb.build_qa_chain()
    
    # 问答
    print("💬 知识库问答 (输入 'quit' 退出)")
    print("-" * 50)
    
    while True:
        question = input("\n❓ 请提问: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = kb.query(question)
        print(f"\n💡 回答: {result['answer']}")
        print(f"\n📚 引用来源:")
        for i, doc in enumerate(result['sources'], 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()

# 运行: python exercises/week2_rag.py
