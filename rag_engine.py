import os
import glob
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import Tool
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PDFPlumberLoader,PyPDFLoader
import time
from tqdm import tqdm # 如果没有安装 tqdm，可以把下面的 tqdm(range(...)) 改为 range(...)
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. 资源初始化 (单例模式)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_reranker():
    """
    加载重排模型，BGE-Reranker
    """
    print("加载BGE-Rerank模型...")
    return CrossEncoder('BAAI/bge-reranker-base', device='cpu')

@st.cache_resource(show_spinner=False)
def setup_knwoledge_base():
    """
    【轻量版】仅负责加载已存在的知识库，不负责构建。
    构建工作交由 build_db.py 独立完成。
    """
    persist_directory = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 1. 检查数据库是否存在
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print("📖 [App] 成功加载现有 Chroma 知识库...")
        return Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings, 
            collection_name="chaos_science_db"
        )
    
    # 2. 如果不存在，直接报错 (不再尝试现场构建，防止显存爆炸)
    else:
        print("❌ [App] 严重错误：未找到本地知识库！")
        print("   -> 请先运行 'python build_db.py' 生成数据库。")
        return None
    
#初始化全局资源 
reranker_model = load_reranker()
vectorstore = setup_knwoledge_base()

# ===========================================
#2.核心检索逻辑（Advance RAG）
# ===========================================
llm_rewriter = ChatOllama(
    model="llama3.1", 
    temperature=0.1, # 重写需要精确，温度调低
    # base_url="http://127.0.0.1:11434"
)

def rewrite_query(user_input: str) -> str:
    """
    Day 7.5 新增：利用 LLM 将用户的模糊提问改写为适合检索的独立句子
    注意：为了简化，这里暂时没传 history，实际项目中可以结合 st.session_state 传入
    """
    try:
        # 定义提示词
        prompt = ChatPromptTemplate.from_template(
            """你是一个关键词提取工具。你的唯一任务是优化搜索词。
        
        【负面约束】
        - 不要回答问题。
        - 不要输出 "好的"、"重写如下" 这种废话。
        - 不要过度联想（比如问 A 不要扩展到 B）。

        【学习以下示例】
        User: "它有什么优点"
        Output: Logistic映射 优点 优势 (假设上下文是Logistic)
        
        User: "OGY控制"
        Output: OGY控制 Ott-Grebogi-Yorke chaos control
        
        User: "计算r=3.5"
        Output: 计算 r=3.5 数值模拟
        
        User: {input}
        Output:"""
        )
        
        # 执行链
        chain = prompt | llm_rewriter
        rewritten_query = chain.invoke({"input": user_input}).content.strip()
        
        # 简单清洗，防止 LLM 废话
        if ":" in rewritten_query:
            rewritten_query = rewritten_query.split(":")[-1].strip()
            
        print(f"🔄 [Rewrite] 原始: '{user_input}' -> 重写: '{rewritten_query}'")
        return rewritten_query
        
    except Exception as e:
        print(f"⚠️ [Rewrite Error] 重写失败，使用原句: {e}")
        return user_input


def advanced_rerank_search(query: str):
    """
    DAY3核心逻辑:Retrieve(recall)-> Rerank(Precision)
    """
    if not vectorstore:
        return "错误：知识库未初始化，请检查data文件夹。"
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 核心修复：在这里调用重写函数！
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    print(f"\n🚀 [RAG Start] 用户原始输入: {query}")
    # effective_query = rewrite_query(query) 
    effective_query = query
    # 1.[Recall]粗筛
    try:
        # 注意：这里我们要用 effective_query (重写后的) 去查库
        initial_docs = vectorstore.similarity_search(effective_query, k=30)
        
        print(f"\n🔍 [Recall Debug] 检索词: '{effective_query}' | 召回: {len(initial_docs)} 条文档。")
    except Exception as e:
        return f"错误：检索失败，{e}"
    
    if not initial_docs:
        print("❌ [Recall Debug] 第一步检索结果为空！")
        return "错误：未找到相关文档。"
    
    # 2.[Rerank]打分
    # 技巧：Rerank 的时候，是用“重写后的查询”还是“原始查询”去和文档比对？
    # 通常用重写后的更准，因为它包含了全称和英文。
    pairs = [[effective_query, doc.page_content] for doc in initial_docs]
    scores = reranker_model.predict(pairs)

    # 3.排序与过滤
    doc_score_pairs = list(zip(initial_docs, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    final_results = []

    print(f"\n====== Rerank Debug (Query: {effective_query}) ======")
    #仅仅取top5
    for doc, score in doc_score_pairs[:5]:
        if score > 0.3: 
            print(f"✅ [Accepted] Score: {score:.4f} | Content: {doc.page_content[:30]}...")
            final_results.append(doc.page_content)
        else:
            print(f"❌ [Rejected] Score: {score:.4f} | Content: {doc.page_content[:30]}...")

    print("==========================================\n")

    if not final_results:
        return "资料不足，相关评分过低"
    
    return "\n\n".join(final_results)

# ===========================================
#3.工具封装导出
# ===========================================
def get_retriever_tool():
    """
    将检索逻辑封装为工具
    """
    return Tool(
        name="search_chaos_knowledge",
        func=advanced_rerank_search,
        description="Search for scientific definitions, theories, and formulas. Use this for questions about Chaos Theory, Meteorology, and specific terms like 'ODGY method'."
    )