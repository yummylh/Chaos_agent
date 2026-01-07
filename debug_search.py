from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. 连接数据库
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings, 
    collection_name="chaos_science_db"
)

# 2. 裸搜 (不经过 Rerank，不经过阈值过滤)
query = "Gierer-Meinhardt"
print(f"🔍 正在数据库底层搜索: '{query}' ...")
docs = vectorstore.similarity_search_with_score(query, k=5)

# 3. 打印“尸体”
print(f"\n找到 {len(docs)} 条原始结果 (Score 越低越相似):")
for doc, score in docs:
    print(f"\n--- [Score: {score:.4f}] ---")
    print(f"📄 来源: {doc.metadata.get('source', '未知')}")
    # 打印前 200 个字符，看看是不是乱码
    print(f"📝 内容片段: {doc.page_content[:200]}...")