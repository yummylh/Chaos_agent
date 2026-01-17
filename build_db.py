import os
import glob
import shutil
import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
# 核心组件：结构化切分器
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# =================配置区域=================
PERSIST_DIRECTORY = "./chroma_db"
# 确保这里指向你存放 DeepSeek 清洗后 Markdown 文件的目录
DATA_DIRECTORY = "./data" 
BATCH_SIZE = 30
# EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "BAAI/bge-m3"
# =========================================

def intelligent_chunking(documents):
    """
    【核心升级】结构化语义切分 + 上下文注入
    实现面试中提到的 "Structure-aware Semantic Chunking"
    """
    print(f"🔪 [Chunking] 开始对 {len(documents)} 份文档进行智能切分...")
    final_chunks = []
    
    # 1. 定义 Markdown 标题层级 (DeepSeek 清洗后的数据通常包含这些)
    headers_to_split_on = [
        ("#", "Title"),      # 一级标题
        ("##", "Section"),   # 二级标题 (章节)
        ("###", "Subsection"), # 三级标题 (小节)
    ]
    
    # 2. 初始化切分器
    # 逻辑层：按 Markdown 结构切
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 物理层：处理超长段落的兜底方案 (窗口大小略大于 Embedding 限制)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       
        chunk_overlap=50,     
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] 
    )

    for doc in documents:
        # 获取原始内容和源文件名
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")
        
        # Step 1: 按 Markdown 结构粗切
        # 这一步出来的 chunk 会自动带有 metadata={'Section': '...', 'Title': '...'}
        md_header_splits = markdown_splitter.split_text(content)

        # Step 2: 遍历粗切后的片段，进行细切和上下文注入
        for split in md_header_splits:
            # 继承源文件名
            split.metadata["source"] = source
            
            # 如果片段本身就很小 (比如 < 800 字符)，不用再切，保持逻辑完整性
            if len(split.page_content) < 800:
                sub_splits = [split]
            else:
                # 超长片段，进行滑动窗口细切
                sub_splits = text_splitter.split_documents([split])
            
            # Step 3: ★★★ 元数据注入 (Metadata Injection) ★★★
            for sub_split in sub_splits:
                # 从 metadata 提取标题结构
                title = sub_split.metadata.get("Title", "")
                section = sub_split.metadata.get("Section", "")
                subsection = sub_split.metadata.get("Subsection", "")
                
                # 构造上下文前缀 (面包屑导航)
                # 格式示例：【文档：混沌理论】【章节：Logistic映射】
                context_prefix = ""
                if title: context_prefix += f"【主题: {title}】"
                if section: context_prefix += f"【章节: {section}】"
                if subsection: context_prefix += f"【小节: {subsection}】"
                
                # 将上下文拼接到正文头部
                # 这样 Embedding 向量就会包含这些层级信息，检索准确率大幅提升
                if context_prefix:
                    sub_split.page_content = f"{context_prefix}\n{sub_split.page_content}"
                
                final_chunks.append(sub_split)

    print(f"✅ [Chunking] 切分完成，生成 {len(final_chunks)} 个语义片段 (已注入上下文元数据)。")
    return final_chunks

def build_vector_db():
    print("🚀 开始构建向量数据库 ...")
    
    # 1. 强制清空旧数据库 (防止旧的垃圾切片残留)
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"🗑️ 检测到旧数据库 {PERSIST_DIRECTORY}，正在删除重建...")
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            time.sleep(1) # 歇一秒，防止 Windows 文件占用报错
        except Exception as e:
            print(f"⚠️ 删除失败: {e}，尝试继续...")

    # 2. 连接 Embedding
    print(f"🔌 连接 BGE-M3 模型: {EMBEDDING_MODEL}...")
    try:
        # 显式指定 device='cpu' 以节省显存
        # 开启 normalize_embeddings 以优化余弦相似度检索
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        # 简单测试一下，触发模型下载（如果第一次运行）
        embeddings.embed_query("test")
        print("✅ BGE-M3 模型加载成功！")
    except Exception as e:
        print(f"❌ 连接失败，请检查sentence-transformers是否安装: {e}")
        return

    # 3. 加载 Markdown 文件
    # 优先加载 .md，因为那是 DeepSeek 清洗后的精华
    docs = []
    files = glob.glob(os.path.join(DATA_DIRECTORY, "*.md")) + glob.glob(os.path.join(DATA_DIRECTORY, "*.txt"))
    
    print(f"📂 发现 {len(files)} 个数据文件 (.md/.txt)")
    if len(files) == 0:
        print("❌ 错误：未找到数据文件！请确保 ./data 目录下有清洗好的 Markdown 文件。")
        return

    for file_path in files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            loaded_docs = loader.load()
            # 记录文件名元数据
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(file_path)
            docs.extend(loaded_docs)
            print(f"  - 已加载: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  - ❌ 加载失败 {file_path}: {e}")

    # 4. 执行智能切分 (替代原来的 TextSplitter)
    if not docs:
        return
        
    chunks = intelligent_chunking(docs)

    # 5. 写入数据库
    print(f"💾 开始写入 Chroma (Batch Size = {BATCH_SIZE})...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="chaos_science_db"
    )

    # 分批写入，显示进度条效果
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            # 简单的进度打印
            progress = ((i + len(batch)) / total_chunks) * 100
            print(f"\r  - 写入进度: {progress:.1f}% ({i + len(batch)}/{total_chunks})", end="")
        except Exception as e:
            print(f"\n  ⚠️ 批次写入失败: {e}")

    print("\n\n知识库构建完成！")
    # print("👉 你的数据现在拥有了【结构化上下文】，快去 app.py 提问试试！")

if __name__ == "__main__":
    build_vector_db()
