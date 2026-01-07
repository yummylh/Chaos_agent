import os
import io
from dotenv import load_dotenv
load_dotenv()
# ★★★ 设置 Hugging Face 镜像源 (防止下载模型超时) ★★★
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from router import init_router_chain,get_route_category
# === 导入我们自己写的模块 ===
from router import init_router_chain, get_route_category
from rag_engine import get_retriever_tool, advanced_rerank_search
import tools 
import re
import history_utils
# ==========================================
# 🔌 核心升级：导入后端引擎
# ==========================================
# 这一行代码就把 PDF 解析、向量库、Rerank 重排序全搞定了
# try:

# except ImportError:
    # st.error("❌ 找不到 rag_engine.py！请确保该文件在同一目录下。")
    # st.stop()

# ================= 2. 页面配置与初始化 =================
st.set_page_config(page_title="Chaos Agent Pro", page_icon="🌪️", layout="wide")

# 初始化 Session State (用于存储当前会话信息)
if "session_id" not in st.session_state:
    st.session_state.session_id = history_utils.generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 Router (只加载一次)
if "router_chain" not in st.session_state:
    # 这里加载 LLM 只用于路由，可以轻量化
    llm_router = ChatOllama(model="llama3.1", temperature=0, base_url="http://127.0.0.1:11434")
    st.session_state.router_chain = init_router_chain(llm_router)

# 初始化主 LLM (用于生成回答)
@st.cache_resource
def load_main_llm():
    return ChatOllama(
        model="llama3.1",
        temperature=0.3,
        keep_alive="1h",
        # base_url="http://127.0.0.1:11434"
    )

llm = load_main_llm()

# ================= 3. 侧边栏 (记忆功能核心) =================
with st.sidebar:
    st.title("🗂️ 历史记录")
    
    # [新建对话]
    if st.button("➕ 新建对话", use_container_width=True):
        st.session_state.session_id = history_utils.generate_session_id()
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # [历史列表] 读取 JSON 文件
    sessions = history_utils.get_history_list()
    
    for sess in sessions:
        # 判断是不是当前选中的会话
        is_current = (sess["id"] == st.session_state.session_id)
        btn_type = "primary" if is_current else "secondary"
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            # 点击标题加载历史
            if st.button(f"📄 {sess['title']}", key=f"btn_{sess['id']}", type=btn_type, use_container_width=True):
                st.session_state.session_id = sess["id"]
                st.session_state.messages = history_utils.load_conversation(sess["id"])
                st.rerun()
        with col2:
            # 删除按钮
            if st.button("🗑️", key=f"del_{sess['id']}"):
                history_utils.delete_conversation(sess["id"])
                if sess["id"] == st.session_state.session_id:
                    # 如果删的是当前会话，重置
                    st.session_state.session_id = history_utils.generate_session_id()
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    st.info("💡 **工作模式:**\n1. 🧮 数学 -> Python 引擎\n2. 📄 专业 -> 本地知识库\n3. 🧠 通用 -> Llama3")

# ================= 4. 主界面显示区域 =================
st.title("🌪️ Chaos-Agent V1.0 (Hybrid Engine)")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 5. 核心处理逻辑 =================
if user_input := st.chat_input("请输入问题 (例如: 计算r=3.2的状态 / Gierer-Meinhardt模型是什么)..."):
    
    # [记录用户输入]
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # [意图识别与路由]
    with st.status("🧠 正在思考...", expanded=True) as status:
        category = get_route_category(user_input, st.session_state.router_chain)
        status.write(f"🏷️ 识别意图: **{category}**")
        
        response_text = ""
        fig = None #用于存储可能生成的图片

        # ➤ 分支 A: 数学计算 (调用 tools.py)
        if category == "COMPUTE":
            status.update(label="🧮 正在调用 Python 计算引擎...", state="running")
            try:
                # 正则提取 r 值
                match = re.search(r"r\s*[=:]\s*(\d+\.?\d*)", user_input)
                r_val = float(match.group(1)) if match else 3.5 # 默认值
                
                if "logistic" in user_input.lower() or "映射" in user_input or "方程" in user_input:
                    response_text, fig = tools.simulate_logistic_map(r_val)
                elif "lorenz" in user_input.lower() or "洛伦兹" in user_input:
                    response_text, fig = tools.simulate_lorenz()
                else:
                    response_text = "⚠️ 未识别具体计算模型，默认计算 Logistic 映射..."
                    response_text_extra, fig = tools.simulate_logistic_map(r_val)
                    response_text += "\n" + response_text_extra
                
                # ★★★ 核心修复：将 Matplotlib Figure 转为内存图片 ★★★
                # 这能防止 Streamlit 报 MediaFileHandler Error
                if fig:
                    # 1. 创建内存缓冲区
                    buf = io.BytesIO()
                    # 2. 把图保存到缓冲区
                    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
                    # 3. 指针归零
                    buf.seek(0)
                    # 4. 显示图片 (使用 st.image 而不是 st.pyplot)
                    st.image(buf, caption="Simulation Result", use_container_width=True)
                    # 5. 显式关闭图表，释放内存
                    plt.close(fig) 
                    
                    # (可选) 如果你想把图存进历史记录，这里需要把 buf 转为 base64 存入 session_state
                    # 但为了简单稳定，目前历史记录只存文字，图只显示一次。

            except Exception as e:
                response_text = f"❌ 计算模块出错: {str(e)}"

        # ➤ 分支 B: RAG + 智能回退
        elif category == "RAG":
            status.update(label="🔍 正在检索本地知识库...", state="running")
            
            # 1. 检索
            retriever = get_retriever_tool()
            rag_result = retriever.func(user_input)
            
            # 2. 判别是否需要回退 (Fallback)
            # 假设 rag_engine 在没搜到时会返回包含"资料不足"的字符串，或者我们可以检查字符串长度
            is_fallback = False
            if "资料不足" in rag_result or "未找到相关文档" in rag_result:
                is_fallback = True
                status.write("⚠️ 本地库未收录，**切换至通用模式**...")
            else:
                status.write("✅ 本地库命中！正在阅读文献...")

            # 3. 生成回答
            if is_fallback:
                prompt = ChatPromptTemplate.from_template(
                    "用户问题: {question}\n请利用你的通用知识回答。如果不知道就直说。"
                )
                chain = prompt | llm
                response_text = chain.invoke({"question": user_input}).content
                response_text += "\n\n*(注: 此回答基于通用知识，非本地文献)*"
            else:
                prompt = ChatPromptTemplate.from_template(
                    "你是一个严谨的研究助手。请仅基于以下文献回答问题：\n\n文献内容:\n{context}\n\n用户问题: {question}"
                    "在回答用户问题时，如果涉及到输出内容有公式，请严格按照Latex进行公式输出"
                    "在回答用户问题时，如果段落过长需要分点回答，请分点回答，并按照一级标题-内容进行输出"
                )
                chain = prompt | llm
                response_text = chain.invoke({"context": rag_result, "question": user_input}).content
                # 可以在这里加个前缀，让 UI 更好看
                response_text = "📚 **基于本地文献的回答：**\n\n" + response_text
        # ➤ 分支 C: 闲聊
        else:
            status.update(label="💬 正在生成回复...", state="running")
            prompt = ChatPromptTemplate.from_template("用户说: {question}\n请用简练、友好的语气回复。"
                                                      "在回答数值计算问题时，严格根据工具返回的结果进行判断，自己别乱说"
                                                      )
            chain = prompt | llm
            response_text = chain.invoke({"question": user_input}).content

        status.update(label="✅ 完成", state="complete", expanded=False)

    # [显示助手回复]
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if fig:
            st.pyplot(fig) # ★★★ 如果有图，在这里显示 ★★★

    # [保存消息到 Session]
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # [自动持久化保存]
    # 调用 history_utils 把当前完整对话存入 JSON
    history_utils.save_conversation(st.session_state.session_id, st.session_state.messages)
    
    # 如果是新对话（第一轮交互），刷新一下让侧边栏出现标题
    if len(st.session_state.messages) <= 2:
        st.rerun()