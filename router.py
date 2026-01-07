import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def init_router_chain(llm_model):
    """
    初始化语义路由链 (针对 自我介绍 和 介绍理论 做了特训)
    """
    system_prompt = """
    你是一个严格的意图分类判官。请分析用户的输入，从 [COMPUTE, RAG, CHAT] 中选择且仅选择一个标签。

    ⚠️ **最高优先级判别规则** ⚠️
    1. **COMPUTE** 必须满足：用户明确要求"计算/画图/模拟" **并且** 提供了具体的数值参数（如 r=3.5, sigma=10）。
    2. 如果用户只是提到"方程/模型/映射"，但**没有**提供具体数值，或者是在问"是什么/定义/含义"，**必须**选 **RAG**。

    【典型案例教学】(请严格模仿以下逻辑)
    
    ❌ 错例 (千万别学): 
    用户: "Logistic方程是什么？" -> COMPUTE (错误！没给数值，是在问定义)
    
    ✅ 正例 (请照做):
    用户: "Logistic方程是什么？" -> RAG
    用户: "Logistic映射的定义" -> RAG
    用户: "介绍一下洛伦兹方程" -> RAG
    用户: "它的参数r范围是多少？" -> RAG
    
    用户: "计算r=3.5时的Logistic映射" -> COMPUTE
    用户: "画出Lorenz吸引子" -> COMPUTE (画图通常隐含默认参数，算COMPUTE)
    用户: "帮我仿真一下这个方程" -> COMPUTE

    【分类标签定义】
    1. COMPUTE: 数值计算、代码仿真、绘图。
    2. RAG: 概念查询、定义解释、参数范围查询、原理说明。
    3. CHAT: 闲聊、问候。

    【输出】
    只输出一个单词，不要加标点。
    """
    
    # Few-Shot: 给它几个易错的例子作为样本
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "你好，介绍一下你自己"),
        ("ai", "CHAT"),  # <--- 强制教它
        ("human", "介绍一下Logistic映射"),
        ("ai", "RAG"),   # <--- 形成对比
        ("human", "r=3.2 是混沌吗"),
        ("ai", "COMPUTE"),
        ("human", "{question}"),
    ])

    return route_prompt | llm_model | StrOutputParser()

def get_route_category(query, router_chain):
    """
    执行分类 (关键词规则 + LLM 智能分类)
    """
    # --- [Debug] ---
    print(f"\n{'='*20} 🚦 ROUTER DEBUG {'='*20}")
    display_query = query[:100] + "..." if len(query) > 100 else query
    print(f"📥 [Input]: {display_query}")

    # =====================================================
    # 1. 规则优先 (Rule-Based Override)
    # =====================================================
    
    # 规则 A: 计算题 (硬核关键词)
    compute_keywords = ["计算指标"]
    if any(k in query for k in compute_keywords):
        print(f"⚡ [Fast Track]: 命中计算关键词")
        print(f"🎯 [Decision]: COMPUTE (强制)")
        print(f"{'='*54}\n")
        return "COMPUTE"

    # 规则 B: 闲聊/自我介绍 (★★★ 新增修复 ★★★)
    # 如果包含这些词，大概率不需要查论文
    chat_keywords = ["你好", "你是谁", "介绍一下你自己", "介绍一下自己", "你是?", "hi", "hello"]
    # 注意：不能光查"介绍"，因为"介绍一下混沌"是RAG。必须查"介绍"+"自己/你"。
    if any(k in query.lower() for k in chat_keywords):
        print(f"⚡ [Fast Track]: 命中闲聊关键词")
        print(f"🎯 [Decision]: CHAT (强制)")
        print(f"{'='*54}\n")
        return "CHAT"

    # =====================================================
    # 2. LLM 智能判断 (如果规则没命中)
    # =====================================================
    try:
        print("🤖 [LLM Analysis]: 正在思考分类...")
        
        raw_output = router_chain.invoke({"question": query})
        print(f"📝 [Raw Output]: '{raw_output}'")
        
        category = raw_output.strip().upper()
        
        # 归一化
        final_category = "CHAT"
        if "COMPUTE" in category: final_category = "COMPUTE"
        elif "RAG" in category: final_category = "RAG"
        else: final_category = "CHAT"
            
        print(f"✅ [Final Decision]: {final_category}")
        print(f"{'='*54}\n")
        return final_category

    except Exception as e:
        print(f"❌ [Error]: {e}")
        return "CHAT"