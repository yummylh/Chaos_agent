import os
import re
import json
import time
from dotenv import load_dotenv
from openai import OpenAI 

# 加载环境
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# === 导入本地 Agent 模块 ===
from langchain_ollama import ChatOllama
from router import init_router_chain, get_route_category
from rag_engine import get_retriever_tool
import tools

# ================= 配置区域 =================
QUESTION_FILE = "questions.txt"
OUTPUT_FILE = "chaos_cot_dataset.jsonl" # 改个名字，区分普通数据集
DEEPSEEK_API_KEY = "sk-3870228272a546e8a9822bf0aa4fbcc7" 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# ===========================================

# 初始化 DeepSeek (现在的角色是：老师/专家)
teacher_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 初始化本地 Llama 3 (现在的角色是：助教，只负责跑腿查资料，不负责写答案)
local_llm = ChatOllama(model="llama3.1", temperature=0, base_url="http://127.0.0.1:11434")
router_chain = init_router_chain(local_llm)

def generate_golden_cot(question, context, category):
    """
    让 DeepSeek 生成带有 <thinking> 的完美 CoT 回答
    """
    
    # 根据不同模式构建 System Prompt
    if category == "RAG":
        system_prompt = """你是一个混沌理论领域的专家教授。
你的任务是根据提供的【参考文献】回答用户问题。

【至关重要的要求 - 思维链 CoT】
请在回答前，先进行深度的逻辑推理。
你的输出必须包含 <thinking> 和 <answer> 两个标签。
在 <thinking> 标签中，请写出：
1. 意图分析：用户想问什么？
2. 信息提取：参考文献中哪句话是关键？
3. 逻辑推导：如果文献没有直接答案，如何根据原理推导？
4. 冲突检查：是否存在矛盾信息？

格式示例：
<thinking>
用户询问 Logistic 映射稳定性... 文献提到 r>3.5699 进入混沌... 推导可知...
</thinking>
<answer>
这里是最终给用户的回答...
</answer>
"""
    else: # COMPUTE 模式
        system_prompt = """你是一个精通 Python 仿真与非线性动力学的专家。
用户已经通过 Python 脚本运行了仿真，你需要根据【仿真结果】来解释现象。

【至关重要的要求 - 思维链 CoT】
请在回答前，先解析数据。
你的输出必须包含 <thinking> 和 <answer> 两个标签。
在 <thinking> 标签中，请写出：
1. 数据解读：仿真结果给出的数值或状态意味着什么？
2. 物理关联：这个结果对应混沌理论中的哪个概念（如倍周期分岔、奇怪吸引子）？
3. 结论综合。

格式示例：
<thinking>
检测到 r=3.5 时为 4 周期振荡... 根据费根鲍姆常数理论...
</thinking>
<answer>
这里是最终给用户的回答...
</answer>
"""

    user_prompt = f"""
    【用户问题】: {question}
    【背景知识/仿真结果】: 
    {context}
    
    请生成 CoT 回答：
    """

    try:
        response = teacher_client.chat.completions.create(
            model="deepseek-chat", # V3 模型能力很强
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 # 稍微有点创造力，让思维链更丰富
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   ⚠️ DeepSeek 生成失败: {e}")
        return None

def run_factory():
    print(f"🏭 启动 CoT 蒸馏工厂 (Teacher-Student 模式)")
    
    if not os.path.exists(QUESTION_FILE):
        print(f"❌ 请先创建 {QUESTION_FILE}")
        return

    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    saved_count = 0

    for i, q in enumerate(questions):
        print(f"\n--------------------------------------------------")
        print(f"Processing [{i+1}/{len(questions)}]: {q}")
        
        # 1. 本地 Agent 负责脏活累活 (路由 + 检索/计算)
        # 我们需要模拟真实环境，让本地 Agent 去拿 Context
        category = get_route_category(q, router_chain)
        context_record = "" 

        try:
            # ➤ RAG 模式
            if category == "RAG":
                retriever = get_retriever_tool()
                rag_result = retriever.func(q)
                
                # 质量控制：如果本地都查不到东西，DeepSeek 再强也编不出来好答案
                if "资料不足" in rag_result or len(rag_result) < 20:
                    print("   ⏭️  [跳过] 本地检索失败，缺乏上下文")
                    continue
                
                context_record = rag_result

            # ➤ 数学模式
            elif category == "COMPUTE":
                if "r=" in q or "r =" in q:
                    match = re.search(r"r\s*[=:]\s*(\d+\.?\d*)", q)
                    r_val = float(match.group(1)) if match else 3.5
                    result_text, _ = tools.simulate_logistic_map(r_val)
                elif "lorenz" in q.lower():
                    result_text, _ = tools.simulate_lorenz()
                else:
                    result_text = "无法识别计算参数"
                
                context_record = f"Python Tool Output: {result_text}"

            else:
                print("   ⏭️  [跳过] 闲聊问题")
                continue

            # 2. 核心改变：让 DeepSeek 老师写标准答案 (包含 <thinking>)
            print(f"   🧠 正在请求 DeepSeek 生成 CoT 思维链...")
            cot_response = generate_golden_cot(q, context_record, category)
            
            if cot_response and "<thinking>" in cot_response:
                # 3. 保存数据
                # 注意：这里我们构造的数据对是：
                # Instruction: 用户问题
                # Input: 本地检索到的 Context (模拟真实环境)
                # Output: DeepSeek 写的 CoT 答案 (作为学习目标)
                entry = {
                    "instruction": q,
                    "input": context_record,
                    "output": cot_response
                }
                
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                print(f"   ✅ [录入] 成功生成 CoT 数据")
                saved_count += 1
            else:
                print(f"   ❌ [失败] 生成格式不符合要求")

        except Exception as e:
            print(f"   ⚠️ 处理出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 蒸馏完成。入库 {saved_count} 条 CoT 数据。")
    print(f"📂 训练数据: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_factory()