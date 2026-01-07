import os
import glob
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader

# 加载环境
load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ================= 配置区域 =================
DATA_DIR = "./data"                  # 清洗后的 Markdown 文件夹
OUTPUT_FILE = "chaos_finetune_v2.jsonl" # 输出文件
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TARGET_COUNT = 450                   # 你想要多少条 RAG 数据
# ===========================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def generate_qa_pair(chunk_text):
    """
    让 DeepSeek 基于【给定的文本片段】同时生成：问题 + CoT答案
    """
    system_prompt = """你是一个负责构建混沌理论题库的数据专家。
    我给你一段【专业文献片段】，请你基于这段内容构建一个训练数据对。
    
    【任务步骤】
    1. **提问 (Question)**: 假设你是用户，针对这段文字的核心知识点，提出了一个问题。
       - 问题要自然、像真人的口吻。
       - 问题必须能完全通过这段文字找到答案。
       
    2. **思维链 (CoT)**: 扮演专家，根据这段文字进行逻辑分析。
       - 必须包含 <thinking> 标签，解释如何从文中找到线索。
       
    3. **回答 (Answer)**: 给出最终答案。
    
    【输出格式 - 严格 JSON】
    {
        "instruction": "生成的问题...",
        "output": "<thinking>分析过程...</thinking><answer>最终回答...</answer>"
    }
    """
    
    user_prompt = f"【文献片段】:\n{chunk_text[:2000]}\n\n请生成 JSON:"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # 强制 JSON 模式，DeepSeek 支持
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ 生成失败: {e}")
        return None

def run_reverse_factory():
    print(f"🏭 启动【逆向数据工厂】(Document -> QA)...")
    
    # 1. 加载所有 Markdown 文件
    files = glob.glob(os.path.join(DATA_DIR, "*.md")) + glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print("❌ 没有找到数据文件，请检查 ./data 目录")
        return

    # 2. 读取所有文本并简单切块
    all_chunks = []
    print("📖 正在读取文档...")
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # 简单按双换行切分段落，每段作为一个潜在的出题素材
                # 过滤掉太短的段落
                chunks = [c for c in text.split('\n\n') if len(c) > 200]
                all_chunks.extend(chunks)
        except:
            pass
            
    print(f"📊 共提取出 {len(all_chunks)} 个有效段落素材。")
    
    # 3. 循环生成
    saved_count = 0
    # 随机打乱，避免只盯着一本书问
    random.shuffle(all_chunks) 
    
    for i, chunk in enumerate(all_chunks):
        if saved_count >= TARGET_COUNT:
            break
            
        print(f"\n--------------------------------------------------")
        print(f"Processing Chunk [{i+1}] (Length: {len(chunk)})")
        
        # 调用 DeepSeek 生成
        qa_pair = generate_qa_pair(chunk)
        
        if qa_pair:
            # 构造训练数据格式
            entry = {
                "instruction": qa_pair["instruction"],
                # 注意：这里 Input 直接放入原文片段！
                # 这样训练时，模型学到的是：当 RAG 检索到这段话(Input)时，我该如何回答(Output)
                "input": chunk, 
                "output": qa_pair["output"]
            }
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            print(f"✅ [生成成功]\n❓ 问: {qa_pair['instruction']}\n💡 答: {qa_pair['output'][:50]}...")
            saved_count += 1
        else:
            print("❌ 生成格式错误或失败")

    # ==========================================
    # 补充：必须加入数学计算题 (Template-Based)
    # 因为文档里没有 python 计算逻辑，这部分必须手动加
    # ==========================================
    print(f"\n➕ 正在补充数学计算题 (Template-Based)...")
    math_count = 0
    # import random
    
    # 生成 50 道不同参数的计算题
    for _ in range(50):
        r = round(random.uniform(3.0, 4.0), 2)
        # 构造问题
        q = f"计算r={r}时的Logistic映射状态"
        
        # 构造 Input (模拟 Python 工具的输出)
        # 这里为了简化，我们假设模型应该学会识别工具输出
        # 但在微调数据里，input 应该是工具的返回结果。
        # 这里我们可以用你的 tools.py 真的算一下
        from tools import simulate_logistic_map
        tool_output, _ = simulate_logistic_map(r)
        
        # 构造 DeepSeek 的 CoT 回答 (也可以用模板写死，省钱)
        cot_output = f"<thinking>检测到 RAG/Tools 返回了计算结果。参数 r={r}。观察最后20次迭代值，判断系统处于周期或混沌状态。</thinking><answer>{tool_output}</answer>"
        
        entry = {
            "instruction": q,
            "input": f"Python Simulation Result: {tool_output}", # 模拟工具输出
            "output": cot_output
        }
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        math_count += 1
        
    print(f"✅ 已补充 {math_count} 条数学计算题。")
    print(f"\n🎉 V2 工厂停工。总计产出 {saved_count + math_count} 条高质量数据。")
    print(f"📂 文件位置: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_reverse_factory()