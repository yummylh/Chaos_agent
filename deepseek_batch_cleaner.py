import os
import glob
import fitz  # PyMuPDF
from openai import OpenAI
import concurrent.futures
import time

# ================= 配置区域 =================
API_KEY = ""  # 记得填入 Key
BASE_URL = "https://api.deepseek.com"

SOURCE_DIR = "./bad_data"          
OUTPUT_DIR = "./data"       

# 如果并发 5 个报错，可以降为 3
MAX_WORKERS = 5 
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"❌ PDF 读取失败 {pdf_path}: {e}")
        return None

def process_single_file(pdf_path):
    file_name = os.path.basename(pdf_path)
    save_name = os.path.splitext(file_name)[0] + ".md"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    if os.path.exists(save_path):
        return f"⏭️ [跳过] {file_name} 已存在"

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return f"❌ [失败] {file_name} 内容为空"

    if len(raw_text) > 50000:
        raw_text = raw_text[:50000] + "\n\n(截断...)"

    # === 针对 V3 和 图表 的专门指令 ===
    system_prompt = """你是一个专业的学术数据清洗专家。你的任务是将用户提供的、可能包含乱码的PDF原始文本，重写为结构完美、排版清晰的 Markdown 格式。
    
    【核心指令】:
    1. **模型确认**：你现在使用的是 DeepSeek-V3 引擎，请发挥你最强的逻辑修复能力。
    2. **图表处理 (重要)**：
       - 由于你看不到图片，如果遇到图表区域解析出的一堆无意义乱码/数字，请**直接丢弃**。
       - **但是**：必须保留图表的标题（如 "Fig. 1: Bifurcation diagram..."），并将其格式化为加粗文本，例如：**图1：分岔图说明**。
    3. **修复内容**：识别并修复全角字符乱码、断裂的单词（'r e s e a r c h' -> 'research'）。
    4. **公式标准化**：将所有数学公式转换为标准 LaTeX 格式（行内 $...$, 独立块 $$...$$）。
    5. **结构保留**：准确保留 # 标题层级。
    6. **去噪**：删除页眉、页脚、页码、参考文献。
    7. **纯净输出**：直接输出 Markdown，不要任何废话。
    """

    try:
        start_time = time.time()
        # 这里指定 model="deepseek-chat" 就是调用 V3
        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"文件名：{file_name}\n\n原始文本：\n{raw_text}"}
            ],
            stream=False,
            temperature=0.1
        )
        
        cleaned_content = response.choices[0].message.content
        
        # 验证模型版本 (通过 response.model 属性)
        used_model = response.model # 通常返回 'deepseek-chat'

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        
        elapsed = time.time() - start_time
        return f"✅ [成功] {file_name} (模型: {used_model}, 耗时: {elapsed:.1f}s)"

    except Exception as e:
        return f"❌ [API错误] {file_name}: {e}"

def main():
    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
    print(f"🚀 启动 DeepSeek-V3 清洗任务 | 目标文件: {len(pdf_files)}")
    print("--------------------------------------------------")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_file, pdf): pdf for pdf in pdf_files}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(pdf_files)}] {result}")

    print("--------------------------------------------------")
    print(f"🎉 全部完成！请检查 {OUTPUT_DIR} 目录。")

if __name__ == "__main__":
    main()