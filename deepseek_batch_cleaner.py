mport os
import glob
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境
load_dotenv()
# 配置 DeepSeek
client = OpenAI(
    api_key="",  # 建议从环境变量读取
    base_url="https://api.deepseek.com"
)

# ================= 配置区域 =================
SOURCE_FOLDER = "./data_pdf"      # 原始 PDF 文件夹 (请确保这里有 PDF)
OUTPUT_FOLDER = "./data"          # 输出 Markdown 的文件夹
PAGES_PER_BATCH = 5               # 每次给 DeepSeek 处理几页 (太大会截断，太小费钱)
# ===========================================

def clean_text_with_deepseek(text_chunk, is_first_batch):
    """
    调用 DeepSeek 将乱码/生硬的 PDF 文本重构为 Markdown
    """
    if not text_chunk.strip():
        return ""

    # 动态调整 Prompt
    # 如果不是第一批次，特意叮嘱不要输出文章标题和目录
    constraint = ""
    if not is_first_batch:
        constraint = "注意：这是文档的中间部分，请直接接着上一部分的内容转换，**不要**重复输出文章标题、作者或目录。保持正文的连续性。"

    system_prompt = f"""你是一个专业的学术数据清洗专家。你的任务是将用户提供的、可能包含乱码的PDF原始文本，重写为结构完美、排版清晰的 Markdown 格式。
    
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
    8. {constraint}
    9.**不要随意转换、翻译原文语言**，保持PDF内原有语言。
    
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk}
            ],
            temperature=0.1, # 清洗数据要严谨
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ API 调用失败: {e}")
        return text_chunk # 失败时保留原文，防止丢数据

def convert_single_pdf(pdf_path):
    filename = os.path.basename(pdf_path).replace('.pdf', '.md')
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    
    print(f"\n📄 正在处理: {os.path.basename(pdf_path)}")
    
    full_text_buffer = ""
    current_batch_text = ""
    page_count = 0
    
    # 1. 使用 pdfplumber 打开 (比 PyPDF2 更好处理双栏)
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # 打开输出文件 (使用 'w' 模式清空旧内容)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            
            # 2. 循环遍历每一页
            for i, page in enumerate(tqdm(pdf.pages, desc="Processing Pages")):
                # 提取文本 (可以使用 x_tolerance 优化排版，这里用默认)
                text = page.extract_text()
                if text:
                    current_batch_text += text + "\n\n"
                    page_count += 1
                
                # 3. 达到批次大小，或者最后一页，发送给 DeepSeek
                if page_count >= PAGES_PER_BATCH or i == total_pages - 1:
                    if current_batch_text.strip():
                        # 判断是否为第一批 (决定是否保留标题/目录)
                        is_first = (i < PAGES_PER_BATCH)
                        
                        # 调用 API
                        cleaned_md = clean_text_with_deepseek(current_batch_text, is_first)
                        
                        # 4. 实时写入文件 (防止程序崩溃全白跑)
                        f_out.write(cleaned_md + "\n\n")
                        f_out.flush() # 强制刷入硬盘
                        
                        # 清空缓冲区
                        current_batch_text = ""
                        page_count = 0
    
    print(f"✅ 完成转换 -> {output_path}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    pdf_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {SOURCE_FOLDER} 下没找到 PDF 文件！")
        return
        
    print(f"🚀 启动增量清洗引擎，共 {len(pdf_files)} 个文件...")
    
    for pdf_file in pdf_files:
        try:
            convert_single_pdf(pdf_file)
        except Exception as e:
            print(f"❌ 处理文件失败 {pdf_file}: {e}")

if __name__ == "__main__":
    main()
