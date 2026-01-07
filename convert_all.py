import os
import glob
import pymupdf4llm

## 这是如果没有API接口去清洗数据的话建议使用这个脚本，但是坏处是转换的md肯定没有deepseek那么完美
# 配置路径
SOURCE_DIR = "./data_pdf"          # PDF 所在的文件夹
OUTPUT_DIR = "./data"    # 清洗后的数据存放处

# 如果输出目录不存在，创建它
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 获取所有 PDF
pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
print(f"🧹 准备清洗 {len(pdf_files)} 个 PDF 文件...")

success_count = 0

for i, pdf_path in enumerate(pdf_files):
    file_name = os.path.basename(pdf_path)
    # 把后缀从 .pdf 改为 .md
    md_name = os.path.splitext(file_name)[0] + ".md"
    save_path = os.path.join(OUTPUT_DIR, md_name)
    
    print(f"[{i+1}/{len(pdf_files)}] 正在转换: {file_name} ...")
    
    try:
        # 核心转换：把 PDF 转为 Markdown (支持提取表格和部分公式)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # 写入新文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ 转换失败 {file_name}: {e}")

print(f"\n✨ 清洗完成！成功转换 {success_count} 个文件。")
print(f"📂 结果保存在: {os.path.abspath(OUTPUT_DIR)}")
print("👉 下一步建议：\n1. 检查 data_clean 里的文件内容是否正常。\n2. 将它们移动到 data 目录(覆盖/清空原PDF)。\n3. 运行 build_db.py 重建库。")