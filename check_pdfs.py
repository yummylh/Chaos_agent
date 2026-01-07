import os
from langchain_community.document_loaders import PDFPlumberLoader

def check_pdfs():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"❌ 错误：找不到 {data_dir} 文件夹")
        return

    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    print(f"🔍 开始检查 {len(files)} 个 PDF 文件...\n")

    bad_files = []

    for i, filename in enumerate(files):
        file_path = os.path.join(data_dir, filename)
        print(f"[{i+1}/{len(files)}] 正在检查: {filename} ... ", end="", flush=True)
        
        try:
            # 尝试加载每一页
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            # 简单的内容检查，确保读到了字
            if len(docs) > 0 and len(docs[0].page_content) > 0:
                print("✅ 通过")
            else:
                print("⚠️ 警告 (内容为空)")
        except Exception as e:
            print(f"❌ 失败！")
            print(f"   错误详情: {str(e)}")
            bad_files.append(filename)

    print("\n" + "="*30)
    if bad_files:
        print(f"🚫 发现 {len(bad_files)} 个损坏或无法读取的文件：")
        for f in bad_files:
            print(f" - {f}")
        print("\n💡 建议：请将上述文件从 data 文件夹中移除，然后重新运行 app.py")
    else:
        print("🎉 所有 PDF 检查通过！如果 app.py 依然报错，可能是内存问题。")

if __name__ == "__main__":
    check_pdfs()