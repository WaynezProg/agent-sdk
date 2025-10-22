# 02_document_loading.py - 文件載入與處理
import os
from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.readers.web import BeautifulSoupWebReader

# 載入環境變數
load_dotenv()

def create_sample_documents():
    """創建範例文件用於教學"""
    print("📝 創建範例文件...")
    
    # 創建範例文字文件
    sample_texts = [
        {
            "filename": "sample_ai.txt",
            "content": """
人工智慧（AI）是電腦科學的一個分支，致力於創建能夠執行通常需要人類智慧的任務的系統。
這些任務包括學習、推理、問題解決、感知和語言理解。

機器學習是 AI 的一個子領域，它使電腦能夠在沒有明確編程的情況下學習和改進。
深度學習是機器學習的一個子集，使用人工神經網路來模擬人腦的工作方式。

AI 的應用領域包括：
- 自然語言處理
- 電腦視覺
- 語音識別
- 推薦系統
- 自動駕駛
            """
        },
        {
            "filename": "sample_tech.txt", 
            "content": """
雲端計算是透過網際網路提供運算服務的技術，包括伺服器、儲存、資料庫、網路、軟體等。

雲端服務的主要類型：
1. 基礎設施即服務 (IaaS)
2. 平台即服務 (PaaS)  
3. 軟體即服務 (SaaS)

主要雲端服務提供商：
- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)

雲端運算的優勢：
- 成本效益
- 可擴展性
- 靈活性
- 可靠性
            """
        }
    ]
    
    # 創建文件資料夾
    docs_dir = "sample_documents"
    os.makedirs(docs_dir, exist_ok=True)
    
    # 寫入範例文件
    for doc in sample_texts:
        filepath = os.path.join(docs_dir, doc["filename"])
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
        print(f"✅ 創建文件: {filepath}")
    
    return docs_dir

def load_documents_from_directory(directory_path):
    """從資料夾載入所有文件"""
    print(f"\n📁 從資料夾載入文件: {directory_path}")
    
    # 使用 SimpleDirectoryReader 載入所有文件
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,  # 遞歸搜尋子資料夾
        required_exts=[".txt", ".md", ".py"]  # 指定檔案類型
    )
    
    documents = reader.load_data()
    
    print(f"✅ 載入了 {len(documents)} 個文件")
    for i, doc in enumerate(documents):
        print(f"   文件 {i+1}: {doc.metadata.get('file_path', 'Unknown')}")
        print(f"   內容長度: {len(doc.text)} 字元")
        print(f"   前 100 字: {doc.text[:100]}...")
        print()
    
    return documents

def load_single_document():
    """載入單一文件並自定義處理"""
    print("\n📄 載入單一文件...")
    
    # 創建自定義文件
    custom_doc = Document(
        text="這是一個自定義文件，用於示範如何手動創建 Document 物件。",
        metadata={
            "source": "manual_creation",
            "author": "教學範例",
            "date": "2024-01-01"
        }
    )
    
    print(f"✅ 自定義文件創建完成")
    print(f"   內容: {custom_doc.text}")
    print(f"   元數據: {custom_doc.metadata}")
    
    return custom_doc

def demonstrate_document_processing():
    """示範文件處理功能"""
    print("\n🔧 文件處理功能示範...")
    
    # 創建範例文件
    docs_dir = create_sample_documents()
    
    # 載入文件
    documents = load_documents_from_directory(docs_dir)
    
    # 載入自定義文件
    custom_doc = load_single_document()
    
    # 合併所有文件
    all_documents = documents + [custom_doc]
    
    print(f"\n📊 文件統計:")
    print(f"   總文件數: {len(all_documents)}")
    print(f"   總字元數: {sum(len(doc.text) for doc in all_documents)}")
    print(f"   平均文件長度: {sum(len(doc.text) for doc in all_documents) / len(all_documents):.0f} 字元")
    
    # 示範文件分割
    print(f"\n✂️ 文件分割示範:")
    for i, doc in enumerate(all_documents[:2]):  # 只示範前兩個文件
        print(f"   文件 {i+1} 原始長度: {len(doc.text)} 字元")
        
        # 簡單分割（實際使用時會用更複雜的分割策略）
        chunks = [doc.text[i:i+200] for i in range(0, len(doc.text), 200)]
        print(f"   分割後塊數: {len(chunks)}")
        print(f"   第一塊內容: {chunks[0][:50]}...")
        print()
    
    return all_documents

def explain_document_loaders():
    """解釋不同文件載入器的使用"""
    print("\n📚 文件載入器說明:")
    print("""
    LlamaIndex 支援多種文件載入器：
    
    1. 📁 SimpleDirectoryReader
       - 從資料夾載入多個文件
       - 支援多種檔案格式
       - 自動處理檔案編碼
    
    2. 📄 PDFReader
       - 專門處理 PDF 文件
       - 提取文字和元數據
       - 支援多頁文件
    
    3. 📝 DocxReader  
       - 處理 Microsoft Word 文件
       - 保留格式資訊
       - 支援表格和圖片
    
    4. 🌐 BeautifulSoupWebReader
       - 從網頁載入內容
       - 自動清理 HTML 標籤
       - 支援多個 URL
    
    5. 📊 其他載入器
       - NotionReader
       - SlackReader
       - DiscordReader
       - 等等...
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習文件載入...")
        
        # 解釋文件載入器
        explain_document_loaders()
        
        # 示範文件處理
        documents = demonstrate_document_processing()
        
        print("\n🎉 文件載入學習完成！")
        print("下一步：學習如何建立向量索引")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()