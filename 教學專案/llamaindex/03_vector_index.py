# 03_vector_index.py - 向量索引建立與管理
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# 載入環境變數
load_dotenv()

def create_basic_vector_index():
    """建立基本的向量索引"""
    print("🔨 建立基本向量索引...")
    
    # 載入文件（使用前一個範例的文件）
    documents_dir = "sample_documents"
    if not os.path.exists(documents_dir):
        print("❌ 找不到範例文件，請先執行 02_document_loading.py")
        return None
    
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data()
    
    print(f"📄 載入了 {len(documents)} 個文件")
    
    # 建立向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    print("✅ 向量索引建立完成！")
    print(f"   索引類型: {type(index).__name__}")
    print(f"   文件節點數: {index.docstore.docs}")
    
    return index

def create_persistent_index():
    """建立持久化的向量索引"""
    print("\n💾 建立持久化向量索引...")
    
    # 檢查是否已有索引
    persist_dir = "./storage"
    if os.path.exists(persist_dir):
        print("📂 載入現有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("✅ 索引載入完成！")
    else:
        print("🆕 建立新索引...")
        
        # 載入文件
        documents_dir = "sample_documents"
        if not os.path.exists(documents_dir):
            print("❌ 找不到範例文件，請先執行 02_document_loading.py")
            return None
            
        reader = SimpleDirectoryReader(input_dir=documents_dir)
        documents = reader.load_data()
        
        # 建立索引並持久化
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=StorageContext.from_defaults(persist_dir=persist_dir)
        )
        
        print("✅ 持久化索引建立完成！")
    
    return index

def create_chroma_index():
    """使用 ChromaDB 建立向量索引"""
    print("\n🌈 使用 ChromaDB 建立向量索引...")
    
    try:
        # 初始化 ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("llamaindex_tutorial")
        
        # 建立 ChromaDB 向量儲存
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 載入文件
        documents_dir = "sample_documents"
        if not os.path.exists(documents_dir):
            print("❌ 找不到範例文件，請先執行 02_document_loading.py")
            return None
            
        reader = SimpleDirectoryReader(input_dir=documents_dir)
        documents = reader.load_data()
        
        # 建立索引
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        print("✅ ChromaDB 索引建立完成！")
        print(f"   集合名稱: {chroma_collection.name}")
        print(f"   文件數量: {chroma_collection.count()}")
        
        return index
        
    except Exception as e:
        print(f"❌ ChromaDB 設定失敗: {e}")
        print("請確保已安裝 chromadb: pip install chromadb")
        return None

def demonstrate_index_operations():
    """示範索引操作"""
    print("\n🔧 索引操作示範...")
    
    # 建立基本索引
    index = create_basic_vector_index()
    if not index:
        return
    
    # 建立查詢引擎
    query_engine = index.as_query_engine()
    
    # 測試查詢
    test_queries = [
        "什麼是人工智慧？",
        "雲端運算有哪些優勢？",
        "機器學習和深度學習的關係是什麼？"
    ]
    
    print("🔍 測試查詢:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n查詢 {i}: {query}")
        try:
            response = query_engine.query(query)
            print(f"回答: {response.response}")
            print(f"來源文件: {len(response.source_nodes)} 個")
        except Exception as e:
            print(f"❌ 查詢失敗: {e}")

def explain_index_types():
    """解釋不同類型的索引"""
    print("\n📚 索引類型說明:")
    print("""
    LlamaIndex 支援多種索引類型：
    
    1. 🔍 VectorStoreIndex
       - 最常用的索引類型
       - 基於向量相似性搜尋
       - 適合大部分 RAG 應用
    
    2. 📊 SummaryIndex
       - 基於摘要的索引
       - 適合需要整體概覽的查詢
       - 查詢速度較快
    
    3. 🌳 TreeIndex
       - 樹狀結構索引
       - 適合層次性查詢
       - 支援從根到葉的查詢路徑
    
    4. 🔗 KeywordTableIndex
       - 基於關鍵字的索引
       - 適合精確匹配查詢
       - 查詢速度極快
    
    5. 🎯 CompositeIndex
       - 複合索引
       - 結合多種索引類型
       - 提供最佳的查詢效果
    """)

def explain_vector_stores():
    """解釋向量儲存選項"""
    print("\n💾 向量儲存選項:")
    print("""
    1. 🗂️ SimpleVectorStore (預設)
       - 記憶體儲存
       - 適合開發和測試
       - 重啟後資料會消失
    
    2. 💿 FaissVectorStore
       - 基於 Facebook AI Similarity Search
       - 高效能向量搜尋
       - 支援大規模資料集
    
    3. 🌈 ChromaVectorStore
       - 基於 ChromaDB
       - 持久化儲存
       - 支援多種距離度量
    
    4. 🚀 PineconeVectorStore
       - 雲端向量資料庫
       - 高可用性和擴展性
       - 適合生產環境
    
    5. 🔍 WeaviateVectorStore
       - 基於 Weaviate
       - 支援多模態搜尋
       - 豐富的查詢功能
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習向量索引...")
        
        # 解釋索引類型
        explain_index_types()
        
        # 解釋向量儲存
        explain_vector_stores()
        
        # 示範索引操作
        demonstrate_index_operations()
        
        # 建立持久化索引
        persistent_index = create_persistent_index()
        
        # 嘗試建立 ChromaDB 索引
        chroma_index = create_chroma_index()
        
        print("\n🎉 向量索引學習完成！")
        print("下一步：學習查詢與檢索策略")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()