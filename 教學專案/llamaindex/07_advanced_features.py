# 07_advanced_features.py - LlamaIndex 進階功能
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# 載入環境變數
load_dotenv()

def setup_advanced_system():
    """設定進階系統"""
    print("🔧 設定進階 LlamaIndex 系統...")
    
    # 設定嵌入模型
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=10
    )
    Settings.embed_model = embed_model
    
    # 載入文件
    documents_dir = "sample_documents"
    if not os.path.exists(documents_dir):
        print("❌ 找不到範例文件，請先執行 02_document_loading.py")
        return None
    
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data()
    
    print("✅ 進階系統設定完成！")
    return documents

def demonstrate_custom_node_parser(documents):
    """示範自定義節點解析器"""
    print("\n✂️ 自定義節點解析器...")
    
    # 創建自定義分割器
    custom_splitter = SentenceSplitter(
        chunk_size=512,      # 較小的塊大小
        chunk_overlap=100,   # 增加重疊
        separator="。",      # 使用句號分割
        paragraph_separator="\n\n"  # 段落分隔符
    )
    
    # 建立索引
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[custom_splitter]
    )
    
    print("✅ 自定義節點解析器設定完成！")
    print(f"   塊大小: {custom_splitter.chunk_size}")
    print(f"   重疊大小: {custom_splitter.chunk_overlap}")
    print(f"   分隔符: {custom_splitter.separator}")
    
    return index

def demonstrate_advanced_retrievers(index):
    """示範進階檢索器"""
    print("\n🎯 進階檢索器...")
    
    # 建立向量檢索器
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    
    # 測試檢索
    query = "人工智慧的應用"
    nodes = vector_retriever.retrieve(query)
    
    print(f"查詢: {query}")
    print(f"檢索到 {len(nodes)} 個節點")
    
    for i, node in enumerate(nodes[:3], 1):
        score = node.score if hasattr(node, 'score') else 'N/A'
        print(f"節點 {i}: 分數 = {score}")
        print(f"內容: {node.text[:100]}...")

def demonstrate_postprocessors(index):
    """示範後處理器"""
    print("\n🔧 後處理器...")
    
    # 建立檢索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )
    
    # 相似度過濾器
    similarity_filter = SimilarityPostprocessor(
        similarity_cutoff=0.7
    )
    
    # 關鍵字過濾器
    keyword_filter = KeywordNodePostprocessor(
        required_keywords=["人工智慧", "機器學習", "深度學習"],
        exclude_keywords=["刪除", "移除"]
    )
    
    # 測試查詢
    query = "機器學習的應用"
    nodes = retriever.retrieve(query)
    
    print(f"原始檢索結果: {len(nodes)} 個節點")
    
    # 應用相似度過濾
    filtered_nodes = similarity_filter.postprocess_nodes(nodes, query_bundle=None)
    print(f"相似度過濾後: {len(filtered_nodes)} 個節點")
    
    # 應用關鍵字過濾
    keyword_filtered_nodes = keyword_filter.postprocess_nodes(filtered_nodes, query_bundle=None)
    print(f"關鍵字過濾後: {len(keyword_filtered_nodes)} 個節點")

def demonstrate_chroma_integration(documents):
    """示範 ChromaDB 整合"""
    print("\n🌈 ChromaDB 整合...")
    
    try:
        # 初始化 ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_advanced")
        collection = chroma_client.get_or_create_collection(
            name="advanced_tutorial",
            metadata={"description": "進階教學範例"}
        )
        
        # 建立向量儲存
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 建立索引
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        print("✅ ChromaDB 整合完成！")
        print(f"   集合名稱: {collection.name}")
        print(f"   文件數量: {collection.count()}")
        
        # 測試查詢
        query_engine = index.as_query_engine()
        response = query_engine.query("什麼是雲端運算？")
        print(f"查詢結果: {response.response[:100]}...")
        
        return index
        
    except Exception as e:
        print(f"❌ ChromaDB 整合失敗: {e}")
        return None

def demonstrate_hybrid_search(index):
    """示範混合搜尋"""
    print("\n🔀 混合搜尋...")
    
    # 建立查詢引擎
    query_engine = index.as_query_engine(
        response_mode="compact",
        similarity_top_k=3
    )
    
    # 測試不同類型的查詢
    queries = [
        "什麼是人工智慧？",  # 概念性查詢
        "AI 應用領域",      # 關鍵字查詢
        "機器學習 vs 深度學習"  # 比較查詢
    ]
    
    for query in queries:
        print(f"\n查詢: {query}")
        response = query_engine.query(query)
        print(f"回答: {response.response[:150]}...")

def demonstrate_custom_embeddings():
    """示範自定義嵌入"""
    print("\n🧠 自定義嵌入...")
    
    # 創建自定義嵌入模型
    custom_embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=5,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # 測試嵌入
    texts = [
        "人工智慧是電腦科學的一個分支",
        "機器學習是 AI 的重要組成部分",
        "深度學習使用神經網路"
    ]
    
    print("測試文字嵌入:")
    for i, text in enumerate(texts, 1):
        embedding = custom_embed_model.get_text_embedding(text)
        print(f"文字 {i}: {text}")
        print(f"嵌入維度: {len(embedding)}")
        print(f"前 5 個值: {embedding[:5]}")

def demonstrate_metadata_filtering(index):
    """示範元數據過濾"""
    print("\n🏷️ 元數據過濾...")
    
    # 建立帶有元數據過濾的查詢引擎
    query_engine = index.as_query_engine(
        filters={"file_path": "sample_ai.txt"},
        response_mode="compact"
    )
    
    query = "人工智慧的定義"
    print(f"查詢: {query}")
    print("過濾條件: 只查詢 sample_ai.txt 文件")
    
    response = query_engine.query(query)
    print(f"回答: {response.response}")

def demonstrate_streaming_response(index):
    """示範串流回應"""
    print("\n🌊 串流回應...")
    
    # 建立串流查詢引擎
    query_engine = index.as_query_engine(
        streaming=True,
        response_mode="compact"
    )
    
    query = "請詳細說明人工智慧的發展歷史"
    print(f"查詢: {query}")
    print("串流回應:")
    
    # 執行串流查詢
    response = query_engine.query(query)
    
    # 模擬串流輸出
    if hasattr(response, 'response_gen'):
        for chunk in response.response_gen:
            print(chunk, end='', flush=True)
    else:
        print(response.response)

def explain_advanced_features():
    """解釋進階功能"""
    print("\n📚 進階功能說明:")
    print("""
    1. ✂️ 自定義節點解析器
       - 控制文件分割策略
       - 優化塊大小和重疊
       - 支援多種分割方式
    
    2. 🎯 進階檢索器
       - 多種檢索策略
       - 自定義相似度計算
       - 支援複雜查詢
    
    3. 🔧 後處理器
       - 相似度過濾
       - 關鍵字過濾
       - 自定義過濾邏輯
    
    4. 💾 向量資料庫整合
       - ChromaDB
       - Pinecone
       - Weaviate
       - 自定義向量儲存
    
    5. 🔀 混合搜尋
       - 結合多種檢索方式
       - 語義 + 關鍵字搜尋
       - 提高檢索準確性
    
    6. 🧠 自定義嵌入
       - 多種嵌入模型
       - 批次處理優化
       - 自定義嵌入策略
    
    7. 🏷️ 元數據管理
       - 豐富的元數據支援
       - 靈活的過濾條件
       - 結構化資訊管理
    
    8. 🌊 串流處理
       - 即時回應生成
       - 改善使用者體驗
       - 支援長文本生成
    """)

def demonstrate_performance_optimization(index):
    """示範效能優化"""
    print("\n⚡ 效能優化...")
    
    import time
    
    # 測試不同配置的效能
    configurations = [
        {"similarity_top_k": 3, "response_mode": "compact"},
        {"similarity_top_k": 5, "response_mode": "compact"},
        {"similarity_top_k": 3, "response_mode": "tree_summarize"}
    ]
    
    query = "什麼是人工智慧？"
    
    for i, config in enumerate(configurations, 1):
        print(f"\n配置 {i}: {config}")
        
        query_engine = index.as_query_engine(**config)
        
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        
        print(f"查詢時間: {end_time - start_time:.2f} 秒")
        print(f"回應長度: {len(response.response)} 字元")

if __name__ == "__main__":
    try:
        print("🚀 開始學習 LlamaIndex 進階功能...")
        
        # 解釋進階功能
        explain_advanced_features()
        
        # 設定進階系統
        documents = setup_advanced_system()
        if not documents:
            print("❌ 無法設定進階系統，請檢查文件是否存在")
            exit(1)
        
        # 自定義節點解析器
        index = demonstrate_custom_node_parser(documents)
        
        # 進階檢索器
        demonstrate_advanced_retrievers(index)
        
        # 後處理器
        demonstrate_postprocessors(index)
        
        # ChromaDB 整合
        chroma_index = demonstrate_chroma_integration(documents)
        
        # 混合搜尋
        demonstrate_hybrid_search(index)
        
        # 自定義嵌入
        demonstrate_custom_embeddings()
        
        # 元數據過濾
        demonstrate_metadata_filtering(index)
        
        # 串流回應
        demonstrate_streaming_response(index)
        
        # 效能優化
        demonstrate_performance_optimization(index)
        
        print("\n🎉 LlamaIndex 進階功能學習完成！")
        print("下一步：學習生產環境部署")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()