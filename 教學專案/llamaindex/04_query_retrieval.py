# 04_query_retrieval.py - 查詢與檢索策略
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    QueryBundle,
    ResponseMode
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 載入環境變數
load_dotenv()

def setup_index():
    """設定索引用於查詢示範"""
    print("🔧 設定索引...")
    
    # 載入文件
    documents_dir = "sample_documents"
    if not os.path.exists(documents_dir):
        print("❌ 找不到範例文件，請先執行 02_document_loading.py")
        return None
    
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data()
    
    # 建立索引
    index = VectorStoreIndex.from_documents(documents)
    print("✅ 索引設定完成！")
    
    return index

def basic_query_demo(index):
    """基本查詢示範"""
    print("\n🔍 基本查詢示範...")
    
    # 建立查詢引擎
    query_engine = index.as_query_engine()
    
    # 測試查詢
    queries = [
        "什麼是人工智慧？",
        "雲端運算的主要優勢有哪些？",
        "機器學習和深度學習有什麼不同？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n查詢 {i}: {query}")
        response = query_engine.query(query)
        print(f"回答: {response.response}")
        print(f"來源節點數: {len(response.source_nodes)}")
        
        # 顯示來源資訊
        for j, node in enumerate(response.source_nodes[:2]):  # 只顯示前兩個
            print(f"  來源 {j+1}: {node.text[:100]}...")

def advanced_retrieval_demo(index):
    """進階檢索示範"""
    print("\n🎯 進階檢索示範...")
    
    # 建立檢索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5  # 檢索前 5 個最相關的節點
    )
    
    # 建立查詢引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode=ResponseMode.COMPACT  # 使用緊湊模式
    )
    
    query = "請詳細說明人工智慧的應用領域"
    print(f"查詢: {query}")
    
    # 執行查詢
    response = query_engine.query(query)
    print(f"回答: {response.response}")
    
    # 分析檢索結果
    print(f"\n📊 檢索分析:")
    print(f"   檢索到的節點數: {len(response.source_nodes)}")
    
    for i, node in enumerate(response.source_nodes):
        score = node.score if hasattr(node, 'score') else 'N/A'
        print(f"   節點 {i+1}: 相似度分數 = {score}")
        print(f"           內容: {node.text[:80]}...")

def similarity_filtering_demo(index):
    """相似度過濾示範"""
    print("\n🎚️ 相似度過濾示範...")
    
    # 建立檢索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10  # 先檢索 10 個
    )
    
    # 建立相似度過濾器
    similarity_postprocessor = SimilarityPostprocessor(
        similarity_cutoff=0.7  # 只保留相似度 > 0.7 的結果
    )
    
    # 建立查詢引擎
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[similarity_postprocessor]
    )
    
    query = "深度學習的應用"
    print(f"查詢: {query}")
    
    response = query_engine.query(query)
    print(f"回答: {response.response}")
    print(f"過濾後的節點數: {len(response.source_nodes)}")

def custom_query_bundle_demo(index):
    """自定義查詢束示範"""
    print("\n📦 自定義查詢束示範...")
    
    # 建立自定義查詢束
    query_bundle = QueryBundle(
        query_str="雲端服務的類型",
        custom_embedding_strs=["cloud computing", "service types", "IaaS", "PaaS", "SaaS"]
    )
    
    # 建立檢索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    )
    
    # 執行檢索
    nodes = retriever.retrieve(query_bundle)
    
    print(f"檢索到的節點數: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"節點 {i+1}: {node.text[:100]}...")

def response_mode_comparison(index):
    """不同回應模式比較"""
    print("\n🔄 回應模式比較...")
    
    query = "什麼是機器學習？"
    print(f"查詢: {query}")
    
    # 測試不同的回應模式
    response_modes = [
        ResponseMode.DEFAULT,
        ResponseMode.COMPACT,
        ResponseMode.TREE_SUMMARIZE,
        ResponseMode.SIMPLE_SUMMARIZE
    ]
    
    for mode in response_modes:
        print(f"\n📋 {mode.value} 模式:")
        try:
            query_engine = index.as_query_engine(response_mode=mode)
            response = query_engine.query(query)
            print(f"回答: {response.response[:150]}...")
        except Exception as e:
            print(f"❌ 模式不支援: {e}")

def explain_retrieval_strategies():
    """解釋檢索策略"""
    print("\n📚 檢索策略說明:")
    print("""
    1. 🔍 向量相似性檢索
       - 基於語義相似性
       - 適合概念性查詢
       - 使用嵌入向量計算相似度
    
    2. 🔤 關鍵字檢索
       - 基於精確文字匹配
       - 適合特定術語查詢
       - 查詢速度極快
    
    3. 🔀 混合檢索
       - 結合向量和關鍵字檢索
       - 提供最佳查詢效果
       - 適合複雜查詢需求
    
    4. 🎯 重新排序檢索
       - 多階段檢索策略
       - 先廣泛檢索，再精確排序
       - 提高檢索準確性
    
    5. 🧠 語義檢索
       - 基於語義理解的檢索
       - 支援同義詞和相關概念
       - 查詢效果最佳
    """)

def explain_query_engines():
    """解釋查詢引擎類型"""
    print("\n🤖 查詢引擎類型:")
    print("""
    1. 🔍 標準查詢引擎
       - 檢索 + 生成
       - 適合一般問答
       - 平衡速度和品質
    
    2. 🎯 檢索器查詢引擎
       - 自定義檢索策略
       - 適合特定需求
       - 高度可客製化
    
    3. 📊 摘要查詢引擎
       - 基於摘要的查詢
       - 適合概覽性查詢
       - 查詢速度較快
    
    4. 🌳 樹狀查詢引擎
       - 層次性查詢
       - 適合結構化查詢
       - 支援複雜推理
    
    5. 🔄 子查詢引擎
       - 多查詢組合
       - 適合複雜問題
       - 提供全面回答
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習查詢與檢索...")
        
        # 解釋檢索策略
        explain_retrieval_strategies()
        
        # 解釋查詢引擎
        explain_query_engines()
        
        # 設定索引
        index = setup_index()
        if not index:
            print("❌ 無法設定索引，請檢查文件是否存在")
            exit(1)
        
        # 基本查詢示範
        basic_query_demo(index)
        
        # 進階檢索示範
        advanced_retrieval_demo(index)
        
        # 相似度過濾示範
        similarity_filtering_demo(index)
        
        # 自定義查詢束示範
        custom_query_bundle_demo(index)
        
        # 回應模式比較
        response_mode_comparison(index)
        
        print("\n🎉 查詢與檢索學習完成！")
        print("下一步：學習 RAG 系統整合")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()