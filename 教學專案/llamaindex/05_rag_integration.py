# 05_rag_integration.py - RAG 系統整合與優化
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Settings,
    PromptTemplate
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# 載入環境變數
load_dotenv()

def setup_rag_system():
    """設定完整的 RAG 系統"""
    print("🔧 設定 RAG 系統...")
    
    # 設定 LLM
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.llm = llm
    
    # 載入文件
    documents_dir = "sample_documents"
    if not os.path.exists(documents_dir):
        print("❌ 找不到範例文件，請先執行 02_document_loading.py")
        return None
    
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data()
    
    # 建立索引
    index = VectorStoreIndex.from_documents(documents)
    
    print("✅ RAG 系統設定完成！")
    return index

def basic_rag_demo(index):
    """基本 RAG 示範"""
    print("\n🤖 基本 RAG 示範...")
    
    # 建立查詢引擎
    query_engine = index.as_query_engine(
        response_mode="compact",
        similarity_top_k=3
    )
    
    # 測試查詢
    queries = [
        "請總結人工智慧的主要應用領域",
        "雲端運算的三種服務類型是什麼？",
        "機器學習和深度學習有什麼關係？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n查詢 {i}: {query}")
        response = query_engine.query(query)
        print(f"回答: {response.response}")
        print(f"來源節點數: {len(response.source_nodes)}")

def custom_prompt_rag(index):
    """自定義提示詞的 RAG"""
    print("\n✏️ 自定義提示詞 RAG...")
    
    # 自定義提示詞模板
    custom_prompt = PromptTemplate(
        """你是一個專業的技術顧問，請根據以下上下文資訊回答問題。

上下文資訊：
{context_str}

問題：{query_str}

請提供：
1. 簡潔明確的回答
2. 相關的技術細節
3. 實際應用建議

回答："""
    )
    
    # 建立查詢引擎
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,
        response_mode="compact"
    )
    
    query = "如何選擇適合的雲端服務類型？"
    print(f"查詢: {query}")
    
    response = query_engine.query(query)
    print(f"回答: {response.response}")

def multi_document_rag(index):
    """多文件 RAG 示範"""
    print("\n📚 多文件 RAG 示範...")
    
    # 建立多個查詢引擎工具
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(),
            description="用於回答關於人工智慧和機器學習的問題"
        )
    ]
    
    # 建立子查詢引擎
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=False
    )
    
    # 複雜查詢
    complex_query = """
    請比較人工智慧和雲端運算的發展趨勢，
    並分析它們如何相互影響和促進。
    """
    
    print(f"複雜查詢: {complex_query}")
    response = sub_question_engine.query(complex_query)
    print(f"回答: {response.response}")

def rag_with_metadata_filtering(index):
    """帶有元數據過濾的 RAG"""
    print("\n🏷️ 元數據過濾 RAG...")
    
    # 建立帶有元數據過濾的查詢引擎
    query_engine = index.as_query_engine(
        filters={"file_path": "sample_ai.txt"},  # 只查詢特定文件
        response_mode="compact"
    )
    
    query = "人工智慧有哪些應用領域？"
    print(f"查詢: {query}")
    print("過濾條件: 只查詢 sample_ai.txt 文件")
    
    response = query_engine.query(query)
    print(f"回答: {response.response}")

def rag_performance_optimization(index):
    """RAG 效能優化示範"""
    print("\n⚡ RAG 效能優化...")
    
    # 優化設定
    optimized_query_engine = index.as_query_engine(
        response_mode="compact",
        similarity_top_k=2,  # 減少檢索數量
        streaming=True,      # 啟用串流
        verbose=True         # 顯示詳細資訊
    )
    
    query = "什麼是深度學習？"
    print(f"查詢: {query}")
    
    # 執行查詢並測量時間
    import time
    start_time = time.time()
    
    response = optimized_query_engine.query(query)
    
    end_time = time.time()
    print(f"查詢時間: {end_time - start_time:.2f} 秒")
    print(f"回答: {response.response}")

def explain_rag_components():
    """解釋 RAG 系統組件"""
    print("\n📚 RAG 系統組件說明:")
    print("""
    RAG (Retrieval-Augmented Generation) 系統包含以下組件：
    
    1. 📄 文件載入器 (Document Loader)
       - 載入各種格式的文件
       - 支援 PDF、Word、網頁等
       - 提取文字和元數據
    
    2. ✂️ 文字分割器 (Text Splitter)
       - 將長文件分割成小塊
       - 保持語義完整性
       - 控制塊大小和重疊
    
    3. 🧠 嵌入模型 (Embedding Model)
       - 將文字轉換為向量
       - 捕捉語義相似性
       - 支援多種預訓練模型
    
    4. 💾 向量儲存 (Vector Store)
       - 儲存文件向量
       - 支援快速相似性搜尋
       - 持久化儲存
    
    5. 🔍 檢索器 (Retriever)
       - 根據查詢檢索相關文件
       - 支援多種檢索策略
       - 可自定義檢索邏輯
    
    6. 🤖 生成模型 (LLM)
       - 基於檢索內容生成回答
       - 整合上下文資訊
       - 提供自然語言回答
    """)

def explain_rag_optimization():
    """解釋 RAG 優化策略"""
    print("\n⚡ RAG 優化策略:")
    print("""
    1. 📊 檢索優化
       - 調整相似度閾值
       - 使用混合檢索策略
       - 實施重新排序機制
    
    2. 🎯 生成優化
       - 自定義提示詞模板
       - 調整溫度參數
       - 使用更好的 LLM 模型
    
    3. 💾 儲存優化
       - 選擇合適的向量資料庫
       - 實施索引壓縮
       - 使用快取機制
    
    4. 🔄 流程優化
       - 並行處理查詢
       - 實施串流回應
       - 使用非同步處理
    
    5. 📈 監控優化
       - 追蹤查詢效能
       - 分析使用者回饋
       - 持續改進系統
    """)

def demonstrate_rag_evaluation():
    """示範 RAG 評估方法"""
    print("\n📊 RAG 評估示範...")
    
    # 建立查詢引擎
    query_engine = index.as_query_engine()
    
    # 測試查詢和預期答案
    test_cases = [
        {
            "query": "什麼是人工智慧？",
            "expected_keywords": ["電腦科學", "機器學習", "人類智慧"]
        },
        {
            "query": "雲端運算的優勢？",
            "expected_keywords": ["成本效益", "可擴展性", "靈活性"]
        }
    ]
    
    print("🧪 評估測試:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n測試 {i}: {test_case['query']}")
        
        response = query_engine.query(test_case['query'])
        answer = response.response.lower()
        
        # 檢查關鍵字
        found_keywords = []
        for keyword in test_case['expected_keywords']:
            if keyword in answer:
                found_keywords.append(keyword)
        
        print(f"回答: {response.response}")
        print(f"找到的關鍵字: {found_keywords}")
        print(f"關鍵字覆蓋率: {len(found_keywords)}/{len(test_case['expected_keywords'])}")

if __name__ == "__main__":
    try:
        print("🚀 開始學習 RAG 系統整合...")
        
        # 解釋 RAG 組件
        explain_rag_components()
        
        # 解釋優化策略
        explain_rag_optimization()
        
        # 設定 RAG 系統
        index = setup_rag_system()
        if not index:
            print("❌ 無法設定 RAG 系統，請檢查文件是否存在")
            exit(1)
        
        # 基本 RAG 示範
        basic_rag_demo(index)
        
        # 自定義提示詞 RAG
        custom_prompt_rag(index)
        
        # 多文件 RAG
        multi_document_rag(index)
        
        # 元數據過濾 RAG
        rag_with_metadata_filtering(index)
        
        # 效能優化 RAG
        rag_performance_optimization(index)
        
        # RAG 評估
        demonstrate_rag_evaluation()
        
        print("\n🎉 RAG 系統整合學習完成！")
        print("下一步：學習與 Agent SDK 的整合")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()