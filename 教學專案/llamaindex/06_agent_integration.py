# 06_agent_integration.py - LlamaIndex 與 Agent SDK 整合
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from agents import Agent, Runner, function_tool
from typing import List, Dict, Any

# 載入環境變數
load_dotenv()

def setup_llamaindex_agent():
    """設定 LlamaIndex 與 Agent 整合系統"""
    print("🔧 設定 LlamaIndex + Agent 整合系統...")
    
    # 載入文件並建立索引
    documents_dir = "sample_documents"
    if not os.path.exists(documents_dir):
        print("❌ 找不到範例文件，請先執行 02_document_loading.py")
        return None
    
    reader = SimpleDirectoryReader(input_dir=documents_dir)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    print("✅ LlamaIndex 索引建立完成！")
    return index

def create_document_search_tool(index):
    """創建文件搜尋工具"""
    print("\n🔍 創建文件搜尋工具...")
    
    @function_tool
    def search_documents(query: str, top_k: int = 3) -> str:
        """
        在知識庫中搜尋相關文件內容
        
        Args:
            query: 搜尋查詢
            top_k: 返回的結果數量
            
        Returns:
            搜尋結果的摘要
        """
        try:
            # 建立查詢引擎
            query_engine = index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            # 執行查詢
            response = query_engine.query(query)
            
            # 格式化結果
            result = f"查詢: {query}\n"
            result += f"回答: {response.response}\n\n"
            result += "來源文件:\n"
            
            for i, node in enumerate(response.source_nodes[:top_k], 1):
                result += f"{i}. {node.text[:200]}...\n"
            
            return result
            
        except Exception as e:
            return f"搜尋時發生錯誤: {str(e)}"
    
    print("✅ 文件搜尋工具創建完成！")
    return search_documents

def create_document_analyzer_tool(index):
    """創建文件分析工具"""
    print("\n📊 創建文件分析工具...")
    
    @function_tool
    def analyze_document_content(topic: str) -> str:
        """
        分析知識庫中特定主題的內容
        
        Args:
            topic: 要分析的主題
            
        Returns:
            主題分析結果
        """
        try:
            # 建立查詢引擎
            query_engine = index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            
            # 執行分析查詢
            analysis_query = f"請詳細分析知識庫中關於 '{topic}' 的所有相關內容，包括定義、特點、應用等"
            response = query_engine.query(analysis_query)
            
            return f"主題分析: {topic}\n\n{response.response}"
            
        except Exception as e:
            return f"分析時發生錯誤: {str(e)}"
    
    print("✅ 文件分析工具創建完成！")
    return analyze_document_content

def create_knowledge_comparison_tool(index):
    """創建知識比較工具"""
    print("\n⚖️ 創建知識比較工具...")
    
    @function_tool
    def compare_topics(topic1: str, topic2: str) -> str:
        """
        比較知識庫中兩個主題的異同
        
        Args:
            topic1: 第一個主題
            topic2: 第二個主題
            
        Returns:
            比較分析結果
        """
        try:
            # 建立查詢引擎
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            
            # 執行比較查詢
            comparison_query = f"請比較 '{topic1}' 和 '{topic2}' 的異同點，包括定義、特點、應用領域等"
            response = query_engine.query(comparison_query)
            
            return f"主題比較: {topic1} vs {topic2}\n\n{response.response}"
            
        except Exception as e:
            return f"比較時發生錯誤: {str(e)}"
    
    print("✅ 知識比較工具創建完成！")
    return compare_topics

def create_smart_agent(index):
    """創建智能代理"""
    print("\n🤖 創建智能代理...")
    
    # 創建工具
    search_tool = create_document_search_tool(index)
    analyze_tool = create_document_analyzer_tool(index)
    compare_tool = create_knowledge_comparison_tool(index)
    
    # 創建智能代理
    smart_agent = Agent(
        name="KnowledgeAssistant",
        instructions="""
        你是一個專業的知識助手，擁有強大的文件搜尋和分析能力。
        
        你可以：
        1. 搜尋知識庫中的相關資訊
        2. 分析特定主題的詳細內容
        3. 比較不同主題的異同點
        
        請根據使用者的問題，選擇最適合的工具來提供準確、詳細的回答。
        回答時請使用繁體中文，並提供具體的資訊來源。
        """,
        tools=[search_tool, analyze_tool, compare_tool]
    )
    
    print("✅ 智能代理創建完成！")
    return smart_agent

def demonstrate_agent_capabilities(agent):
    """示範代理能力"""
    print("\n🎯 示範代理能力...")
    
    # 測試查詢
    test_queries = [
        "請搜尋關於人工智慧的資訊",
        "分析雲端運算的內容",
        "比較機器學習和深度學習的差異"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n測試 {i}: {query}")
        print("-" * 50)
        
        try:
            response = Runner.run_sync(agent, query)
            print(f"回答: {response.final_output}")
        except Exception as e:
            print(f"❌ 查詢失敗: {e}")

def create_specialized_agents(index):
    """創建專業化代理"""
    print("\n👥 創建專業化代理...")
    
    # 技術分析代理
    tech_agent = Agent(
        name="TechAnalyst",
        instructions="你是技術分析專家，專門分析技術概念和趨勢。",
        tools=[create_document_analyzer_tool(index)]
    )
    
    # 搜尋專家代理
    search_agent = Agent(
        name="SearchExpert", 
        instructions="你是搜尋專家，擅長在知識庫中快速找到相關資訊。",
        tools=[create_document_search_tool(index)]
    )
    
    # 比較分析代理
    comparison_agent = Agent(
        name="ComparisonExpert",
        instructions="你是比較分析專家，擅長分析不同概念間的異同。",
        tools=[create_knowledge_comparison_tool(index)]
    )
    
    print("✅ 專業化代理創建完成！")
    return tech_agent, search_agent, comparison_agent

def demonstrate_agent_collaboration(agents):
    """示範代理協作"""
    print("\n🤝 示範代理協作...")
    
    tech_agent, search_agent, comparison_agent = agents
    
    # 協作查詢
    collaboration_query = """
    請幫我分析人工智慧和雲端運算的關係：
    1. 先搜尋相關資訊
    2. 分析這兩個技術的特點
    3. 比較它們的異同點
    """
    
    print(f"協作查詢: {collaboration_query}")
    print("-" * 50)
    
    # 使用搜尋代理
    print("🔍 搜尋階段:")
    search_response = Runner.run_sync(search_agent, "搜尋人工智慧和雲端運算的相關資訊")
    print(f"搜尋結果: {search_response.final_output}")
    
    # 使用分析代理
    print("\n📊 分析階段:")
    analysis_response = Runner.run_sync(tech_agent, "分析人工智慧和雲端運算的技術特點")
    print(f"分析結果: {analysis_response.final_output}")
    
    # 使用比較代理
    print("\n⚖️ 比較階段:")
    comparison_response = Runner.run_sync(comparison_agent, "比較人工智慧和雲端運算的異同點")
    print(f"比較結果: {comparison_response.final_output}")

def explain_integration_benefits():
    """解釋整合優勢"""
    print("\n📚 整合優勢說明:")
    print("""
    LlamaIndex + Agent SDK 整合的優勢：
    
    1. 🔍 強大的檢索能力
       - 語義搜尋
       - 多文件檢索
       - 上下文理解
    
    2. 🤖 智能工具使用
       - 自動選擇合適工具
       - 動態查詢策略
       - 結果整合
    
    3. 🎯 專業化分工
       - 不同專業的代理
       - 協作解決複雜問題
       - 知識互補
    
    4. 🔄 靈活的工作流程
       - 多步驟推理
       - 動態調整策略
       - 持續學習改進
    
    5. 📊 豐富的應用場景
       - 智能客服
       - 知識管理
       - 研究助手
       - 教育輔助
    """)

def demonstrate_advanced_integration(index):
    """示範進階整合功能"""
    print("\n🚀 進階整合功能...")
    
    # 創建自定義工具
    @function_tool
    def smart_knowledge_query(question: str, context: str = "") -> str:
        """
        智能知識查詢，結合上下文進行深度分析
        
        Args:
            question: 問題
            context: 額外的上下文資訊
            
        Returns:
            智能分析結果
        """
        try:
            # 建立查詢引擎
            query_engine = index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )
            
            # 構建智能查詢
            if context:
                smart_query = f"基於以下上下文：{context}\n\n問題：{question}\n\n請提供詳細的分析和回答。"
            else:
                smart_query = f"問題：{question}\n\n請提供詳細的分析，包括相關概念、應用場景和實際例子。"
            
            response = query_engine.query(smart_query)
            
            return f"智能分析結果:\n{response.response}"
            
        except Exception as e:
            return f"智能查詢時發生錯誤: {str(e)}"
    
    # 創建進階代理
    advanced_agent = Agent(
        name="AdvancedKnowledgeAssistant",
        instructions="""
        你是一個進階知識助手，能夠進行深度分析和智能推理。
        你可以結合多種資訊來源，提供全面、準確的回答。
        """,
        tools=[smart_knowledge_query]
    )
    
    # 測試進階功能
    advanced_query = "請分析人工智慧在雲端運算環境中的應用，並提供具體的實施建議"
    print(f"進階查詢: {advanced_query}")
    
    response = Runner.run_sync(advanced_agent, advanced_query)
    print(f"進階回答: {response.final_output}")

if __name__ == "__main__":
    try:
        print("🚀 開始學習 LlamaIndex + Agent SDK 整合...")
        
        # 解釋整合優勢
        explain_integration_benefits()
        
        # 設定整合系統
        index = setup_llamaindex_agent()
        if not index:
            print("❌ 無法設定整合系統，請檢查文件是否存在")
            exit(1)
        
        # 創建智能代理
        smart_agent = create_smart_agent(index)
        
        # 示範代理能力
        demonstrate_agent_capabilities(smart_agent)
        
        # 創建專業化代理
        specialized_agents = create_specialized_agents(index)
        
        # 示範代理協作
        demonstrate_agent_collaboration(specialized_agents)
        
        # 示範進階整合
        demonstrate_advanced_integration(index)
        
        print("\n🎉 LlamaIndex + Agent SDK 整合學習完成！")
        print("下一步：學習進階功能和生產環境部署")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()