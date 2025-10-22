# 01_basic_setup.py - LangChain 基礎設定與概念介紹
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 載入環境變數
load_dotenv()

def setup_langchain():
    """設定 LangChain 的基本配置"""
    print("🚀 開始設定 LangChain...")
    
    # 檢查 API Key
    api_key = os.getenv('openaikey')
    if not api_key:
        raise ValueError("請在 .env 檔案中設定 openaikey")
    
    print(f"✅ API Key 已載入: {api_key[:10]}...")
    
    # 設定 LLM
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512
    )
    
    # 設定嵌入模型
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    print("✅ LangChain 設定完成！")
    print(f"   - LLM: {llm.model_name}")
    print(f"   - 嵌入模型: {embeddings.model}")
    
    return llm, embeddings

def test_basic_llm(llm):
    """測試基本 LLM 功能"""
    print("\n🧪 測試基本 LLM 功能...")
    
    # 基本對話
    messages = [
        SystemMessage(content="你是一個友善的助手，請用繁體中文回答。"),
        HumanMessage(content="請用一句話介紹什麼是 LangChain？")
    ]
    
    response = llm.invoke(messages)
    print(f"LLM 回應: {response.content}")
    
    # 測試流式回應
    print("\n🌊 測試流式回應:")
    for chunk in llm.stream(messages):
        print(chunk.content, end='', flush=True)
    print()

def test_prompt_templates():
    """測試提示詞模板"""
    print("\n📝 測試提示詞模板...")
    
    # 創建提示詞模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的{role}，請用繁體中文回答。"),
        ("human", "請解釋什麼是{concept}？")
    ])
    
    # 格式化提示詞
    messages = template.format_messages(
        role="技術顧問",
        concept="人工智慧"
    )
    
    print("格式化後的提示詞:")
    for message in messages:
        print(f"  {message.type}: {message.content}")
    
    return template

def test_embeddings(embeddings):
    """測試嵌入功能"""
    print("\n🧠 測試嵌入功能...")
    
    # 測試文字嵌入
    texts = [
        "人工智慧是電腦科學的一個分支",
        "機器學習是 AI 的重要組成部分",
        "深度學習使用神經網路"
    ]
    
    print("測試文字嵌入:")
    for i, text in enumerate(texts, 1):
        embedding = embeddings.embed_query(text)
        print(f"文字 {i}: {text}")
        print(f"嵌入維度: {len(embedding)}")
        print(f"前 5 個值: {embedding[:5]}")
        print()

def explain_langchain_concepts():
    """解釋 LangChain 核心概念"""
    print("\n📚 LangChain 核心概念:")
    print("""
    1. 🧠 LLM (Large Language Models)
       - 大型語言模型
       - 支援多種模型提供商
       - 統一的介面
    
    2. 📝 Prompts (提示詞)
       - 模板化提示詞
       - 變數替換
       - 多種格式支援
    
    3. 🔗 Chains (鏈式處理)
       - 串聯多個 LLM 調用
       - 複雜的處理流程
       - 可組合的組件
    
    4. 🧠 Memory (記憶)
       - 會話記憶
       - 長期記憶
       - 多種記憶類型
    
    5. 🤖 Agents (智能代理)
       - 工具使用能力
       - 決策邏輯
       - 自主執行任務
    
    6. 🔧 Tools (工具)
       - 外部工具整合
       - 自定義工具
       - 豐富的工具生態
    
    7. 📊 Retrievers (檢索器)
       - 文件檢索
       - 向量搜尋
       - RAG 系統核心
    
    8. 🗃️ Vector Stores (向量儲存)
       - 向量資料庫
       - 相似性搜尋
       - 持久化儲存
    """)

def demonstrate_basic_workflow(llm, template):
    """示範基本工作流程"""
    print("\n🔄 基本工作流程示範...")
    
    # 使用模板和 LLM
    chain = template | llm
    
    # 執行鏈式處理
    response = chain.invoke({
        "role": "教育專家",
        "concept": "機器學習"
    })
    
    print(f"鏈式處理結果: {response.content}")

def explain_langchain_advantages():
    """解釋 LangChain 優勢"""
    print("\n✨ LangChain 優勢:")
    print("""
    1. 🔧 模組化設計
       - 可組合的組件
       - 靈活的架構
       - 易於擴展
    
    2. 🌐 豐富的生態系統
       - 多種 LLM 支援
       - 豐富的工具庫
       - 活躍的社群
    
    3. 🚀 生產就緒
       - 企業級功能
       - 效能優化
       - 監控和日誌
    
    4. 📚 完整的文檔
       - 詳細的教程
       - 豐富的範例
       - 最佳實踐指南
    
    5. 🔄 持續更新
       - 快速迭代
       - 新功能支援
       - 社群貢獻
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習 LangChain...")
        
        # 解釋核心概念
        explain_langchain_concepts()
        
        # 解釋優勢
        explain_langchain_advantages()
        
        # 設定 LangChain
        llm, embeddings = setup_langchain()
        
        # 測試基本功能
        test_basic_llm(llm)
        
        # 測試提示詞模板
        template = test_prompt_templates()
        
        # 測試嵌入功能
        test_embeddings(embeddings)
        
        # 示範基本工作流程
        demonstrate_basic_workflow(llm, template)
        
        print("\n🎉 LangChain 基礎設定完成！")
        print("下一步：學習 LLM 模型使用")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()