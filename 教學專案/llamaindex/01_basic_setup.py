# 01_basic_setup.py - LlamaIndex 基礎設定與概念介紹
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 載入環境變數
load_dotenv()

def setup_llamaindex():
    """設定 LlamaIndex 的基本配置"""
    print("🚀 開始設定 LlamaIndex...")
    
    # 檢查 API Key
    api_key = os.getenv('openaikey')
    if not api_key:
        raise ValueError("請在 .env 檔案中設定 openaikey")
    
    print(f"✅ API Key 已載入: {api_key[:10]}...")
    
    # 設定 LLM
    llm = OpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512
    )
    
    # 設定嵌入模型
    embed_model = OpenAIEmbedding(
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    # 全域設定
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200
    
    print("✅ LlamaIndex 設定完成！")
    print(f"   - LLM: {llm.model}")
    print(f"   - 嵌入模型: {embed_model.model_name}")
    print(f"   - 文件分塊大小: {Settings.chunk_size}")
    print(f"   - 分塊重疊: {Settings.chunk_overlap}")
    
    return llm, embed_model

def test_basic_functionality():
    """測試基本功能"""
    print("\n🧪 測試基本功能...")
    
    # 測試 LLM
    llm = Settings.llm
    response = llm.complete("請用一句話介紹什麼是 RAG？")
    print(f"LLM 回應: {response.text}")
    
    # 測試嵌入模型
    embed_model = Settings.embed_model
    embeddings = embed_model.get_text_embedding("這是一個測試文字")
    print(f"嵌入向量維度: {len(embeddings)}")
    print(f"嵌入向量前 5 個值: {embeddings[:5]}")

def explain_llamaindex_concepts():
    """解釋 LlamaIndex 核心概念"""
    print("\n📚 LlamaIndex 核心概念:")
    print("""
    1. 📄 Document: 原始文件（PDF、Word、網頁等）
    2. 🔤 Node: 文件被分割後的小塊文字
    3. 🧠 Index: 向量索引，用於快速檢索
    4. 🔍 Retriever: 檢索器，根據查詢找到相關文件
    5. 🤖 Query Engine: 查詢引擎，整合檢索和生成
    6. 📊 Response: 最終的回答結果
    
    工作流程：
    文件 → 分割 → 向量化 → 建立索引 → 查詢 → 檢索 → 生成回答
    """)

if __name__ == "__main__":
    try:
        # 設定 LlamaIndex
        llm, embed_model = setup_llamaindex()
        
        # 解釋核心概念
        explain_llamaindex_concepts()
        
        # 測試基本功能
        test_basic_functionality()
        
        print("\n🎉 基礎設定完成！您可以繼續學習下一個範例。")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        print("請檢查您的 API Key 設定和網路連線。")