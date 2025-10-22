# 04_chains.py - 鏈式處理
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from pydantic import BaseModel, Field

# 載入環境變數
load_dotenv()

class AnalysisResult(BaseModel):
    """分析結果模型"""
    topic: str = Field(description="主題")
    summary: str = Field(description="摘要")
    key_points: list[str] = Field(description="關鍵點")
    applications: list[str] = Field(description="應用領域")

def setup_llm():
    """設定 LLM"""
    api_key = os.getenv('openaikey')
    if not api_key:
        raise ValueError("請在 .env 檔案中設定 openaikey")
    
    return ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1
    )

def demonstrate_basic_chains(llm):
    """示範基本鏈式處理"""
    print("🔗 基本鏈式處理示範...")
    
    # 創建提示詞模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個技術專家，請用繁體中文回答。"),
        ("human", "請解釋什麼是{concept}？")
    ])
    
    # 創建輸出解析器
    output_parser = StrOutputParser()
    
    # 創建鏈
    chain = prompt | llm | output_parser
    
    # 執行鏈
    result = chain.invoke({"concept": "機器學習"})
    print(f"鏈式處理結果: {result}")

def demonstrate_llm_chain(llm):
    """示範 LLMChain"""
    print("\n📋 LLMChain 示範...")
    
    # 創建提示詞模板
    prompt = PromptTemplate(
        input_variables=["topic", "audience"],
        template="請為{audience}寫一篇關於{topic}的介紹文章，不超過200字："
    )
    
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 執行鏈
    result = llm_chain.run(topic="人工智慧", audience="大學生")
    print(f"LLMChain 結果: {result}")

def demonstrate_sequential_chains(llm):
    """示範順序鏈"""
    print("\n🔄 順序鏈示範...")
    
    # 第一個鏈：生成摘要
    summary_prompt = PromptTemplate(
        input_variables=["topic"],
        template="請為{topic}寫一個簡短的摘要，不超過100字："
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
    
    # 第二個鏈：生成詳細說明
    detail_prompt = PromptTemplate(
        input_variables=["summary"],
        template="基於以下摘要，請提供更詳細的說明：{summary}"
    )
    detail_chain = LLMChain(llm=llm, prompt=detail_prompt, output_key="details")
    
    # 創建順序鏈
    sequential_chain = SequentialChain(
        chains=[summary_chain, detail_chain],
        input_variables=["topic"],
        output_variables=["summary", "details"]
    )
    
    # 執行順序鏈
    result = sequential_chain({"topic": "深度學習"})
    print(f"摘要: {result['summary']}")
    print(f"詳細說明: {result['details']}")

def demonstrate_simple_sequential_chain(llm):
    """示範簡單順序鏈"""
    print("\n🔗 簡單順序鏈示範...")
    
    # 第一個鏈
    first_prompt = PromptTemplate(
        input_variables=["topic"],
        template="請用一句話解釋什麼是{topic}："
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)
    
    # 第二個鏈
    second_prompt = PromptTemplate(
        input_variables=["text"],
        template="請將以下文字改寫成更專業的表達：{text}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    
    # 創建簡單順序鏈
    simple_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain],
        verbose=True
    )
    
    # 執行鏈
    result = simple_chain.run("量子計算")
    print(f"簡單順序鏈結果: {result}")

def demonstrate_custom_chains(llm):
    """示範自定義鏈"""
    print("\n🎨 自定義鏈示範...")
    
    # 定義自定義函數
    def extract_keywords(text: str) -> str:
        """提取關鍵字"""
        return f"關鍵字：{text.split()[:5]}"  # 簡單的關鍵字提取
    
    def format_output(text: str) -> str:
        """格式化輸出"""
        return f"📝 {text}"
    
    # 創建自定義鏈
    custom_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請用繁體中文回答。"),
            ("human", "請簡潔地解釋什麼是{concept}？")
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(extract_keywords)
        | RunnableLambda(format_output)
    )
    
    # 執行自定義鏈
    result = custom_chain.invoke({"concept": "區塊鏈"})
    print(f"自定義鏈結果: {result}")

def demonstrate_conditional_chains(llm):
    """示範條件鏈"""
    print("\n🔀 條件鏈示範...")
    
    # 條件判斷函數
    def route_question(inputs):
        """根據問題類型路由到不同的處理鏈"""
        question = inputs["question"]
        if "什麼是" in question:
            return "definition"
        elif "如何" in question:
            return "how_to"
        else:
            return "general"
    
    # 定義鏈
    definition_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請提供準確的定義。"),
            ("human", "請定義：{question}")
        ])
        | llm
        | StrOutputParser()
    )
    
    how_to_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請提供詳細的步驟說明。"),
            ("human", "請說明：{question}")
        ])
        | llm
        | StrOutputParser()
    )
    
    general_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個友善的助手，請回答問題。"),
            ("human", "{question}")
        ])
        | llm
        | StrOutputParser()
    )
    
    # 創建條件鏈
    from langchain_core.runnables import RunnableBranch
    
    conditional_chain = RunnableBranch(
        (lambda x: route_question(x) == "definition", definition_chain),
        (lambda x: route_question(x) == "how_to", how_to_chain),
        general_chain
    )
    
    # 測試不同類型的問題
    test_questions = [
        "什麼是機器學習？",
        "如何開始學習程式設計？",
        "今天天氣怎麼樣？"
    ]
    
    for question in test_questions:
        result = conditional_chain.invoke({"question": question})
        print(f"問題: {question}")
        print(f"回答: {result[:100]}...")
        print()

def demonstrate_parallel_chains(llm):
    """示範並行鏈"""
    print("\n⚡ 並行鏈示範...")
    
    # 創建多個並行鏈
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請提供簡潔的摘要。"),
            ("human", "請為{topic}寫一個摘要：")
        ])
        | llm
        | StrOutputParser()
    )
    
    pros_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請列出優點。"),
            ("human", "請列出{topic}的優點：")
        ])
        | llm
        | StrOutputParser()
    )
    
    cons_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術專家，請列出缺點。"),
            ("human", "請列出{topic}的缺點：")
        ])
        | llm
        | StrOutputParser()
    )
    
    # 創建並行鏈
    from langchain_core.runnables import RunnableParallel
    
    parallel_chain = RunnableParallel(
        summary=summary_chain,
        pros=pros_chain,
        cons=cons_chain
    )
    
    # 執行並行鏈
    result = parallel_chain.invoke({"topic": "人工智慧"})
    
    print("並行鏈結果:")
    print(f"摘要: {result['summary']}")
    print(f"優點: {result['pros']}")
    print(f"缺點: {result['cons']}")

def demonstrate_json_output_chain(llm):
    """示範 JSON 輸出鏈"""
    print("\n📊 JSON 輸出鏈示範...")
    
    # 創建 JSON 解析器
    json_parser = JsonOutputParser(pydantic_object=AnalysisResult)
    
    # 創建 JSON 輸出鏈
    json_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一個技術分析師，請分析給定的主題。"),
            ("human", "請分析{topic}，並以 JSON 格式返回結果。\n{format_instructions}")
        ])
        | llm
        | json_parser
    )
    
    # 執行 JSON 鏈
    result = json_chain.invoke({
        "topic": "機器學習",
        "format_instructions": json_parser.get_format_instructions()
    })
    
    print("JSON 輸出鏈結果:")
    print(f"主題: {result.topic}")
    print(f"摘要: {result.summary}")
    print(f"關鍵點: {result.key_points}")
    print(f"應用領域: {result.applications}")

def demonstrate_error_handling_chains(llm):
    """示範錯誤處理鏈"""
    print("\n🛡️ 錯誤處理鏈示範...")
    
    def safe_llm_call(inputs):
        """安全的 LLM 調用"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一個技術專家，請用繁體中文回答。"),
                ("human", "請解釋什麼是{concept}？")
            ])
            chain = prompt | llm | StrOutputParser()
            return chain.invoke(inputs)
        except Exception as e:
            return f"抱歉，處理時發生錯誤：{str(e)}"
    
    # 創建錯誤處理鏈
    error_handling_chain = RunnableLambda(safe_llm_call)
    
    # 測試正常情況
    result1 = error_handling_chain.invoke({"concept": "人工智慧"})
    print(f"正常結果: {result1[:100]}...")
    
    # 測試錯誤情況（使用無效的 LLM 配置）
    try:
        # 這裡可以模擬錯誤情況
        result2 = error_handling_chain.invoke({"concept": "測試錯誤處理"})
        print(f"錯誤處理結果: {result2}")
    except Exception as e:
        print(f"捕獲到錯誤: {e}")

def explain_chain_concepts():
    """解釋鏈式處理概念"""
    print("\n📚 鏈式處理概念說明:")
    print("""
    1. 🔗 基本鏈
       - 線性處理流程
       - 組件串聯
       - 簡單易用
       - 適合基本任務
    
    2. 📋 LLMChain
       - 專門的 LLM 鏈
       - 提示詞 + LLM
       - 輸出解析
       - 標準化流程
    
    3. 🔄 順序鏈
       - 多步驟處理
       - 前一步輸出作為下一步輸入
       - 複雜邏輯處理
       - 適合多階段任務
    
    4. 🎨 自定義鏈
       - 靈活的組件組合
       - 自定義函數
       - 複雜邏輯
       - 高度可定制
    
    5. 🔀 條件鏈
       - 根據條件選擇不同路徑
       - 動態路由
       - 智能決策
       - 適合複雜場景
    
    6. ⚡ 並行鏈
       - 同時執行多個鏈
       - 提高效率
       - 結果合併
       - 適合獨立任務
    
    7. 📊 結構化輸出
       - JSON 格式輸出
       - 類型安全
       - 易於處理
       - 適合 API 整合
    
    8. 🛡️ 錯誤處理
       - 異常捕獲
       - 優雅降級
       - 提高穩定性
       - 生產環境必備
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習鏈式處理...")
        
        # 解釋鏈式處理概念
        explain_chain_concepts()
        
        # 設定 LLM
        llm = setup_llm()
        
        # 基本鏈
        demonstrate_basic_chains(llm)
        
        # LLMChain
        demonstrate_llm_chain(llm)
        
        # 順序鏈
        demonstrate_sequential_chains(llm)
        
        # 簡單順序鏈
        demonstrate_simple_sequential_chain(llm)
        
        # 自定義鏈
        demonstrate_custom_chains(llm)
        
        # 條件鏈
        demonstrate_conditional_chains(llm)
        
        # 並行鏈
        demonstrate_parallel_chains(llm)
        
        # JSON 輸出鏈
        demonstrate_json_output_chain(llm)
        
        # 錯誤處理鏈
        demonstrate_error_handling_chains(llm)
        
        print("\n🎉 鏈式處理學習完成！")
        print("下一步：學習記憶機制")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()