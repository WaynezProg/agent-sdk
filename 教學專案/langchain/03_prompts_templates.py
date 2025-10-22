# 03_prompts_templates.py - 提示詞模板
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate, 
    PromptTemplate, 
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.messages import HumanMessage, AIMessage

# 載入環境變數
load_dotenv()

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

def demonstrate_basic_templates(llm):
    """示範基本模板"""
    print("📝 基本提示詞模板示範...")
    
    # ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的{role}，請用繁體中文回答。"),
        ("human", "請解釋什麼是{concept}？")
    ])
    
    # 格式化提示詞
    messages = chat_template.format_messages(
        role="技術顧問",
        concept="區塊鏈"
    )
    
    response = llm.invoke(messages)
    print(f"ChatPromptTemplate 結果: {response.content}")
    
    # PromptTemplate
    text_template = PromptTemplate(
        input_variables=["topic", "audience"],
        template="請為{audience}寫一篇關於{topic}的介紹文章，不超過200字："
    )
    
    formatted_prompt = text_template.format(
        topic="人工智慧",
        audience="小學生"
    )
    
    print(f"\nPromptTemplate 格式化結果: {formatted_prompt}")

def demonstrate_few_shot_learning(llm):
    """示範少樣本學習"""
    print("\n🎯 少樣本學習示範...")
    
    # 定義範例
    examples = [
        {
            "input": "什麼是機器學習？",
            "output": "機器學習是人工智慧的一個分支，它使電腦能夠在沒有明確編程的情況下學習和改進。"
        },
        {
            "input": "什麼是深度學習？",
            "output": "深度學習是機器學習的一個子集，使用人工神經網路來模擬人腦的工作方式。"
        },
        {
            "input": "什麼是自然語言處理？",
            "output": "自然語言處理是人工智慧的一個領域，專注於讓電腦理解、解釋和生成人類語言。"
        }
    ]
    
    # 創建少樣本模板
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="問題: {input}\n回答: {output}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="你是一個技術專家，請根據以下範例回答問題：",
        suffix="問題: {input}\n回答:",
        input_variables=["input"]
    )
    
    # 測試
    test_input = "什麼是電腦視覺？"
    formatted_prompt = few_shot_prompt.format(input=test_input)
    
    print(f"少樣本學習提示詞:\n{formatted_prompt}")
    
    # 使用 LLM 生成回答
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    print(f"\nLLM 回答: {response.content}")

def demonstrate_chat_few_shot(llm):
    """示範聊天少樣本學習"""
    print("\n💬 聊天少樣本學習示範...")
    
    # 定義聊天範例
    examples = [
        {
            "input": "你好，我想了解人工智慧",
            "output": "你好！我很樂意為您介紹人工智慧。人工智慧是讓機器模擬人類智慧的技術，包括學習、推理和解決問題的能力。"
        },
        {
            "input": "機器學習和深度學習有什麼不同？",
            "output": "機器學習是人工智慧的一個分支，而深度學習是機器學習的一個子集。深度學習使用多層神經網路來處理複雜的數據模式。"
        }
    ]
    
    # 創建聊天少樣本模板
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    
    few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    # 創建最終提示詞
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個友善的技術顧問，請根據以下對話範例來回答問題："),
        few_shot_chat_prompt,
        ("human", "{input}")
    ])
    
    # 測試
    test_input = "請解釋什麼是自然語言處理"
    messages = final_prompt.format_messages(input=test_input)
    
    response = llm.invoke(messages)
    print(f"聊天少樣本學習結果: {response.content}")

def demonstrate_example_selectors(llm):
    """示範範例選擇器"""
    print("\n🎲 範例選擇器示範...")
    
    # 定義更多範例
    examples = [
        {"input": "什麼是 AI？", "output": "AI 是人工智慧的縮寫。"},
        {"input": "什麼是機器學習？", "output": "機器學習是 AI 的一個分支。"},
        {"input": "什麼是深度學習？", "output": "深度學習是機器學習的一個子集。"},
        {"input": "什麼是自然語言處理？", "output": "自然語言處理是讓電腦理解人類語言的技術。"},
        {"input": "什麼是電腦視覺？", "output": "電腦視覺是讓電腦理解圖像和視頻的技術。"},
        {"input": "什麼是強化學習？", "output": "強化學習是通過獎懲機制學習的機器學習方法。"}
    ]
    
    # 創建基於長度的選擇器
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=PromptTemplate(
            input_variables=["input", "output"],
            template="問題: {input}\n回答: {output}"
        ),
        max_length=100
    )
    
    # 創建少樣本模板
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["input", "output"],
            template="問題: {input}\n回答: {output}"
        ),
        prefix="你是一個技術專家，請根據以下範例回答問題：",
        suffix="問題: {input}\n回答:",
        input_variables=["input"]
    )
    
    # 測試不同長度的輸入
    test_inputs = [
        "什麼是 AI？",
        "請詳細解釋什麼是人工智慧，包括其發展歷史和主要應用領域？"
    ]
    
    for test_input in test_inputs:
        print(f"\n測試輸入: {test_input}")
        formatted_prompt = few_shot_prompt.format(input=test_input)
        print(f"選擇的範例數量: {len(example_selector.select_examples({'input': test_input}))}")
        
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        print(f"回答: {response.content[:100]}...")

def demonstrate_partial_prompts():
    """示範部分提示詞"""
    print("\n🔧 部分提示詞示範...")
    
    # 創建基本模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一個{role}，請用{language}回答。"),
        ("human", "請解釋什麼是{concept}？")
    ])
    
    # 部分填充
    partial_template = template.partial(
        role="技術專家",
        language="繁體中文"
    )
    
    # 完整填充
    messages = partial_template.format_messages(concept="量子計算")
    
    print("部分提示詞結果:")
    for message in messages:
        print(f"  {message.type}: {message.content}")

def demonstrate_custom_prompts():
    """示範自定義提示詞"""
    print("\n🎨 自定義提示詞示範...")
    
    # 創建自定義模板
    custom_template = ChatPromptTemplate.from_messages([
        ("system", """你是一個專業的{domain}顧問。
        
你的任務是：
1. 分析用戶的問題
2. 提供準確的答案
3. 給出實用的建議
4. 使用{language}回答

請確保回答：
- 準確且專業
- 易於理解
- 包含實際例子
- 不超過{max_words}字"""),
        ("human", "問題：{question}")
    ])
    
    # 使用自定義模板
    messages = custom_template.format_messages(
        domain="人工智慧",
        language="繁體中文",
        max_words="200",
        question="如何開始學習機器學習？"
    )
    
    print("自定義提示詞:")
    for message in messages:
        print(f"  {message.type}: {message.content}")

def demonstrate_prompt_composition():
    """示範提示詞組合"""
    print("\n🔗 提示詞組合示範...")
    
    # 創建多個模板
    system_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個{role}，具有{experience}年的經驗。")
    ])
    
    context_template = ChatPromptTemplate.from_messages([
        ("system", "當前主題：{topic}")
    ])
    
    question_template = ChatPromptTemplate.from_messages([
        ("human", "問題：{question}")
    ])
    
    # 組合模板
    combined_template = system_template + context_template + question_template
    
    # 使用組合模板
    messages = combined_template.format_messages(
        role="資料科學家",
        experience="10",
        topic="機器學習",
        question="什麼是過擬合？"
    )
    
    print("組合提示詞:")
    for message in messages:
        print(f"  {message.type}: {message.content}")

def explain_prompt_techniques():
    """解釋提示詞技術"""
    print("\n📚 提示詞技術說明:")
    print("""
    1. 📝 基本模板
       - ChatPromptTemplate: 對話格式
       - PromptTemplate: 文本格式
       - 變數替換
       - 格式化輸出
    
    2. 🎯 少樣本學習
       - FewShotPromptTemplate
       - FewShotChatMessagePromptTemplate
       - 範例驅動學習
       - 提高回答品質
    
    3. 🎲 範例選擇器
       - LengthBasedExampleSelector
       - 動態範例選擇
       - 優化提示詞長度
       - 提高相關性
    
    4. 🔧 部分提示詞
       - 預填充變數
       - 模板重用
       - 提高效率
       - 減少重複
    
    5. 🎨 自定義提示詞
       - 特定領域優化
       - 結構化輸出
       - 角色定義
       - 約束條件
    
    6. 🔗 提示詞組合
       - 模組化設計
       - 靈活組合
       - 可重用組件
       - 複雜邏輯處理
    
    7. 💡 最佳實踐
       - 清晰的指令
       - 具體的範例
       - 適當的約束
       - 持續優化
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習提示詞模板...")
        
        # 解釋提示詞技術
        explain_prompt_techniques()
        
        # 設定 LLM
        llm = setup_llm()
        
        # 基本模板
        demonstrate_basic_templates(llm)
        
        # 少樣本學習
        demonstrate_few_shot_learning(llm)
        
        # 聊天少樣本學習
        demonstrate_chat_few_shot(llm)
        
        # 範例選擇器
        demonstrate_example_selectors(llm)
        
        # 部分提示詞
        demonstrate_partial_prompts()
        
        # 自定義提示詞
        demonstrate_custom_prompts()
        
        # 提示詞組合
        demonstrate_prompt_composition()
        
        print("\n🎉 提示詞模板學習完成！")
        print("下一步：學習鏈式處理")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()