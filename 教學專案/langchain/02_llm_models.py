# 02_llm_models.py - LLM 模型使用
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# 載入環境變數
load_dotenv()

class PersonInfo(BaseModel):
    """人員資訊模型"""
    name: str = Field(description="姓名")
    age: int = Field(description="年齡")
    occupation: str = Field(description="職業")
    hobbies: list[str] = Field(description="愛好列表")

def setup_llm_models():
    """設定不同的 LLM 模型"""
    print("🔧 設定 LLM 模型...")
    
    api_key = os.getenv('openaikey')
    if not api_key:
        raise ValueError("請在 .env 檔案中設定 openaikey")
    
    # ChatOpenAI (推薦用於對話)
    chat_llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=512
    )
    
    # OpenAI (用於文本生成)
    text_llm = OpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo-instruct",
        temperature=0.1,
        max_tokens=512
    )
    
    print("✅ LLM 模型設定完成！")
    print(f"   - ChatOpenAI: {chat_llm.model_name}")
    print(f"   - OpenAI: {text_llm.model_name}")
    
    return chat_llm, text_llm

def demonstrate_chat_llm(chat_llm):
    """示範 ChatOpenAI 使用"""
    print("\n💬 ChatOpenAI 示範...")
    
    # 基本對話
    messages = [
        SystemMessage(content="你是一個友善的助手，請用繁體中文回答。"),
        HumanMessage(content="請介紹一下 Python 程式語言的特點。")
    ]
    
    response = chat_llm.invoke(messages)
    print(f"回應: {response.content}")
    
    # 多輪對話
    print("\n🔄 多輪對話示範:")
    conversation = [
        SystemMessage(content="你是一個技術顧問，請用繁體中文回答。"),
        HumanMessage(content="什麼是機器學習？"),
        AIMessage(content="機器學習是人工智慧的一個分支，它使電腦能夠在沒有明確編程的情況下學習和改進。"),
        HumanMessage(content="它有哪些主要類型？")
    ]
    
    response = chat_llm.invoke(conversation)
    print(f"回應: {response.content}")

def demonstrate_text_llm(text_llm):
    """示範 OpenAI 文本生成"""
    print("\n📝 OpenAI 文本生成示範...")
    
    # 基本文本生成
    prompt = "請用繁體中文寫一段關於人工智慧發展的短文，不超過100字："
    
    response = text_llm.invoke(prompt)
    print(f"生成的文本: {response}")
    
    # 創意寫作
    creative_prompt = """
    請創作一個關於機器人與人類友誼的短故事，要求：
    1. 用繁體中文
    2. 不超過200字
    3. 包含對話
    4. 有溫暖的結局
    """
    
    creative_response = text_llm.invoke(creative_prompt)
    print(f"\n創意故事: {creative_response}")

def demonstrate_prompt_templates(chat_llm):
    """示範提示詞模板"""
    print("\n📋 提示詞模板示範...")
    
    # ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的{role}，具有豐富的{field}知識。"),
        ("human", "請解釋什麼是{concept}，並提供一個實際應用例子。")
    ])
    
    # 格式化並執行
    messages = chat_template.format_messages(
        role="資料科學家",
        field="機器學習",
        concept="深度學習"
    )
    
    response = chat_llm.invoke(messages)
    print(f"模板化回應: {response.content}")
    
    # PromptTemplate (用於文本生成)
    text_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="請用{style}的風格，寫一篇關於{topic}的文章，不超過150字："
    )
    
    formatted_prompt = text_template.format(
        topic="量子計算",
        style="科普"
    )
    
    print(f"\n格式化提示詞: {formatted_prompt}")

def demonstrate_output_parsers(chat_llm):
    """示範輸出解析器"""
    print("\n🔍 輸出解析器示範...")
    
    # 字串解析器
    str_parser = StrOutputParser()
    
    # JSON 解析器
    json_parser = JsonOutputParser(pydantic_object=PersonInfo)
    
    # 使用字串解析器
    str_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個友善的助手，請用繁體中文回答。"),
        ("human", "請用一句話介紹什麼是區塊鏈技術。")
    ])
    
    str_chain = str_template | chat_llm | str_parser
    str_result = str_chain.invoke({})
    print(f"字串解析結果: {str_result}")
    
    # 使用 JSON 解析器
    json_template = ChatPromptTemplate.from_messages([
        ("system", "你是一個資料分析師，請根據用戶描述生成結構化資訊。"),
        ("human", "請分析以下描述並生成 JSON 格式的人員資訊：{description}"),
        ("system", "請使用以下格式：{format_instructions}")
    ])
    
    json_chain = json_template | chat_llm | json_parser
    
    description = "張小明，25歲，是一名軟體工程師，喜歡程式設計、閱讀和爬山。"
    json_result = json_chain.invoke({
        "description": description,
        "format_instructions": json_parser.get_format_instructions()
    })
    
    print(f"\nJSON 解析結果: {json_result}")

def demonstrate_streaming(chat_llm):
    """示範串流回應"""
    print("\n🌊 串流回應示範...")
    
    messages = [
        SystemMessage(content="你是一個技術專家，請用繁體中文回答。"),
        HumanMessage(content="請詳細解釋什麼是雲端運算，包括其特點和優勢。")
    ]
    
    print("串流回應:")
    for chunk in chat_llm.stream(messages):
        print(chunk.content, end='', flush=True)
    print()

def demonstrate_batch_processing(chat_llm):
    """示範批次處理"""
    print("\n📦 批次處理示範...")
    
    # 準備多個提示詞
    prompts = [
        "什麼是人工智慧？",
        "什麼是機器學習？",
        "什麼是深度學習？"
    ]
    
    # 批次處理
    messages_list = [
        [SystemMessage(content="你是一個技術顧問，請用繁體中文簡潔回答。"),
         HumanMessage(content=prompt)]
        for prompt in prompts
    ]
    
    responses = chat_llm.batch(messages_list)
    
    print("批次處理結果:")
    for i, response in enumerate(responses, 1):
        print(f"{i}. {response.content}")

def demonstrate_model_parameters():
    """示範模型參數調整"""
    print("\n⚙️ 模型參數示範...")
    
    api_key = os.getenv('openaikey')
    
    # 不同溫度設定
    temperatures = [0.1, 0.5, 0.9]
    prompt = "請創作一個關於未來科技的短故事："
    
    for temp in temperatures:
        print(f"\n溫度 {temp}:")
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=temp,
            max_tokens=100
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"回應: {response.content[:100]}...")

def explain_llm_models():
    """解釋 LLM 模型類型"""
    print("\n📚 LLM 模型類型說明:")
    print("""
    1. 💬 ChatOpenAI
       - 專為對話設計
       - 支援多輪對話
       - 結構化訊息格式
       - 推薦用於聊天應用
    
    2. 📝 OpenAI
       - 專為文本生成設計
       - 單次提示詞處理
       - 適合創意寫作
       - 推薦用於文本生成
    
    3. 🔧 模型參數
       - temperature: 控制創意程度 (0-1)
       - max_tokens: 最大輸出長度
       - top_p: 核採樣參數
       - frequency_penalty: 頻率懲罰
    
    4. 📊 輸出格式
       - 字串輸出
       - JSON 輸出
       - 結構化輸出
       - 自定義解析器
    
    5. 🚀 效能優化
       - 批次處理
       - 串流回應
       - 快取機制
       - 並行處理
    """)

if __name__ == "__main__":
    try:
        print("🚀 開始學習 LLM 模型使用...")
        
        # 解釋模型類型
        explain_llm_models()
        
        # 設定模型
        chat_llm, text_llm = setup_llm_models()
        
        # 示範 ChatOpenAI
        demonstrate_chat_llm(chat_llm)
        
        # 示範 OpenAI
        demonstrate_text_llm(text_llm)
        
        # 示範提示詞模板
        demonstrate_prompt_templates(chat_llm)
        
        # 示範輸出解析器
        demonstrate_output_parsers(chat_llm)
        
        # 示範串流回應
        demonstrate_streaming(chat_llm)
        
        # 示範批次處理
        demonstrate_batch_processing(chat_llm)
        
        # 示範模型參數
        demonstrate_model_parameters()
        
        print("\n🎉 LLM 模型學習完成！")
        print("下一步：學習提示詞模板")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()