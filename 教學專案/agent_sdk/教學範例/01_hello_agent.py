# 01_hello_agent.py
import os
from dotenv import load_dotenv
from agents import Agent, Runner

# 載入 .env 檔案中的環境變數
load_dotenv()

# 現在可以直接使用環境變數
openai_key = os.getenv('openaikey')
print(f"已載入 OpenAI API Key: {openai_key[:10]}...")

agent = Agent(
    name="Greeter",
    instructions="You are a concise assistant. Reply in Traditional Chinese."
)

result = Runner.run_sync(agent, "請用一句話自我介紹")
print(result.final_output)
