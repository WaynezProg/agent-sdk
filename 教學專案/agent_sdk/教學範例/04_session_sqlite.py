from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="MemoryDemo",
    instructions="你會記住前文的地名與喜好，並用繁體中文回覆。"
)

session = SQLiteSession("demo_user", "../db/conversations.db")

print(Runner.run_sync(agent, "我住在台中，喜歡鹹酥雞。", session=session).final_output)
print(Runner.run_sync(agent, "剛剛我說我住哪？我喜歡吃什麼？", session=session).final_output)
