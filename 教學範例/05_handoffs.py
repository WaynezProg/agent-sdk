# 05_handoffs.py
from agents import Agent, Runner, handoff

security_agent = Agent(
    name="Security",
    instructions="處理跟安全/門鎖/警報相關的需求，回答務實簡短。"
)

energy_agent = Agent(
    name="Energy",
    instructions="處理節能/電器耗能/省電建議相關的需求。"
)

triage = Agent(
    name="Triage",
    instructions=(
        "判斷使用者意圖：如果是安全相關就交給 Security；"
        "如果是節能相關就交給 Energy；否則自己回答。回覆繁體中文。"
    ),
    handoffs=[
        handoff(security_agent),   # 不帶第二參數，避免產生非法的 tool name
        handoff(energy_agent),
    ],
)

print(Runner.run_sync(triage, "我想知道怎麼降低待機耗電").final_output)
print(Runner.run_sync(triage, "想了解門窗是否鎖好要注意哪些項目").final_output)
