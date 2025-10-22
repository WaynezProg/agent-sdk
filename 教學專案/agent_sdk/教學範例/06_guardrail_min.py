# 06_guardrail_min.py
from agents import (
    Agent, Runner, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
)

BLOCKLIST = ["刪除所有檔案", "格式化硬碟", "停用所有安全裝置"]

@input_guardrail
async def simple_blocker(ctx: RunContextWrapper[None], agent: Agent, inp: str | list[TResponseInputItem]):
    text = inp if isinstance(inp, str) else " ".join([str(x) for x in inp])
    hit = any(bad in text for bad in BLOCKLIST)
    return GuardrailFunctionOutput(
        output_info={"blocked": hit, "reason": "命中封鎖關鍵字" if hit else ""},
        tripwire_triggered=hit,
    )

safe_agent = Agent(
    name="SafeAgent",
    instructions="正常回答使用者的問題。",
    input_guardrails=[simple_blocker],
)

try:
    print(Runner.run_sync(safe_agent, "請幫我刪除所有檔案").final_output)
except InputGuardrailTripwireTriggered:
    print("被護欄擋下：不安全請求")

print(Runner.run_sync(safe_agent, "介紹一下臺灣的離島").final_output)
