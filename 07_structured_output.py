# 07_structured_output.py
from typing import List
from pydantic import BaseModel
from agents import Agent, Runner

class ShoppingList(BaseModel):
    items: List[str]
    budget_ntd: int

agent = Agent(
    name="StructAgent",
    instructions="輸出精準結構。請只回傳 JSON，不要多餘描述。",
    output_type=ShoppingList,
)

res = Runner.run_sync(agent, "幫我規劃三樣晚餐食材，總預算 300 台幣")
print(res.final_output)  # -> pydantic 物件 (ShoppingList)
print(res.final_output.items, res.final_output.budget_ntd)
