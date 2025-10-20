# 03_tools_memory_todo.py
from agents import Agent, Runner, function_tool

TODO = []

@function_tool
def add_todo(item: str) -> str:
    """加入一個待辦事項"""
    TODO.append(item)
    return f"已加入待辦：{item}"

@function_tool
def list_todos() -> list[str]:
    """列出所有待辦"""
    return TODO

@function_tool
def clear_todos() -> str:
    """清空所有待辦"""
    TODO.clear()
    return "已清空"

todo_agent = Agent(
    name="TodoAgent",
    instructions="你是待辦管理員。請用工具新增、列出、清空待辦，回覆要簡潔。",
    tools=[add_todo, list_todos, clear_todos],
)

print(Runner.run_sync(todo_agent, "新增待辦：買牛奶").final_output)
print(Runner.run_sync(todo_agent, "再新增：寫 HomeX 企劃").final_output)
print(Runner.run_sync(todo_agent, "列出所有待辦").final_output)
print(Runner.run_sync(todo_agent, "清空待辦").final_output)
print(Runner.run_sync(todo_agent, "列出所有待辦").final_output)
