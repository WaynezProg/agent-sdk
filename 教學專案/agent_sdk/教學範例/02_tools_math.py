# 02_tools_math.py
from agents import Agent, Runner, function_tool

@function_tool
def add(a: float, b: float) -> float:
    """回傳 a + b"""
    return a + b

@function_tool
def fib(n: int) -> list[int]:
    """回傳前 n 個 Fibonacci 數列"""
    seq = [0, 1]
    for _ in range(max(0, n-2)):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

math_agent = Agent(
    name="MathAgent",
    instructions="你會在需要時使用可用的工具來計算，答案請用繁體中文。",
    tools=[add, fib],
)

print(Runner.run_sync(math_agent, "幫我算 12.5 + 7.25").final_output)
print(Runner.run_sync(math_agent, "給我前 10 個費波那契數列").final_output)
