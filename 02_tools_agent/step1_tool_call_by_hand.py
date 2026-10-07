"""2-1. 도구 호출을 손으로 한 바퀴 돌려 보기.

LLM은 도구를 '요청'만 하고, 실행은 우리 코드가 한다는 걸 눈으로 확인한다.

실행: python 02_tools_agent/step1_tool_call_by_hand.py
"""

import os

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools import TOOLS

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)

# 1) LLM에게 "이런 도구들이 있어"라고 알려 준다.
llm_with_tools = llm.bind_tools(TOOLS)
tools_by_name = {t.name: t for t in TOOLS}

messages = [HumanMessage(content="payment-service 장애야. 누구를 호출해야 해?")]

# 2) 첫 호출 — LLM은 답 대신 '도구 호출 요청'을 돌려준다.
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print("=== ① LLM의 응답 ===")
print("본문(content):", repr(ai_msg.content))
print("도구 호출 요청(tool_calls):")
for tc in ai_msg.tool_calls:
    print(f"   - {tc['name']}({tc['args']})")

if not ai_msg.tool_calls:
    print("\nLLM이 도구를 부르지 않았습니다. 다시 실행하거나 질문을 바꿔 보세요.")
    raise SystemExit

# 3) 요청받은 도구를 '우리 코드'가 실행하고, 결과를 ToolMessage로 돌려준다.
print("\n=== ② 파이썬이 도구 실행 ===")
for tc in ai_msg.tool_calls:
    result = tools_by_name[tc["name"]].invoke(tc["args"])
    print(f"   {tc['name']} → {result}")
    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

# 4) 도구 결과까지 포함해서 다시 LLM에게 보낸다 → 이제 최종 답을 쓴다.
final = llm_with_tools.invoke(messages)
print("\n=== ③ 최종 답변 ===")
if final.tool_calls:
    # LLM이 도구를 또 요청한 경우 — 손으로 짠 루프는 한 바퀴만 돌기 때문에 여기서 멈춘다.
    print("LLM이 도구를 한 번 더 요청했습니다:", [tc["name"] for tc in final.tool_calls])
    print("→ 이렇게 '몇 바퀴 돌지 모르는' 반복을 자동으로 처리하는 게 step2의 에이전트입니다.")
else:
    print(final.content)

print(f"\n(이번 대화에 오간 메시지: {[type(m).__name__ for m in messages + [final]]})")

# 정리:
#  ① LLM 판단 → ② 코드 실행 → ③ LLM 답변.
#  도구가 여러 번 필요하면 ①②를 반복해야 한다. 이 반복을 자동으로 해 주는 게 '에이전트'.
