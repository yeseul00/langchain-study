"""1-3. LLM은 이전 대화를 기억하지 못한다.

실행: python 01_llm_basics/step3_memory.py
"""

import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)

# ── 실험 A: 따로따로 호출 ─────────────────────────────
print("=== 실험 A: 매번 새로 호출 ===")
llm.invoke("내 이름은 슬기야. 기억해 줘.")
answer = llm.invoke("내 이름이 뭐라고 했지? 한국어로 답해.")
print(answer.content)
print("→ 두 번째 호출은 첫 번째 호출 내용을 전혀 받지 못했다.\n")

# ── 실험 B: 대화 기록을 같이 보내기 ─────────────────────
print("=== 실험 B: 대화 기록(history)을 매번 같이 보냄 ===")
history = [HumanMessage(content="내 이름은 슬기야. 기억해 줘.")]

first = llm.invoke(history)
history.append(AIMessage(content=first.content))  # LLM 답도 기록에 추가

history.append(HumanMessage(content="내 이름이 뭐라고 했지? 한국어로 답해."))
answer = llm.invoke(history)
print(answer.content)

print(f"\n→ 이번에 보낸 메시지 수: {len(history)}개")
for m in history:
    print(f"   {type(m).__name__:<13} {m.content[:40]!r}")

# 정리:
#  '기억'은 LLM 안에 있는 게 아니라, 프로그램이 대화 목록을 들고 다니는 것이다.
#  대화가 길어질수록 보낼 글이 많아져 느려지고, 모델이 한 번에 읽을 수 있는
#  양(컨텍스트 길이)을 넘으면 앞부분을 잘라야 한다.
