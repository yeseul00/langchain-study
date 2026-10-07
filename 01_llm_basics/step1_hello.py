"""1-1. LLM에게 질문하고 답 받기.

실행: python 01_llm_basics/step1_hello.py
"""

import os

from langchain_ollama import ChatOllama

# 1) 사용할 LLM 준비 — Ollama에 받아 둔 모델 이름을 적는다.
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0,  # 0 = 매번 비슷한 답, 높을수록 다양한(창의적인) 답
)

# 2) 질문 보내기 — 문자열을 그대로 넣어도 된다.
#response = llm.invoke("서버 응답이 느려질 때 가장 먼저 확인할 것 3가지를 한국어로 짧게 알려줘.")
response = llm.invoke('서버 응답이 느려질 때 확인할 것 3가지를 {"items": ["...", "...", "..."]} 형식의 JSON으로만 답해. 다른 말은 쓰지 마.  답은 한국어로 알려줘.')

# 3) 결과 확인 — 답은 AIMessage 객체로 오고, 본문은 .content 에 있다.
print("=== 답변 ===")
print(response.content)

print("\n=== 응답 객체의 종류 ===")
print(type(response).__name__)
