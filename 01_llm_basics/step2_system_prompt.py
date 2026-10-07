"""1-2. 시스템 프롬프트로 역할 바꾸기.

같은 질문이라도 시스템 프롬프트(역할 지시)에 따라 답이 달라진다.

실행: python 01_llm_basics/step2_system_prompt.py
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)

question = "데이터베이스 커넥션 풀이 고갈됐다는 게 무슨 뜻이야?"

# 역할만 바꿔 가며 같은 질문을 던진다.
roles = {
    "시니어 엔지니어": "너는 10년차 백엔드 엔지니어야. 기술 용어를 정확히 쓰고 3문장 이내로 답해. 한국어로 답해.",
    "초등학생 선생님": "너는 초등학생에게 설명하는 선생님이야. 비유를 써서 아주 쉽게 3문장 이내로 답해. 한국어로 답해.",
    "규칙 지키기": "너는 반드시 '예' 또는 '아니오' 다음에 한 문장만 덧붙여 답해야 해. 한국어로 답해.",
}

for name, system_prompt in roles.items():
    messages = [
        SystemMessage(content=system_prompt),  # 역할·규칙
        HumanMessage(content=question),        # 사용자 질문
    ]
    answer = llm.invoke(messages)
    print(f"=== [{name}] ===")
    print(answer.content)
    print()

# 생각해 볼 것:
#  - '규칙 지키기'는 질문과 형식이 잘 맞지 않는다. LLM이 규칙을 얼마나 잘 지켰나?
#  - 에이전트를 만들 때 시스템 프롬프트에 "도구를 이렇게 써라"를 적게 된다 (2단계).
