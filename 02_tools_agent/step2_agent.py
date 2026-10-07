"""2-2. create_agent로 에이전트 만들기.

step1에서 손으로 돌린 루프를 create_agent가 자동으로 돌린다.

실행: python 02_tools_agent/step2_agent.py
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools import TOOLS

SYSTEM_PROMPT = """\
너는 개발팀의 장애 대응 도우미야.
- 담당자나 온콜을 물으면 get_service_owner 도구로 확인해.
- 시간 정보가 필요하면 get_current_time 도구를 써.
- 도구 결과에 없는 내용은 지어내지 마.
- 항상 한국어로 짧게 답해.
"""

agent = create_agent(
    model=ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


def ask(question: str) -> None:
    print("=" * 60)
    print("질문:", question)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # 에이전트가 거친 과정을 순서대로 보여 준다.
    print("\n[에이전트 동작 기록]")
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  🔧 도구 요청: {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            print(f"  📦 도구 결과: {m.content}")

    print("\n[최종 답변]")
    print(result["messages"][-1].content)
    print()


ask("order-service 응답이 느려. 지금 몇 시인지랑, 누구한테 연락해야 하는지 알려줘.")
ask("search-service는 중요도 등급이 뭐야?")
ask("쿠버네티스 파드가 뭐야?")  # 도구가 필요 없는 질문 — 도구를 안 부르는지 확인
