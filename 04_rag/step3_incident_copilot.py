"""4-3. 미니 Incident Copilot — RAG 검색을 도구로 쥔 에이전트.

2단계(에이전트) + 4-2(RAG)를 합친 것. 원본 프로젝트의 축소판이다.
에이전트가 스스로 판단해서 '비슷한 장애 검색 → 런북 조회 → 담당자 조회'를 한다.

실행: python 04_rag/step3_incident_copilot.py
     (질문을 입력하고 Enter, 종료는 빈 줄 Enter)
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from knowledge import REGIONS, RUNBOOKS, SERVICES, build_incident_store

store = build_incident_store()


# ── 도구 1: 비슷한 과거 장애 검색 (RAG의 R) ──────────────
@tool
def search_similar_incidents(query: str, service: str = "", region: str = "") -> list[dict]:
    """현재 증상과 의미가 비슷한 과거 장애 기록을 최대 3건 찾는다.

    Args:
        query: 지금 겪는 증상을 설명하는 문장 (사용자 표현 그대로).
        service: 특정 서비스로 좁힐 때만 지정. payment-service, order-service,
            search-service, notification-service 중 하나. 모르면 빈 문자열.
        region: 특정 리전으로 좁힐 때만 지정. kr-seoul 또는 kr-busan. 모르면 빈 문자열.
    """
    # LLM이 넘긴 값은 허용 목록으로 검사한다 (엉뚱한 값이면 필터 무시).
    service = service if service in SERVICES else ""
    region = region if region in REGIONS else ""

    def match(doc):
        m = doc.metadata
        return (not service or m["service"] == service) and (not region or m["region"] == region)

    hits = store.similarity_search_with_score(query, k=3, filter=match)
    return [{**doc.metadata, "content": doc.page_content, "similarity": round(score, 3)} for doc, score in hits]


# ── 도구 2: 런북 조회 ─────────────────────────────────
@tool
def get_runbook(runbook_id: str) -> dict:
    """런북 번호(예: RB-01)로 대응 절차를 조회한다. 장애 기록의 runbook_id 값을 넣는다."""
    return RUNBOOKS.get(runbook_id, {"error": f"{runbook_id} 런북 없음", "known": list(RUNBOOKS)})


# ── 도구 3: 담당자 조회 ───────────────────────────────
@tool
def get_service_owner(service_name: str) -> dict:
    """서비스 이름으로 담당 팀, 온콜 담당자, 중요도 등급을 조회한다."""
    info = SERVICES.get(service_name)
    return {"service": service_name, **info} if info else {"error": "알 수 없는 서비스", "known": list(SERVICES)}


SYSTEM_PROMPT = """\
너는 개발팀의 장애 대응 도우미야. 일반 지식이 아니라 팀의 과거 장애 기록과 런북을 근거로 답해.

도구 사용 순서:
1. search_similar_incidents 로 비슷한 과거 장애를 찾는다. 사용자가 서비스나 리전을 말했으면 필터로 넣는다.
2. 가장 비슷한 장애의 runbook_id 로 get_runbook 을 호출한다.
3. 해당 서비스의 get_service_owner 를 호출해 연락할 사람을 찾는다.

답변 형식 (한국어):
- 가장 유력한 원인 1문장 (근거 장애 번호 #표기)
- 런북의 다음 조치 2~4개 (런북 번호 표기)
- 연락할 팀과 온콜 담당자

도구 결과에 없는 장애나 절차를 지어내지 마.
"""

agent = create_agent(
    model=ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0),
    tools=[search_similar_incidents, get_runbook, get_service_owner],
    system_prompt=SYSTEM_PROMPT,
)


def diagnose(question: str) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("\n[에이전트 동작 기록]")
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  🔧 {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            print(f"  📦 {m.content[:120]}{'...' if len(m.content) > 120 else ''}")
    print("\n[답변]")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    print("미니 Incident Copilot  (빈 줄 Enter = 종료)")
    print("예) 결제 API가 갑자기 느려졌어. CPU는 멀쩡해.")
    print("예) 부산 리전에서 알림이 중복으로 가고 있어\n")
    while True:
        q = input("🚨 알림/증상> ").strip()
        if not q:
            break
        diagnose(q)
        print()
