"""4-2. 가장 단순한 RAG: 검색 → 프롬프트에 붙이기 → 답변.

에이전트 없이, '검색 결과를 프롬프트에 넣는다'는 RAG의 본질만 본다.

실행: python 04_rag/step2_simple_rag.py
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from knowledge import RUNBOOKS, build_incident_store

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
store = build_incident_store()

question = "주문 API가 배포하고 나서 에러가 많이 나. 어떻게 해야 해?"

# ── R: Retrieval (검색) ──────────────────────────────
hits = store.similarity_search(question, k=2)

# ── A: Augmentation (프롬프트 보강) ──────────────────
context_lines = []
for doc in hits:
    m = doc.metadata
    rb = RUNBOOKS[m["runbook_id"]]
    context_lines.append(
        f"[장애 #{m['incident_id']}] {doc.page_content}\n"
        f"  └ 사용한 런북 {rb['runbook_id']} '{rb['title']}': " + " / ".join(rb["steps"])
    )
context = "\n".join(context_lines)

prompt = [
    SystemMessage(content=(
        "너는 장애 대응 도우미야. 아래 [참고 자료]에 있는 내용만 근거로 한국어로 답해. "
        "답에 참고한 장애 번호(#)와 런북 번호를 꼭 적어. 자료에 없으면 모른다고 해."
    )),
    HumanMessage(content=f"[참고 자료]\n{context}\n\n[질문]\n{question}"),
]

print("=== LLM에게 실제로 보내는 글 ===")
print(prompt[1].content)

# ── G: Generation (답변 생성) ────────────────────────
answer = llm.invoke(prompt)
print("\n=== 답변 ===")
print(answer.content)

# 정리:
#  LLM은 우리 회사 장애 기록을 '학습'한 적이 없다. 질문할 때마다 관련 기록을 찾아서
#  프롬프트에 붙여 줬을 뿐이다. 이게 RAG(Retrieval-Augmented Generation)의 전부다.
#  한계: 무엇을, 몇 번 검색할지 코드가 고정해 두었다 → 다음 단계에서 에이전트에게 맡긴다.
