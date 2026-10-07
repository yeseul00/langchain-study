"""4단계 공통: 장애 기록을 벡터 저장소에 넣고(Ingestion), 조회 함수를 제공한다.

- 벡터 저장소: LangChain 기본 제공 InMemoryVectorStore (설치 필요 없음, 메모리에만 저장)
- 원본 프로젝트는 같은 역할을 Oracle 26ai DB가 한다.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).parent / "data"

INCIDENTS = json.loads((DATA_DIR / "incidents.json").read_text(encoding="utf-8"))
RUNBOOKS = {rb["runbook_id"]: rb for rb in json.loads((DATA_DIR / "runbooks.json").read_text(encoding="utf-8"))}

SERVICES = {
    "payment-service": {"team": "결제팀", "oncall": "@kim-payments", "tier": "critical"},
    "order-service": {"team": "주문팀", "oncall": "@lee-orders", "tier": "critical"},
    "search-service": {"team": "검색팀", "oncall": "@park-search", "tier": "high"},
    "notification-service": {"team": "플랫폼팀", "oncall": "@choi-platform", "tier": "medium"},
}
REGIONS = {"kr-seoul", "kr-busan"}


def get_embeddings():
    return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"))


@lru_cache(maxsize=1)
def build_incident_store() -> InMemoryVectorStore:
    """장애 기록 → Document → 임베딩 → 벡터 저장소 (프로그램 실행 시 한 번만)."""
    docs = [
        Document(
            # page_content: 임베딩(의미 좌표)으로 바뀔 본문
            page_content=f"{inc['summary']} 원인: {inc['cause']}",
            # metadata: 필터링·표시용 정보 (임베딩되지 않음)
            metadata={k: inc[k] for k in ("incident_id", "service", "region", "severity", "occurred_at", "runbook_id")},
        )
        for inc in INCIDENTS
    ]
    store = InMemoryVectorStore(embedding=get_embeddings())
    store.add_documents(docs)
    return store
