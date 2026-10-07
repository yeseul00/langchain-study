"""4-1. 벡터 저장소에 장애 기록을 넣고 검색하기.

실행: python 04_rag/step1_vector_search.py
"""

from knowledge import INCIDENTS, build_incident_store

print(f"장애 기록 {len(INCIDENTS)}건을 임베딩해서 저장하는 중...")
store = build_incident_store()
print("완료!\n")

query = "결제 화면이 너무 느리다는 문의가 쏟아지고 있어"

# 1) 의미 검색 — 가장 비슷한 3건
print(f"=== 검색: '{query}' ===")
for doc, score in store.similarity_search_with_score(query, k=3):
    m = doc.metadata
    print(f"  [{score:.3f}] #{m['incident_id']} {m['service']} ({m['region']}) → {doc.page_content[:50]}...")

# 2) 의미 검색 + 조건 필터 — 부산 리전 기록에서만 찾기
print("\n=== 같은 검색 + 필터(region == 'kr-busan') ===")
results = store.similarity_search_with_score(
    query, k=3, filter=lambda doc: doc.metadata["region"] == "kr-busan"
)
for doc, score in results:
    m = doc.metadata
    print(f"  [{score:.3f}] #{m['incident_id']} {m['service']} ({m['region']}) → {doc.page_content[:50]}...")

# 정리:
#  - 검색은 '질문을 임베딩 → 저장된 벡터와 유사도 비교 → 상위 k개'.
#  - 필터는 '의미'와 무관한 조건(서비스, 리전, 날짜)으로 후보를 좁힌다.
#    원본 프로젝트는 이 둘을 Oracle SQL 한 줄(WHERE + VECTOR_DISTANCE)로 처리한다.
