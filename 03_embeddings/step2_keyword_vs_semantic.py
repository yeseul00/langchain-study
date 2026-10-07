"""3-2. 키워드 검색 vs 의미 검색.

같은 장애 기록에서 키워드 검색과 의미(벡터) 검색이 무엇을 찾는지 비교한다.

실행: python 03_embeddings/step2_keyword_vs_semantic.py
"""

import math
import os

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"))

# 과거 장애 기록 (실제로는 수백~수천 건)
past_incidents = [
    "주문 API p99 응답시간이 8초까지 치솟음. 원인은 DB 커넥션 풀 고갈",
    "카드 승인 단계에서 PG사 타임아웃 다발, 결제 완료율 급락",
    "검색 인덱스 재색인 중 검색 결과 누락",
    "배포 직후 로그인 토큰 검증 실패로 401 오류 급증",
    "푸시 알림 큐 적체로 알림이 30분 지연 발송",
]

query = "사용자들이 돈을 못 내고 있어"  # 새로 들어온 알림 (기록과 같은 단어가 거의 없다)


def keyword_score(q: str, doc: str) -> int:
    """질문의 단어가 문서에 몇 개 들어 있는지 센다 (단순 키워드 검색)."""
    return sum(1 for word in q.split() if word in doc)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


print(f"질문: {query}\n")

print("=== 키워드 검색 결과 ===")
for doc in sorted(past_incidents, key=lambda d: keyword_score(query, d), reverse=True)[:3]:
    print(f"  겹치는 단어 {keyword_score(query, doc)}개 | {doc}")

print("\n=== 의미(벡터) 검색 결과 ===")
q_vec = embeddings.embed_query(query)
d_vecs = embeddings.embed_documents(past_incidents)
ranked = sorted(zip(d_vecs, past_incidents), key=lambda p: cosine(q_vec, p[0]), reverse=True)
for vec, doc in ranked[:3]:
    print(f"  유사도 {cosine(q_vec, vec):.3f} | {doc}")

# 정리:
#  키워드 검색은 같은 단어가 없으면 아무것도 못 찾는다(모두 0개).
#  의미 검색은 '돈을 못 낸다' ≈ '결제 완료율 급락'을 연결한다.
#  4단계에서는 이 검색을 에이전트의 도구로 만든다.
