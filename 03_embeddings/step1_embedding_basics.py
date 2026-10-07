"""3-1. 문장을 벡터로 바꾸고 유사도 비교하기.

실행: python 03_embeddings/step1_embedding_basics.py
"""

import math
import os

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터가 같은 방향을 가리키는 정도 (1에 가까울수록 비슷)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# 1) 문장 하나를 벡터로 바꿔 보기
vector = embeddings.embed_query("결제가 자꾸 실패해")
print("=== 벡터 모양 ===")
print(f"숫자 개수(차원): {len(vector)}")
print(f"앞 5개 숫자: {[round(v, 4) for v in vector[:5]]}")

# 2) 기준 문장과 여러 문장의 유사도 비교
query = "결제가 자꾸 실패해"
sentences = [
    "카드 승인 오류율이 갑자기 올라갔다",
    "PG사 연동 API에서 타임아웃이 계속 난다",
    "Payment requests are failing repeatedly",  # 영어, 같은 뜻
    "검색 결과 순서가 이상하게 나온다",
    "점심 메뉴로 김치찌개를 먹었다",
]

query_vec = embeddings.embed_query(query)
doc_vecs = embeddings.embed_documents(sentences)  # 여러 문장을 한 번에 변환

scored = sorted(
    ((cosine_similarity(query_vec, v), s) for v, s in zip(doc_vecs, sentences)),
    reverse=True,
)

print(f"\n=== '{query}' 와(과) 비슷한 순서 ===")
for score, sentence in scored:
    bar = "█" * int(max(score, 0) * 30)
    print(f"{score:.3f} {bar:<30} {sentence}")

# 생각해 볼 것:
#  - '결제'라는 단어가 없는 문장도 위에 올라왔나?
#  - 전혀 상관없는 문장(김치찌개)의 점수는 0이 아닐 수 있다. 절대값보다 '순위'가 중요하다.
