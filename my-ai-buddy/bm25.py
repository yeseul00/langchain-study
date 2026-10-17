"""🔤 키워드 검색(BM25)과 순위 합치기(RRF) — Lv6 청킹 실험실이 사용

임베딩(의미 검색)은 "뜻이 비슷한 글"을 잘 찾지만, `4.3.4`·`제31조의18`·`access_token`처럼
글자 그대로 맞아야 하는 번호·이름에는 약해요. 키워드 검색은 그 반대예요.
둘의 순위를 합치는 것이 하이브리드 검색입니다. (sproute-lex 문서 09의 계획: FTS5 + RRF)

LLM도 임베딩도 쓰지 않는 순수 계산이라, 숫자를 직접 따라가 볼 수 있어요.
"""

import math
import re
from collections import Counter

# 번호(4.3.4), 영문·코드(access_token), 숫자, 한글 낱말을 각각 하나의 토큰으로 자른다.
TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|[A-Za-z_][A-Za-z0-9_\-]*|\d+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """글 → 토큰 목록.

    한국어는 '처리상태를'·'처리상태' 처럼 조사가 붙어 낱말이 달라지므로,
    한글 낱말은 두 글자씩 겹쳐 자른 조각(바이그램)도 함께 넣는다: 처리상태 → 처리·리상·상태
    """
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        out.append(tok)
        if re.fullmatch(r"[가-힣]{3,}", tok):
            out += [tok[i:i + 2] for i in range(len(tok) - 1)]
    return out


class BM25:
    """가장 널리 쓰이는 키워드 점수. 질문 단어가 조각에 많이 나올수록, 그 단어가 드물수록 점수가 높다."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(t.values()) for t in self.toks]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        n = len(docs)
        df = Counter(w for t in self.toks for w in t)
        # 드문 단어일수록 큰 가중치 (모든 조각에 나오는 단어는 거의 0)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.toks, self.lens):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def ranks(scores: list[float]) -> list[int]:
    """점수 → 등수(1등부터). 점수가 0인 조각은 '못 찾음'으로 보고 등수를 주지 않는다(None)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    r = [None] * len(scores)
    for place, i in enumerate(order, 1):
        if scores[i] > 0:
            r[i] = place
    return r


def rrf(*rank_lists: list, k: int = 60) -> list[float]:
    """RRF(Reciprocal Rank Fusion): 점수 대신 '등수'만으로 합친다.

    점수 = Σ 1 / (k + 등수).  k=60은 관례값. 한쪽에서만 1등이어도 충분히 위로 올라온다.
    임베딩 점수(0~1)와 BM25 점수(0~수십)는 단위가 달라 그냥 더할 수 없어서 등수를 쓴다.
    """
    n = len(rank_lists[0])
    return [sum(1 / (k + r[i]) for r in rank_lists if r[i] is not None) for i in range(n)]
