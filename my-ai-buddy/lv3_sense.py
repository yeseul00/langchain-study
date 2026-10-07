"""Lv3 · 눈치 있는 동료 — 임베딩과 의미 유사도

LLM 없이 임베딩 모델만 씁니다. 그래서 빠릅니다.
실행: streamlit run my-ai-buddy/lv3_sense.py
"""

import math

import pandas as pd
import streamlit as st

import buddy_config as cfg
from common import get_embeddings, init_state, level_header
from tutorials import LV3

st.set_page_config(page_title=f"Lv3 · {cfg.NAME}", page_icon=cfg.EMOJI)
init_state(lv3_query="사용자들이 돈을 못 내고 있어", lv3_feeling="아 오늘 배포만 세 번째 롤백이야")

level_header(
    3, "눈치 있는 동료",
    f"{cfg.EMOJI} {cfg.NAME}이(가) '말의 뜻'을 알아듣는 법을 배워요. 같은 단어가 없어도 비슷한 말을 찾아냅니다.",
    missions=[
        "[의미 놀이터] 기본 문장으로 검색해서 1등이 뭔지 본다",
        "[의미 놀이터] 겹치는 단어가 0개인데 유사도 1등인 문장을 만들어 낸다",
        "[의미 놀이터] 영어 문장을 후보에 넣어 한국어 질문과 비교해 본다",
        "[기분 눈치] 동료가 내 기분을 맞히게 해 본다",
        "[기분 눈치] 일부러 헷갈리는 문장(반어법 등)으로 동료를 속여 본다",
    ],
    principle="""
**① 임베딩 = 문장을 '의미 지도' 위의 좌표로 바꾸기**
`embed_query("배고파")` → 숫자 1024개(bge-m3 기준). 뜻이 비슷한 문장은 좌표도 가깝습니다.

**② 코사인 유사도 = 두 좌표가 같은 방향을 가리키는 정도**
1에 가까울수록 비슷. 이 파일의 `cosine()` 함수 4줄이 전부예요.

**③ 키워드 검색 vs 의미 검색**
표의 '겹치는 단어' 열과 '유사도' 열을 비교해 보세요.
"돈을 못 내" ↔ "결제 실패"처럼 단어가 하나도 안 겹쳐도 의미 검색은 찾아냅니다.

**④ 이건 학습이 아니다**
임베딩 모델은 이미 다 배운 상태로 **변환만** 합니다. Lv4에서 이 검색을 동료의 '기억'으로 씁니다.
""",
    tutorial=LV3,
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


emb = get_embeddings()
tab1, tab2 = st.tabs(["🔍 의미 놀이터", "😶 기분 눈치"])

# ── 탭 1: 의미 놀이터 ───────────────────────────────────
with tab1:
    query = st.text_input("검색할 문장", key="lv3_query")
    candidates_text = st.text_area(
        "후보 문장 (한 줄에 하나, 마음대로 고치세요)",
        "카드 결제 승인이 계속 실패한다\n"
        "PG사 연동에서 타임아웃이 난다\n"
        "Payment requests keep failing\n"
        "검색 결과 순서가 이상하다\n"
        "점심으로 김치찌개를 먹었다\n"
        "서버 디스크가 꽉 찼다",
        height=170,
    )
    candidates = [c.strip() for c in candidates_text.splitlines() if c.strip()]

    # 버튼을 누르거나, 튜토리얼 ▶ 해보기가 실행 플래그를 켜 두면 계산한다.
    if (st.button("유사도 계산", type="primary") or st.session_state.pop("lv3_run_sim", False)) and query and candidates:
        with st.spinner("문장을 좌표로 바꾸는 중..."):
            q_vec = emb.embed_query(query)
            c_vecs = emb.embed_documents(candidates)
        df = pd.DataFrame({
            "문장": candidates,
            "유사도": [round(cosine(q_vec, v), 3) for v in c_vecs],
            "겹치는 단어": [sum(w in c for w in query.split()) for c in candidates],
        }).sort_values("유사도", ascending=False)
        st.bar_chart(df, x="문장", y="유사도", horizontal=True)
        st.dataframe(df, hide_index=True)
        st.caption(f"벡터 크기: 숫자 {len(q_vec)}개 · 질문 벡터 앞 5개: {[round(v, 3) for v in q_vec[:5]]}")

# ── 탭 2: 기분 눈치 ─────────────────────────────────────
MOODS = {
    "😋 배고픔": ["배고파 죽겠다", "점심 언제 먹지", "뭐라도 먹고 싶어"],
    "😴 피곤함": ["너무 졸려", "어제 잠을 못 잤어", "눈이 감긴다"],
    "😤 짜증남": ["배포가 또 실패했어", "회의가 왜 이렇게 길어", "아 진짜 왜 안 돼"],
    "🥳 신남": ["드디어 버그 잡았다", "오늘 칼퇴다", "코드 리뷰 한 번에 통과했어"],
}
REACTIONS = {
    "😋 배고픔": "밥 먹으러 가자! 해바라기씨라도 나눠 줄까?",
    "😴 피곤함": "오늘은 무리하지 말자. 커피 한 잔 어때?",
    "😤 짜증남": "에고… 로그부터 같이 볼까? 화내면 버그가 더 숨어.",
    "🥳 신남": "와 최고다! 오늘은 축하하는 날~",
}

with tab2:
    st.markdown(f"{cfg.NAME}은(는) 아래 예시 문장들만 알고 있어요. LLM 없이 **임베딩 유사도만으로** 기분을 맞힙니다.")
    with st.expander("동료가 아는 예시 문장 보기"):
        st.json(MOODS)
    feeling = st.text_input("지금 하고 싶은 말", key="lv3_feeling")
    if (st.button("눈치 보기", type="primary") or st.session_state.pop("lv3_run_mood", False)) and feeling:
        with st.spinner("눈치 보는 중..."):
            f_vec = emb.embed_query(feeling)
            scores = {}
            for mood, examples in MOODS.items():
                ex_vecs = emb.embed_documents(examples)
                scores[mood] = max(cosine(f_vec, v) for v in ex_vecs)  # 가장 비슷한 예시 점수
        best = max(scores, key=scores.get)
        with st.chat_message("assistant", avatar=cfg.EMOJI):
            st.write(f"혹시 지금 **{best}** 상태야? {REACTIONS[best]}")
        st.bar_chart(pd.DataFrame({"기분": list(scores), "유사도": list(scores.values())}),
                     x="기분", y="유사도", horizontal=True)
