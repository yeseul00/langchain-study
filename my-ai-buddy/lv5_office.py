"""Lv5 · 출근하는 동료 — 성격 + 도구 + 기억을 모두 합친 완성형 에이전트

실행: streamlit run my-ai-buddy/lv5_office.py
"""

from collections import Counter

import streamlit as st
from langchain.agents import create_agent

import buddy_config as cfg
from buddy_memory import build_store, make_search_tool, notes_signature
from buddy_tools import TOOLS
from common import chat_area, get_embeddings, get_llm, level_header, run_agent
from tutorials import LV5

st.set_page_config(page_title=f"Lv5 · {cfg.NAME}", page_icon=cfg.EMOJI, layout="wide")

level_header(
    5, "출근하는 동료",
    f"{cfg.EMOJI} {cfg.NAME}의 첫 출근날! 성격(Lv1) + 도구(Lv2) + 기억(Lv4)을 모두 갖춘 ai office 1호 직원입니다.",
    missions=[
        "한 질문에 도구를 2개 이상 쓰게 만든다 (예: '오늘 무슨 요일이야? 배포해도 되는 날이야?')",
        "'점심 추천해 주고, 그 메뉴 파는 맛집 있으면 알려줘'처럼 도구 + 기억을 섞어 쓰게 한다",
        "'내일 10시에 회고 있어, 기억해 줘' → 대화 지우기 → '내일 일정 뭐였지?' 로 메모가 남아 있는지 확인한다",
        "오른쪽 근무 일지에서 어떤 도구를 가장 많이 썼는지 본다",
        "동료가 틀린 답을 한 경우를 찾고, 👀 보낸 내용을 보며 왜 틀렸는지 추리한다",
        "다음 단계 구상: 이 동료를 Slack에 출근시키려면 무엇이 필요할지 Claude와 이야기해 본다",
    ],
    principle="""
**지금까지 만든 것이 전부 이 한 줄에 들어 있어요**
```python
create_agent(
    model=get_llm(),                          # Lv1 말하기
    tools=TOOLS + [search_my_notes],          # Lv2 손 + Lv4 기억(Lv3 임베딩 사용)
    system_prompt=build_system_prompt(...),   # Lv1 성격
)
```

**원본 Incident Copilot과 비교하면**
도구가 '장애 검색·런북·담당자'일 뿐 구조는 똑같습니다.
원본은 여기에 진짜 DB(Oracle), API 서버(FastAPI)가 붙어 있어요.

**ai office로 가려면 (다음 이야기)**
- 화면 대신 **Slack 메시지**로 질문을 받고 답을 보내기
- 동료를 여러 명 만들고 (기획 동료, 코딩 동료…) 역할 나누기
- 동료끼리 서로 일을 넘기기 → 멀티 에이전트
""",
    tutorial=LV5,
)


@st.cache_resource(show_spinner="출근 준비 중... (메모 읽는 중)")
def get_agent(signature: tuple):
    store = build_store(get_embeddings())
    tools = TOOLS + [make_search_tool(store)]
    return create_agent(model=get_llm(0.3), tools=tools, system_prompt=SYSTEM_PROMPT)


SYSTEM_PROMPT = cfg.build_system_prompt(
    extra=("너는 도구와 메모를 가진 팀 동료야. 필요한 만큼 도구를 여러 번 써도 돼.\n"
           "- 시간·요일은 get_now, 계산은 calculate, 메뉴는 pick_lunch, 기억해 달라는 건 save_memo.\n"
           "- 팀 규칙·맛집·서버·예전 메모에 관한 건 search_my_notes 로 먼저 찾아봐.\n"
           "- 도구 결과와 메모에 없는 사실은 지어내지 마.")
)

agent = get_agent(notes_signature())
turns = st.session_state.setdefault("lv5", [])

chat_col, log_col = st.columns([3, 1])

# ── 근무 일지 ───────────────────────────────────────────
with log_col:
    st.subheader("📋 근무 일지")
    usage = Counter(t["tool"] for turn in turns for t in turn["trace"])
    st.metric("받은 질문", len(turns))
    st.metric("도구 사용", sum(usage.values()))
    if usage:
        st.bar_chart(dict(usage), horizontal=True)
    if st.button("🧹 대화 지우기"):
        st.session_state.lv5 = []
        st.rerun()

# ── 채팅 ───────────────────────────────────────────────
user_text, live = chat_area(5, turns, cfg.EMOJI, f"{cfg.NAME}에게 업무 요청하기", where=chat_col)

if user_text:
    with live.container(border=True):
        with st.chat_message("user"):
            st.write(user_text)
        with st.spinner(f"{cfg.NAME}이(가) 일하는 중..."):
            turn = run_agent(agent, SYSTEM_PROMPT, turns, user_text)
    turns.append(turn)
    st.rerun()
