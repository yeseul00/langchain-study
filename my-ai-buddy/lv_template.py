"""🧩 나만의 레벨 템플릿 — 복사해서 새 레벨을 만드세요

사용법:
  1. 이 파일을 복사해서 새 이름으로 저장 (예: lv6_my_buddy.py)
  2. 아래 ✏️ 표시된 곳만 바꾸기
  3. 실행: python -m streamlit run my-ai-buddy/lv6_my_buddy.py

화면 규칙은 common.chat_area()가 알아서 지켜 줍니다.
  - 입력창은 화면 맨 아래에 고정
  - 대화는 최신순 (새 질문·답이 맨 위)
  - 질문 + 답 + 🔧도구 기록 + 👀보낸 내용이 박스 하나로 묶임
"""

import streamlit as st
from langchain.agents import create_agent

import buddy_config as cfg
from buddy_memory import build_store, make_search_tool, notes_signature
from buddy_tools import TOOLS
from common import chat_area, get_embeddings, get_llm, run_agent

# ✏️ 1) 레벨 번호와 제목 — 번호는 다른 레벨과 겹치지 않게 (대화 기록 저장 칸 이름으로 쓰임)
LEVEL = 6
TITLE = "나만의 동료"

# ✏️ 2) 이번 레벨에서 동료에게 줄 추가 지시
EXTRA_PROMPT = (
    "필요하면 도구와 메모 검색을 써. 도구 결과와 메모에 없는 사실은 지어내지 마."
)

# ✏️ 3) 기억(메모 검색) 도구를 쓸지
USE_MEMORY = True

st.set_page_config(page_title=f"Lv{LEVEL} · {cfg.NAME}", page_icon=cfg.EMOJI)
st.title(f"Lv{LEVEL} · {TITLE}")
st.caption(f"{cfg.EMOJI} {cfg.NAME}를 내 마음대로 키우는 레벨")

SYSTEM_PROMPT = cfg.build_system_prompt(extra=EXTRA_PROMPT)


@st.cache_resource(show_spinner="동료 준비 중...")
def get_agent(signature: tuple):
    tools = list(TOOLS)  # ✏️ 4) 도구를 고르고 싶으면 여기서 (예: [t for t in TOOLS if t.name != "roll_dice"])
    if USE_MEMORY:
        tools.append(make_search_tool(build_store(get_embeddings())))
    return create_agent(model=get_llm(0.3), tools=tools, system_prompt=SYSTEM_PROMPT)


agent = get_agent(notes_signature())

with st.sidebar:
    if st.button("🧹 대화 지우기"):
        st.session_state[f"lv{LEVEL}"] = []

# ── 채팅 (화면 규칙은 chat_area가 처리 — 이 패턴은 그대로 두세요) ──────────
turns = st.session_state.setdefault(f"lv{LEVEL}", [])
user_text, live = chat_area(LEVEL, turns, cfg.EMOJI, f"{cfg.NAME}에게 말 걸기")

if user_text:
    with live.container(border=True):  # 새 대화는 지난 대화들보다 위에 그린다
        with st.chat_message("user"):
            st.write(user_text)
        with st.spinner(f"{cfg.NAME}이(가) 생각하는 중..."):
            turn = run_agent(agent, SYSTEM_PROMPT, turns, user_text)
    turns.append(turn)
    st.rerun()
