"""Lv2 · 손이 생긴 동료 — 도구(Tool)와 에이전트

실행: streamlit run my-ai-buddy/lv2_hands.py
"""

import streamlit as st
from langchain.agents import create_agent

import buddy_config as cfg
from buddy_tools import TOOLS
from common import chat_area, get_llm, init_state, level_header, run_agent
from tutorials import LV2

st.set_page_config(page_title=f"Lv2 · {cfg.NAME}", page_icon=cfg.EMOJI)
init_state(**{f"lv2_tool_{t.name}": True for t in TOOLS})

level_header(
    2, "손이 생긴 동료",
    f"{cfg.EMOJI} {cfg.NAME}에게 도구가 생겼어요! 이제 시간을 보고, 계산하고, 메모도 할 수 있어요.",
    missions=[
        "'지금 몇 시야?'라고 물어보고 🔧 도구 사용 기록을 확인한다",
        "'점심 뭐 먹지? 중식으로'라고 물어본다 — 도구에 넘긴 인자(args)를 확인한다",
        "'3847 곱하기 291은?'을 계산기 도구를 켠 채로, 끈 채로 각각 물어본다 (답이 맞나?)",
        "'다음 주 회의는 수요일이야, 기억해 줘'라고 해서 my_notes/메모장.md 파일이 생기는지 본다",
        "도구가 필요 없는 질문('오늘 기분 어때?')에는 도구를 안 쓰는지 확인한다",
        "도전: buddy_tools.py 에 나만의 도구를 하나 만들어 TOOLS에 추가한다",
    ],
    principle="""
**① LLM은 함수를 실행하지 못한다 — '요청'만 한다**
LLM의 답이 글 대신 `get_now()를 불러줘` 같은 **도구 요청**으로 옵니다.
실제 실행은 파이썬이 하고, 결과를 `ToolMessage`로 다시 LLM에게 보냅니다.

**② 에이전트 = 이 반복을 자동으로**
```
생각 → 도구 요청 → 파이썬 실행 → 결과 확인 → (또 필요하면 반복) → 최종 답
```
`create_agent(model, tools, system_prompt)` 한 줄이 이 반복을 돌려 줍니다.

**③ LLM은 무엇을 보고 도구를 고를까?**
함수 코드가 아니라 **이름 + docstring + 인자**만 봅니다.
`buddy_tools.py`의 설명(docstring)을 엉터리로 바꾸면 동료가 도구를 못 쓰게 돼요.

**④ 👀 LLM에게 보낸 내용을 펼쳐 보면**
🤖 ai(도구 요청) → 📦 tool(결과) → 🤖 ai(최종 답) 순서가 보입니다.
""",
    tutorial=LV2,
)

# ── 사이드바: 도구 켜고 끄기 ────────────────────────────
with st.sidebar:
    st.subheader("🔧 장착할 도구")
    enabled = [t for t in TOOLS if st.toggle(t.name, key=f"lv2_tool_{t.name}", help=t.description.split("\n")[0])]
    st.caption(f"{len(enabled)}개 장착됨")
    if st.button("🧹 대화 지우기"):
        st.session_state.lv2 = []

SYSTEM_PROMPT = cfg.build_system_prompt(
    extra="도구가 도움이 되면 도구를 사용해. 도구 결과에 없는 사실은 지어내지 마."
)


@st.cache_resource
def get_agent(tool_names: tuple):
    # ★ 학습 포인트: 모델 + 도구 + 시스템 프롬프트 = 에이전트
    tools = [t for t in TOOLS if t.name in tool_names]
    return create_agent(model=get_llm(0.3), tools=tools, system_prompt=SYSTEM_PROMPT)


agent = get_agent(tuple(t.name for t in enabled))

# ── 채팅 ───────────────────────────────────────────────
turns = st.session_state.setdefault("lv2", [])
user_text, live = chat_area(2, turns, cfg.EMOJI, f"{cfg.NAME}에게 일 시키기")

if user_text:
    with live.container(border=True):
        with st.chat_message("user"):
            st.write(user_text)
        with st.spinner(f"{cfg.NAME}이(가) 생각하고 도구를 쓰는 중..."):
            turn = run_agent(agent, SYSTEM_PROMPT, turns, user_text)
    turns.append(turn)
    st.rerun()
