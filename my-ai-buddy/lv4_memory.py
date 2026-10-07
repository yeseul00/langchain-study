"""Lv4 · 기억하는 동료 — RAG (내 메모를 찾아서 답하기)

실행: streamlit run my-ai-buddy/lv4_memory.py
"""

import streamlit as st
from langchain.agents import create_agent

import buddy_config as cfg
from buddy_memory import NOTES_DIR, build_store, load_chunks, make_search_tool, note_files, notes_signature
from common import chat_area, get_embeddings, get_llm, init_state, level_header, run_agent
from tutorials import LV4

st.set_page_config(page_title=f"Lv4 · {cfg.NAME}", page_icon=cfg.EMOJI)
init_state(lv4_probe="배포 언제 해?")

level_header(
    4, "기억하는 동료",
    f"{cfg.EMOJI} {cfg.NAME}이(가) my_notes 폴더의 메모를 읽고 기억해요. LLM이 원래 모르는 '우리 팀 이야기'에 답할 수 있게 됩니다.",
    missions=[
        "'금요일에 배포해도 돼?'라고 물어보고 어느 메모를 찾았는지 확인한다",
        "'국밥집은 몇 시까지 가야 해?'처럼 메모에 있는 세부 내용을 물어본다",
        "사이드바 '검색만 해 보기'로 LLM 없이 검색 결과만 먼저 본다",
        "메모에 없는 질문('회사 주차비 얼마야?')을 해서 지어내는지 확인한다",
        "my_notes 폴더에 나만의 메모 파일(.md)을 만들고 '🔄 메모 다시 읽기' 후 물어본다",
        "Lv2에서 저장한 메모장.md 내용을 물어본다",
    ],
    principle="""
**① RAG = 찾아서(Retrieval) → 붙여서(Augmented) → 답하기(Generation)**
LLM은 우리 팀 규칙을 **학습한 적이 없어요.** 질문할 때마다 관련 메모를 찾아서 건네줄 뿐입니다.

**② 넣기 (앱 시작 시 한 번, `buddy_memory.py`)**
```
메모 파일 → 빈 줄로 조각내기 → 조각마다 임베딩 → 벡터 저장소
```

**③ 찾기 (질문할 때마다)**
에이전트가 `search_my_notes` 도구를 불러서 → 질문을 임베딩 → Lv3의 유사도로 가까운 조각 3개 → 도구 결과로 받아서 답합니다.

**④ 메모가 바뀌면?** 다시 '넣기'만 하면 끝. 모델은 그대로예요.
""",
    tutorial=LV4,
)


@st.cache_resource(show_spinner="메모를 읽고 기억하는 중...")
def get_store(signature: tuple):
    return build_store(get_embeddings())


store = get_store(notes_signature())

SYSTEM_PROMPT = cfg.build_system_prompt(
    extra=("팀·회사·사용자에 대한 질문이나 예전에 기억해 달라고 한 것을 물으면 반드시 search_my_notes 로 먼저 찾아봐. "
           "찾은 메모에 없는 내용은 모른다고 말해. 답할 때 어느 메모 파일에서 봤는지 짧게 밝혀.")
)


@st.cache_resource
def get_agent(signature: tuple):
    return create_agent(model=get_llm(0.3), tools=[make_search_tool(store)], system_prompt=SYSTEM_PROMPT)


agent = get_agent(notes_signature())

# ── 사이드바: 기억 상태 + 검색만 해 보기 ─────────────────────
with st.sidebar:
    st.subheader("🧠 기억하고 있는 메모")
    chunks = load_chunks()
    for p in note_files():
        st.markdown(f"- `{p.name}` — 조각 {sum(c.metadata['source'] == p.name for c in chunks)}개")
    st.caption(f"폴더: {NOTES_DIR}")
    if st.button("🔄 메모 다시 읽기"):
        st.rerun()  # 파일 수정 시각이 바뀌었으면 저장소를 새로 만든다

    st.subheader("🔍 검색만 해 보기 (LLM 없이)")
    probe = st.text_input("검색어", key="lv4_probe")
    if probe:
        for doc, score in store.similarity_search_with_score(probe, k=3):
            st.markdown(f"**{score:.3f}** · `{doc.metadata['source']}`")
            st.caption(doc.page_content)

    if st.button("🧹 대화 지우기"):
        st.session_state.lv4 = []

# ── 채팅 ───────────────────────────────────────────────
turns = st.session_state.setdefault("lv4", [])
user_text, live = chat_area(4, turns, cfg.EMOJI, f"{cfg.NAME}에게 팀 이야기 물어보기")

if user_text:
    with live.container(border=True):
        with st.chat_message("user"):
            st.write(user_text)
        with st.spinner(f"{cfg.NAME}이(가) 메모를 뒤지는 중..."):
            turn = run_agent(agent, SYSTEM_PROMPT, turns, user_text)
    turns.append(turn)
    st.rerun()
