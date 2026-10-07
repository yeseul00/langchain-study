"""Lv1 · 말하는 동료 — LLM 호출, 시스템 프롬프트, 대화 기억

실행: streamlit run my-ai-buddy/lv1_talk.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

import buddy_config as cfg
from common import chat_area, get_llm, history_messages, init_state, level_header, show_sent
from tutorials import LV1

st.set_page_config(page_title=f"Lv1 · {cfg.NAME}", page_icon=cfg.EMOJI)
init_state(lv1_name=cfg.NAME, lv1_avatar=cfg.EMOJI, lv1_role=cfg.ROLE, lv1_personality=cfg.PERSONALITY,
           lv1_speech=cfg.SPEECH, lv1_remember=True, lv1_temp=0.7)

level_header(
    1, "말하는 동료",
    f"새 AI 동료 {cfg.EMOJI} {cfg.NAME}이(가) 입사했어요. 아직 말만 할 줄 알아요. 성격을 만들어 주세요!",
    missions=[
        "동료에게 자기소개를 시켜 본다",
        "사이드바에서 성격·말투를 바꾸고 같은 질문을 다시 해 본다",
        "'👀 LLM에게 실제로 보낸 내용'을 펼쳐서 시스템 프롬프트를 찾아본다",
        "내 이름을 알려준 뒤 '기억하기'를 끄고 '내 이름 뭐야?'라고 물어본다",
        "창의성(temperature)을 0과 1.5로 바꿔 같은 질문을 두 번씩 해 본다",
        "마음에 드는 성격을 buddy_config.py 에 저장한다",
    ],
    principle="""
**① 시스템 프롬프트 = 동료의 업무 지침서**
사이드바 설정이 `build_system_prompt()`로 한 덩어리 글이 되어
매 질문 앞에 `SystemMessage`로 붙어서 갑니다. 성격은 '학습'이 아니라 **매번 읽혀 주는 글**이에요.

**② LLM은 기억이 없다**
'기억하기'가 켜져 있으면 지난 대화를 **통째로 다시 보냅니다** (`history_messages`).
꺼 보면 보내는 메시지 수가 확 줄고, 동료가 방금 한 얘기도 모릅니다.

**③ 핵심 코드 (이 파일 아래쪽)**
```python
sent = [SystemMessage(시스템프롬프트), *지난대화, HumanMessage(내 말)]
answer = llm.stream(sent)   # = llm.invoke(sent) 를 글자 단위로 받기
```
""",
    tutorial=LV1,
)

# ── 사이드바: 동료 꾸미기 + 실험 스위치 ────────────────────
with st.sidebar:
    st.subheader("🎨 동료 꾸미기 (임시)")
    name = st.text_input("이름", key="lv1_name")
    avatar = st.text_input("이모지", key="lv1_avatar")
    role = st.text_input("역할", key="lv1_role")
    personality = st.text_area("성격", key="lv1_personality")
    speech = st.text_input("말투", key="lv1_speech")

    st.subheader("🧪 실험 스위치")
    remember = st.toggle("이전 대화 기억하기", key="lv1_remember")
    temperature = st.slider("창의성 (temperature)", 0.0, 1.5, step=0.1, key="lv1_temp",
                            help="0이면 매번 비슷한 답, 높을수록 엉뚱하고 다양한 답")
    if st.button("🧹 대화 지우기"):
        st.session_state.lv1 = []

    system_prompt = cfg.build_system_prompt(name=name, role=role, personality=personality, speech=speech)
    with st.expander("⚙️ 지금 시스템 프롬프트"):
        st.code(system_prompt, language=None, wrap_lines=True)

# ── 채팅 ───────────────────────────────────────────────
turns = st.session_state.setdefault("lv1", [])
user_text, live = chat_area(1, turns, avatar, f"{name}에게 말 걸기")

if user_text:
    # ★ 학습 포인트: LLM에게 보낼 메시지 목록을 직접 조립한다.
    sent = [SystemMessage(content=system_prompt)]
    if remember:
        sent += history_messages(turns)          # 지난 대화를 다시 붙여 보낸다
    sent.append(HumanMessage(content=user_text))  # 이번에 한 말

    with live.container(border=True):  # 새 대화는 입력창 바로 아래에 그린다
        with st.chat_message("user"):
            st.write(user_text)
        with st.chat_message("assistant", avatar=avatar):
            answer = st.write_stream(chunk.content for chunk in get_llm(temperature).stream(sent))
            show_sent(sent)

    turns.append({"user": user_text, "ai": answer, "sent": sent})
