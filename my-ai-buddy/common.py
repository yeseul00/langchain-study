"""모든 레벨이 같이 쓰는 화면·실행 도우미.

학습 포인트는 각 lvN_*.py 파일에 있습니다. 이 파일은 '화면 꾸미기'용이라 안 읽어도 됩니다.
"""

import json
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")


# ── 모델 ─────────────────────────────────────────────
@st.cache_resource
def get_llm(temperature: float = 0.7) -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


@st.cache_resource
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


# ── 레벨 머리말: 제목 + 튜토리얼 + 도전 과제 + 원리 설명 ──────────
def level_header(level: int, title: str, story: str, missions: list[str], principle: str,
                 tutorial: list[dict] | None = None) -> None:
    st.title(f"Lv{level} · {title}")
    st.caption(story)
    if tutorial:
        run_tutorial(level, tutorial)
    col1, col2 = st.columns([3, 2])
    with col1:
        with st.expander("🏆 도전 과제 (튜토리얼 다 하고 혼자 해 보기)"):
            done = sum(st.checkbox(m, key=f"lv{level}_m{i}") for i, m in enumerate(missions))
            st.progress(done / len(missions), text=f"{done} / {len(missions)} 완료")
            if done == len(missions):
                st.success("레벨 클리어! 🎉 다음 레벨로 가 보세요.")
    with col2:
        with st.expander("💡 원리 요약: 코드에서 무슨 일이?"):
            st.markdown(principle)
    st.divider()


def run_tutorial(level: int, steps: list[dict]) -> None:
    """단계별 튜토리얼 패널.

    step 항목:
      title  단계 제목            do     할 일 설명
      prompt 대신 보내 줄 문장     set    바꿔 줄 화면 설정 {위젯 key: 값}
      run    켜 줄 실행 플래그 목록 clear  True면 대화 지우기
      look   확인할 곳            why    원리 설명 (펼쳐 보기)
    """
    key = f"lv{level}_step"
    i = min(st.session_state.setdefault(key, 0), len(steps) - 1)
    step = steps[i]

    with st.container(border=True):
        st.markdown(f"##### 📖 튜토리얼 {i + 1} / {len(steps)} · {step['title']}")
        st.progress((i + 1) / len(steps))
        st.markdown(step["do"])
        if step.get("prompt"):
            st.info(f"💬 보낼 말: **{step['prompt']}**")

        prev_col, act_col, next_col = st.columns([1, 2, 1])
        if prev_col.button("◀ 이전", disabled=i == 0, key=f"{key}_prev"):
            st.session_state[key] = i - 1
            st.rerun()
        has_action = any(step.get(k) for k in ("prompt", "set", "run", "clear"))
        if has_action and act_col.button(step.get("button", "▶ 해보기"), type="primary",
                                         key=f"{key}_do_{i}"):
            for k, v in step.get("set", {}).items():
                st.session_state[k] = v
            if step.get("clear"):
                st.session_state[f"lv{level}"] = []
            for flag in step.get("run", []):
                st.session_state[flag] = True
            if step.get("prompt"):
                st.session_state[f"lv{level}_pending"] = step["prompt"]
        last = i == len(steps) - 1
        if last:
            # 마지막 단계: '완료'를 누르면 튜토리얼 완료로 표시하고 다음 레벨 안내를 보여 준다.
            if next_col.button("완료 🎉", type="primary", key=f"{key}_done"):
                st.session_state[f"lv{level}_tutorial_done"] = True
                st.balloons()
        elif next_col.button("다음 ▶", key=f"{key}_next"):
            st.session_state[key] = i + 1
            st.rerun()

        if step.get("look"):
            st.markdown(f"**🔎 확인할 곳** — {step['look']}")
        if step.get("why"):
            with st.expander("🤔 왜 그럴까? (해 본 다음 펼쳐 보세요)"):
                st.markdown(step["why"])
        if last and st.session_state.get(f"lv{level}_tutorial_done"):
            show_next_level(level)


NEXT_LEVEL = {1: "lv2_hands.py", 2: "lv3_sense.py", 3: "lv4_memory.py", 4: "lv5_office.py", 5: "lv6_chunk_lab.py"}

# app.py로 실행하면 레벨 페이지 목록이 여기 채워진다 (파일 이름 → 페이지).
# 레벨 파일을 하나씩 따로 실행했다면 비어 있다.
PAGES: dict = {}


def show_next_level(level: int) -> None:
    """튜토리얼을 끝냈을 때 다음 레벨로 가는 버튼(또는 명령어)을 보여 준다."""
    nxt = NEXT_LEVEL.get(level)
    if not nxt:
        st.success("모든 레벨 튜토리얼 완료! 🎉 이제 나만의 레벨을 만들어 보세요.")
        st.markdown("`lv_template.py`를 복사해 `lv7_이름.py`로 저장하면 레벨 목록에 자동으로 추가돼요.")
        st.code("cp my-ai-buddy/lv_template.py my-ai-buddy/lv7_my_buddy.py", language="bash")
        return

    st.success(f"Lv{level} 튜토리얼 완료! 🎉 아래 🏆 도전 과제를 더 해 보거나 Lv{level + 1}로 넘어가세요.")
    if nxt in PAGES:
        if st.button(f"▶ Lv{level + 1}로 이동", type="primary", key=f"lv{level}_goto_next"):
            st.switch_page(PAGES[nxt])
    else:
        st.markdown("이 레벨만 따로 실행 중이에요. 모든 레벨을 한 화면에서 쓰려면 `Ctrl+C`로 끄고:")
        st.code("python -m streamlit run my-ai-buddy/app.py", language="bash")


def pending_prompt(level: int) -> str | None:
    """튜토리얼의 ▶ 해보기 버튼이 대신 보내 달라고 맡겨 둔 문장."""
    return st.session_state.pop(f"lv{level}_pending", None)


def init_state(**defaults) -> None:
    """위젯 기본값을 세션에 한 번만 넣는다 (튜토리얼이 값을 바꿀 수 있도록 key로 관리)."""
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ── 메시지 보여주기 ──────────────────────────────────────
_ROLE_LABEL = {SystemMessage: "⚙️ system (역할 지시)", HumanMessage: "🙋 human (사용자)",
               AIMessage: "🤖 ai (LLM)", ToolMessage: "📦 tool (도구 결과)"}


def _fmt(msg) -> str:
    label = _ROLE_LABEL.get(type(msg), type(msg).__name__)
    body = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
    if isinstance(msg, AIMessage) and msg.tool_calls:
        calls = ", ".join(f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})" for tc in msg.tool_calls)
        body = (body + "\n" if body else "") + f"→ 도구 요청: {calls}"
    return f"[{label}]\n{body}"


def show_sent(messages: list, title: str = "👀 LLM에게 실제로 보낸 내용") -> None:
    """LLM이 이번 답을 만들 때 받은 메시지 전체를 펼쳐 보여 준다."""
    with st.expander(f"{title} — 메시지 {len(messages)}개"):
        st.code("\n\n".join(_fmt(m) for m in messages), language=None, wrap_lines=True)


def show_trace(trace: list[dict]) -> None:
    """에이전트가 사용한 도구 기록."""
    if not trace:
        st.caption("🔧 도구 사용 없음 — 스스로 답했어요")
        return
    with st.expander(f"🔧 도구 사용 기록 — {len(trace)}번", expanded=True):
        for i, t in enumerate(trace, 1):
            st.markdown(f"**{i}. `{t['tool']}`** ← `{json.dumps(t['args'], ensure_ascii=False)}`")
            st.code(t["result"][:600], language=None, wrap_lines=True)


def render_turn(turn: dict, avatar: str) -> None:
    """저장된 대화 한 턴(사용자 말 + 동료 답 + 뒷이야기)을 테두리 박스 하나로 그린다."""
    with st.container(border=True):
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant", avatar=avatar):
            st.write(turn["ai"])
            if "trace" in turn:
                show_trace(turn["trace"])
            show_sent(turn["sent"])


def chat_area(level: int, turns: list[dict], avatar: str, placeholder: str, where=None):
    """입력창은 화면 맨 아래에 고정하고, 대화는 최신순(새 대화가 위)으로 보여 준다.

    where      대화를 그릴 영역 (Lv5처럼 칼럼 안에 그릴 때 지정). 없으면 본문.
    돌려주는 값:
      user_text  이번에 입력한 말 (튜토리얼 ▶ 해보기로 맡겨 둔 말 포함), 없으면 None
      live       새 대화를 그릴 자리 — 지난 대화들보다 위
    """
    # 본문 최상위에서 부르면 입력창이 스크롤과 상관없이 화면 맨 아래에 고정된다.
    user_text = st.chat_input(placeholder, key=f"lv{level}_chat") or pending_prompt(level)
    where = where or st.container()
    with where:
        live = st.container()
        if turns:
            st.caption(f"💬 지난 대화 {len(turns)}개 · 최신순")
        for turn in reversed(turns):  # 최신 대화가 위로
            render_turn(turn, avatar)
    return user_text, live


# ── 에이전트 실행 ────────────────────────────────────────
def history_messages(turns: list[dict]) -> list:
    """지난 대화를 LLM에게 다시 보낼 메시지 목록으로 바꾼다 (LLM은 기억이 없으니까!)."""
    msgs = []
    for t in turns:
        msgs += [HumanMessage(content=t["user"]), AIMessage(content=t["ai"])]
    return msgs


def run_agent(agent, system_prompt: str, turns: list[dict], user_text: str, remember: bool = True) -> dict:
    """에이전트를 한 번 실행하고, 화면에 보여 줄 정보(답·도구 기록·전체 메시지)를 돌려준다."""
    inputs = (history_messages(turns) if remember else []) + [HumanMessage(content=user_text)]
    result = agent.invoke({"messages": inputs})
    out = result["messages"]

    # 이번 턴에 새로 생긴 메시지에서 '도구 요청 ↔ 도구 결과'를 짝지어 기록한다.
    new = out[len(inputs):]
    results = {m.tool_call_id: m.content for m in new if isinstance(m, ToolMessage)}
    trace = [
        {"tool": tc["name"], "args": tc["args"], "result": str(results.get(tc["id"], "(결과 없음)"))}
        for m in new if isinstance(m, AIMessage) for tc in (m.tool_calls or [])
    ]
    return {
        "user": user_text,
        "ai": out[-1].content,
        "trace": trace,
        "sent": [SystemMessage(content=system_prompt)] + out,
    }
