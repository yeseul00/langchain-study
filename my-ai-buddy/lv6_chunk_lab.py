"""Lv6 · 청킹 실험실 — 문서를 어떻게 자르면 검색이 잘 될까?

LLM 없이 임베딩 모델만 씁니다. Lv3(임베딩)·Lv4(RAG)의 '넣기' 단계를 직접 해 보는 레벨이에요.
실행: python -m streamlit run my-ai-buddy/app.py  (사이드바에서 Lv6 선택)
"""

import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import buddy_config as cfg
from common import get_embeddings, init_state, level_header
from tutorials import LV6, SAMPLE_LAW

st.set_page_config(page_title=f"Lv6 · {cfg.NAME}", page_icon=cfg.EMOJI)

SAMPLES_DIR = Path(__file__).parent / "law_samples"
SAMPLE_NAME = "연습용 가상 법령 (기본)"
METHODS = ["조 단위", "항 단위", "글자 수 단위"]
MIN_RELEVANCE = 0.45  # sproute-lex의 미수록 기준과 같은 값 (참고용)

init_state(lv6_text=SAMPLE_LAW, lv6_source=SAMPLE_NAME, lv6_method="조 단위", lv6_size=200,
           lv6_overlap=0, lv6_prefix=False, lv6_k=3,
           lv6_query="책을 잃어버렸는데 돈을 안 내면 어떻게 돼?")

level_header(
    6, "청킹 실험실",
    f"{cfg.EMOJI} {cfg.NAME}에게 법령을 기억시키기 전에, 문서를 '어떻게 자를지' 직접 실험해요. "
    "자르는 방식에 따라 검색 결과가 달라집니다.",
    missions=[
        "[자르기] 세 가지 방식으로 잘라 보고 청크 개수·길이를 비교한다",
        "[자르기] 글자 수 단위에서 '조 경계를 넘은' 청크를 찾는다",
        "[검색] 같은 질문에서 방식마다 1등 청크가 다른 경우를 찾는다",
        "[검색] '조 제목 붙이기'를 켜고 끄며 유사도가 바뀌는 걸 확인한다",
        "[검색] 유사도가 0.45 아래로 떨어지는 질문을 만들어 본다",
        "[내 법령] 법제처에서 복사한 진짜 조문으로 같은 실험을 해 본다",
    ],
    principle="""
**① 청킹 = 검색 단위 정하기**
임베딩은 '조각 하나 = 좌표 하나'예요. 조각이 크면 여러 뜻이 섞여 좌표가 흐려지고,
너무 작으면 앞뒤 맥락("그 기간", "제2항에 따른")을 잃어요.

**② 세 가지 자르는 법**
- 조 단위: 의미는 온전하지만 조각이 큼
- 항 단위: 검색은 날카롭지만 조 제목·주어가 빠짐 (sproute-lex가 쓰는 방식)
- 글자 수 단위: 구조를 모를 때 쓰는 방법. 문장·조 중간이 잘림 → 겹치기(overlap)로 보완

**③ 맥락 붙이기**
조각 앞에 `[제3조(대출)]`처럼 제목을 붙여서 임베딩하면 잃어버린 맥락을 일부 되찾아요.

**④ 부모 복원**
검색은 작은 조각으로, LLM에게는 그 조각이 속한 **조문 전체**를 넘겨요. sproute-lex의 방식이에요.

**⑤ 넣기는 한 번, 찾기는 매번**
조각 임베딩은 원문이 바뀔 때만 다시 하고, 질문은 올 때마다 임베딩해요.
""",
    tutorial=LV6,
)


# ── 자르기 ──────────────────────────────────────────────
ART_RE = re.compile(r"^\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?", re.M)
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def split_articles(text: str) -> list[dict]:
    """원문 → 조문 목록. 각 조문 = {no, title, header, body, start}. 조가 없으면 통째로 하나."""
    marks = list(ART_RE.finditer(text))
    if not marks:
        return [{"no": "전체", "title": "", "header": "", "body": text.strip(), "start": 0}]
    arts = []
    head = text[:marks[0].start()].strip()
    if head:
        arts.append({"no": "머리말", "title": "", "header": "", "body": head, "start": 0})
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        arts.append({"no": m.group(1), "title": m.group(2) or "", "header": m.group(0).strip(),
                     "body": text[m.start():end].strip(), "start": m.start()})
    return arts


def label(art: dict) -> str:
    return f"{art['no']}({art['title']})" if art["title"] else art["no"]


def split_clauses(art: dict) -> list[tuple[str, str]]:
    """조문 → [(항 번호, 항 본문)]. 줄 맨 앞이나 조 제목 바로 뒤의 ①②… 를 항의 시작으로 본다."""
    body = art["body"]
    starts = [i for i, ch in enumerate(body)
              if ch in CIRCLED and (i == 0 or body[i - 1] in "\n )")]
    if not starts:
        return [("", body)]
    return [(body[s], body[s:e].strip()) for s, e in zip(starts, starts[1:] + [len(body)])]


def make_chunks(text: str, method: str, size: int, overlap: int, prefix: bool) -> list[dict]:
    """청크 = {id, article, text(원래 조각), embed_text(실제로 임베딩할 글), note}."""
    text = text.strip()
    arts = split_articles(text)
    chunks = []
    if method == "조 단위":
        for a in arts:
            chunks.append({"id": a["no"], "article": a["no"], "text": a["body"], "note": ""})
    elif method == "항 단위":
        for a in arts:
            for no, body in split_clauses(a):
                chunks.append({"id": f"{a['no']} {no}".strip(), "article": a["no"], "text": body, "note": ""})
    else:  # 글자 수 단위
        flat = text
        step = max(size - overlap, 1)
        for start in range(0, len(flat), step):
            piece = flat[start:start + size]
            touched = [a["no"] for a in arts if a["start"] < start + size and
                       (a["start"] + len(a["body"]) > start)]
            first = next((a["no"] for a in reversed(arts) if a["start"] <= start), arts[0]["no"])
            note = f"⚠️ 조 경계를 넘음: {', '.join(touched)}" if len(set(touched)) > 1 else ""
            chunks.append({"id": f"#{len(chunks) + 1} ({start}~{start + len(piece)}자)",
                           "article": first, "text": piece, "note": note,
                           "overlap": flat[start:start + overlap] if start and overlap else ""})
            if start + size >= len(flat):
                break
    by_no = {a["no"]: a for a in arts}
    for c in chunks:
        a = by_no.get(c["article"])
        need = prefix and a and a["header"] and not c["text"].startswith(a["header"])
        c["embed_text"] = f"[{label(a)}] {c['text']}" if need else c["text"]
    return chunks


# ── 임베딩 ──────────────────────────────────────────────
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@st.cache_data(show_spinner=False, max_entries=50)
def embed_texts(texts: tuple[str, ...]) -> list[list[float]]:
    """같은 조각은 다시 임베딩하지 않도록 결과를 기억해 둔다 (lex의 Chroma 저장과 같은 역할)."""
    return get_embeddings().embed_documents(list(texts))


# ── 원문 고르기 ─────────────────────────────────────────
def load_source() -> None:
    name = st.session_state["lv6_source"]
    st.session_state["lv6_text"] = (SAMPLE_LAW if name == SAMPLE_NAME
                                    else (SAMPLES_DIR / name).read_text(encoding="utf-8"))


files = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix in (".txt", ".md") and p.name != "README.md") if SAMPLES_DIR.exists() else []

with st.sidebar:
    st.subheader("✂️ 자르기 설정")
    st.radio("자르는 방식 (자르기 탭)", METHODS, key="lv6_method")
    st.slider("글자 수 단위: 조각 크기", 50, 600, step=10, key="lv6_size")
    st.slider("글자 수 단위: 겹치는 글자 수", 0, 200, step=10, key="lv6_overlap")
    st.toggle("조각 앞에 조 제목 붙이기", key="lv6_prefix",
              help="예: '② 대출 기간은…' → '[제3조(대출)] ② 대출 기간은…' 으로 임베딩")
    if st.session_state["lv6_overlap"] >= st.session_state["lv6_size"]:
        st.warning("겹치는 글자 수는 조각 크기보다 작아야 해요.")

with st.expander("📜 원문 (sproute-lex의 '정본' 역할) — 법제처에서 복사한 조문을 붙여 넣어도 돼요",
                 expanded=False):
    st.selectbox("원문 불러오기", [SAMPLE_NAME] + files, key="lv6_source", on_change=load_source,
                 help="my-ai-buddy/law_samples/ 폴더에 .txt나 .md 파일을 넣으면 여기 나타나요.")
    st.text_area("원문", key="lv6_text", height=300)

text = st.session_state["lv6_text"]
size, overlap, prefix = st.session_state["lv6_size"], st.session_state["lv6_overlap"], st.session_state["lv6_prefix"]
articles = {a["no"]: a for a in split_articles(text.strip())}

tab1, tab2 = st.tabs(["✂️ 자르기", "🔍 검색 비교"])

# ── 탭 1: 자르기 결과 보기 ──────────────────────────────
with tab1:
    method = st.session_state["lv6_method"]
    chunks = make_chunks(text, method, size, overlap, prefix)
    lens = [len(c["embed_text"]) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("청크 수", len(chunks))
    c2.metric("평균 글자", round(sum(lens) / len(lens)) if lens else 0)
    c3.metric("가장 짧은", min(lens, default=0))
    c4.metric("가장 긴", max(lens, default=0))
    st.bar_chart(pd.DataFrame({"청크": [c["id"] for c in chunks], "글자 수": lens}),
                 x="청크", y="글자 수", horizontal=True)
    st.caption(f"방식: **{method}** · 조 제목 붙이기: **{'켬' if prefix else '끔'}** · "
               "아래 상자 안의 글이 실제로 임베딩되는 내용이에요.")
    for c in chunks:
        with st.container(border=True):
            st.markdown(f"**{c['id']}** · {len(c['embed_text'])}자 {c['note']}")
            st.text(c["embed_text"])
            if c.get("overlap"):
                st.caption(f"↩️ 앞 조각과 겹친 부분: “{c['overlap']}”")

# ── 탭 2: 세 방식 검색 비교 ─────────────────────────────
with tab2:
    st.markdown("같은 질문으로 **세 가지 방식을 동시에** 검색해요. 왼쪽 사이드바 설정(크기·겹침·제목 붙이기)이 그대로 적용돼요.")
    q_col, k_col = st.columns([4, 1])
    query = q_col.text_input("질문", key="lv6_query")
    k = k_col.number_input("몇 개", 1, 10, key="lv6_k")
    if (st.button("검색 비교", type="primary") or st.session_state.pop("lv6_run_search", False)) and query:
        all_chunks = {m: make_chunks(text, m, size, overlap, prefix) for m in METHODS}
        with st.spinner("조각과 질문을 좌표로 바꾸는 중... (처음 한 번은 조금 걸려요)"):
            q_vec = get_embeddings().embed_query(query)
            vecs = {m: embed_texts(tuple(c["embed_text"] for c in cs)) for m, cs in all_chunks.items()}
        cols = st.columns(3)
        for col, m in zip(cols, METHODS):
            scored = sorted(zip(all_chunks[m], (cosine(q_vec, v) for v in vecs[m])),
                            key=lambda x: x[1], reverse=True)[:k]
            with col:
                st.markdown(f"#### {m}")
                st.caption(f"조각 {len(all_chunks[m])}개 중 상위 {len(scored)}")
                for rank, (c, s) in enumerate(scored, 1):
                    with st.container(border=True):
                        warn = " · ⚠️ 0.45 미만" if s < MIN_RELEVANCE else ""
                        st.markdown(f"**{rank}. {c['id']}**  \n유사도 `{s:.3f}`{warn}")
                        st.text(c["embed_text"][:220] + ("…" if len(c["embed_text"]) > 220 else ""))
                        parent = articles.get(c["article"])
                        if parent and m != "조 단위":
                            with st.expander(f"📄 부모 복원: {label(parent)} 전체"):
                                st.text(parent["body"])
        st.info(f"⚠️ 표시는 sproute-lex라면 '수록 자료에 없음'으로 처리했을 점수예요 (기준 {MIN_RELEVANCE}). "
                "lex는 HuggingFace로, 여기는 Ollama로 같은 bge-m3를 돌려서 점수가 조금 다를 수 있어요.")
