"""Lv6 · 청킹 실험실 — 문서를 어떻게 자르면 검색이 잘 될까?

LLM 없이 임베딩 모델만 씁니다. Lv3(임베딩)·Lv4(RAG)의 '넣기' 단계를 직접 해 보는 레벨이에요.
실행: python -m streamlit run my-ai-buddy/app.py  (사이드바에서 Lv6 선택)
"""

import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import buddy_config as cfg
from bm25 import BM25, ranks, rrf
from chunking import METHODS, available_methods, make_chunks, prepare
from common import get_embeddings, init_state, level_header
from tutorials import LV6, SAMPLE_LAW

st.set_page_config(page_title=f"Lv6 · {cfg.NAME}", page_icon=cfg.EMOJI)

SAMPLES_DIR = Path(__file__).parent / "law_samples"
SAMPLE_NAME = "연습용 가상 법령 (기본)"
MIN_RELEVANCE = 0.45   # sproute-lex의 미수록 기준과 같은 값 (참고용)
MAX_EMBED_CHARS = 6000  # 이보다 긴 조각은 앞부분만 임베딩 (bge-m3 입력 한도 안전선)
BATCH = 32              # 한 번에 임베딩할 조각 수
SEARCH_MODES = ["의미 (임베딩)", "키워드 (BM25)", "하이브리드 (RRF)"]

init_state(lv6_text=SAMPLE_LAW, lv6_source=SAMPLE_NAME,
           lv6_dir=os.getenv("LV6_SAMPLES_DIR", str(SAMPLES_DIR)), lv6_method="조 단위", lv6_size=200,
           lv6_overlap=0, lv6_prefix=False, lv6_k=3, lv6_mode="의미 (임베딩)",
           lv6_query="책을 잃어버렸는데 돈을 안 내면 어떻게 돼?")

level_header(
    6, "청킹 실험실",
    f"{cfg.EMOJI} {cfg.NAME}에게 법령을 기억시키기 전에, 문서를 '어떻게 자를지' 직접 실험해요. "
    "자르는 방식에 따라 검색 결과가 달라집니다.",
    missions=[
        "[자르기] 여러 방식으로 잘라 보고 청크 개수·길이를 비교한다",
        "[자르기] 글자 수 단위에서 '경계를 넘은' 청크를 찾는다",
        "[검색] 같은 질문에서 방식마다 1등 청크가 다른 경우를 찾는다",
        "[검색] '조 제목 붙이기'를 켜고 끄며 유사도가 바뀌는 걸 확인한다",
        "[검색] 유사도가 0.45 아래로 떨어지는 질문을 만들어 본다",
        "[내 문서] 고시(조문)와 안내서(제목 구조)에서 가장 잘 맞는 방식이 다른지 확인한다",
    ],
    principle="""
**① 청킹 = 검색 단위 정하기**
임베딩은 '조각 하나 = 좌표 하나'예요. 조각이 크면 여러 뜻이 섞여 좌표가 흐려지고,
너무 작으면 앞뒤 맥락("그 기간", "제2항에 따른")을 잃어요.

**② 네 가지 자르는 법** (코드는 `chunking.py`)
- 조 단위: 의미는 온전하지만 조각이 큼
- 항 단위: 검색은 날카롭지만 조 제목·주어가 빠짐 (sproute-lex가 쓰는 방식)
- 제목 단위: 안내서·기술규격처럼 조가 없는 문서는 `##` 제목(절)마다 자름
- 글자 수 단위: 구조를 모를 때 쓰는 방법. 문장·조 중간이 잘림 → 겹치기(overlap)로 보완

**③ 맥락 붙이기**
조각 앞에 `[제1장 총칙 > 제3조(대출)]`처럼 위쪽 제목을 붙여서 임베딩하면 잃어버린 맥락을 일부 되찾아요.
PDF에서 바꾼 문서는 머리 정보(`---`)와 페이지 표시(`<!-- page -->`)를 떼고, 페이지 번호는 **메타데이터**로 따로 기억해요.

**④ 부모 복원**
검색은 작은 조각으로, LLM에게는 그 조각이 속한 **조문 전체**를 넘겨요. sproute-lex의 방식이에요.

**⑤ 하이브리드 검색** (코드는 `bm25.py`)
임베딩은 `4.3.4`·`제31조의18` 같은 번호·이름에 약해요. 글자가 겹치는지 보는 **키워드 검색(BM25)**과
두 결과의 **등수**를 합치는 **RRF**로 서로의 약점을 메워요.

**⑥ 넣기는 한 번, 찾기는 매번**
조각 임베딩은 원문이 바뀔 때만 다시 하고, 질문은 올 때마다 임베딩해요.
""",
    tutorial=LV6,
)



# ── 임베딩 ──────────────────────────────────────────────
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@st.cache_data(show_spinner=False, max_entries=500)
def embed_batch(texts: tuple[str, ...]) -> list[list[float]]:
    """같은 조각 묶음은 다시 임베딩하지 않도록 결과를 기억해 둔다 (lex의 Chroma 저장과 같은 역할)."""
    return get_embeddings().embed_documents(list(texts))


def embed_all(texts: list[str], label: str) -> list[list[float]]:
    """조각을 BATCH개씩 나눠 임베딩하며 진행률을 보여 준다."""
    clipped = [t[:MAX_EMBED_CHARS] for t in texts]
    bar = st.progress(0.0, text=f"{label}: 조각 {len(texts)}개 임베딩 중...")
    out: list[list[float]] = []
    for i in range(0, len(clipped), BATCH):
        out += embed_batch(tuple(clipped[i:i + BATCH]))
        bar.progress(min((i + BATCH) / len(clipped), 1.0), text=f"{label}: {len(out)} / {len(texts)}")
    bar.empty()
    return out


# ── 원문 고르기 ─────────────────────────────────────────
def list_files(folder: str) -> list[str]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return []
    return sorted(str(f.relative_to(p)) for f in p.rglob("*")
                  if f.suffix in (".txt", ".md") and f.name != "README.md")


def load_source() -> None:
    name = st.session_state["lv6_source"]
    if name == SAMPLE_NAME:
        st.session_state["lv6_text"] = SAMPLE_LAW
    else:
        path = Path(st.session_state["lv6_dir"]).expanduser() / name
        st.session_state["lv6_text"] = path.read_text(encoding="utf-8")


with st.expander("📜 원문 (sproute-lex의 '정본' 역할) — 파일을 고르거나 직접 붙여 넣어요", expanded=False):
    st.text_input("원문 폴더", key="lv6_dir",
                  help="이 폴더(하위 폴더 포함)의 .md·.txt 파일이 아래 목록에 나와요. "
                       "예: ~/DEVELOPER/sprout-lex/docs/seul_test/md")
    files = list_files(st.session_state["lv6_dir"])
    if st.session_state["lv6_source"] not in [SAMPLE_NAME] + files:
        st.session_state["lv6_source"] = SAMPLE_NAME
    st.selectbox("원문 불러오기", [SAMPLE_NAME] + files, key="lv6_source", on_change=load_source)
    if not files:
        st.caption("폴더에 .md·.txt 파일이 없어요. 경로를 확인하세요.")
    st.text_area("원문", key="lv6_text", height=300)

doc = prepare(st.session_state["lv6_text"])
methods_here = available_methods(doc)
if st.session_state["lv6_method"] not in methods_here:
    st.session_state["lv6_method"] = methods_here[0]

with st.sidebar:
    st.subheader("✂️ 자르기 설정")
    st.radio("자르는 방식 (자르기 탭)", methods_here, key="lv6_method",
             help="원문에 조가 없으면 조·항 단위, 마크다운 제목이 없으면 제목 단위는 목록에서 빠져요.")
    st.slider("글자 수 단위: 조각 크기", 50, 1000, step=10, key="lv6_size")
    st.slider("글자 수 단위: 겹치는 글자 수", 0, 300, step=10, key="lv6_overlap")
    st.toggle("조각 앞에 위쪽 제목 붙이기", key="lv6_prefix",
              help="예: '② 대출 기간은…' → '[제3조(대출)] ② 대출 기간은…' 으로 임베딩")
    if st.session_state["lv6_overlap"] >= st.session_state["lv6_size"]:
        st.warning("겹치는 글자 수는 조각 크기보다 작아야 해요.")

size, overlap, prefix = st.session_state["lv6_size"], st.session_state["lv6_overlap"], st.session_state["lv6_prefix"]
info = [f"본문 {len(doc.text):,}자"]
if doc.removed_front:
    info.append("머리 정보(---) 제외")
if doc.pages:
    info.append(f"페이지 표시 {len(doc.pages)}개 → 메타데이터로 보관")
st.caption(" · ".join(info) + f" · 쓸 수 있는 방식: {', '.join(methods_here)}")

tab1, tab2 = st.tabs(["✂️ 자르기", "🔍 검색 비교"])


def show_chunk(c: dict) -> None:
    clip = " · ✂️ 너무 길어 앞부분만 임베딩" if len(c["embed_text"]) > MAX_EMBED_CHARS else ""
    page = f" · p.{c['page']}" if c["page"] else ""
    st.markdown(f"**{c['id']}** · {len(c['embed_text']):,}자{page}{clip} {c['note']}")
    st.text(c["embed_text"])
    if c.get("overlap"):
        st.caption(f"↩️ 앞 조각과 겹친 부분: “{c['overlap']}”")


# ── 탭 1: 자르기 결과 보기 ──────────────────────────────
with tab1:
    method = st.session_state["lv6_method"]
    chunks = make_chunks(doc, method, size, overlap, prefix)
    lens = [len(c["embed_text"]) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("청크 수", len(chunks))
    c2.metric("평균 글자", round(sum(lens) / len(lens)) if lens else 0)
    c3.metric("가장 짧은", min(lens, default=0))
    c4.metric("가장 긴", max(lens, default=0))
    long_n = sum(n > MAX_EMBED_CHARS for n in lens)
    if long_n:
        st.warning(f"{MAX_EMBED_CHARS:,}자가 넘는 조각이 {long_n}개 있어요. 임베딩할 때 뒷부분이 잘려서 검색에 안 걸려요.")
    st.caption(f"방식: **{method}** · 위쪽 제목 붙이기: **{'켬' if prefix else '끔'}** · "
               "보이는 글이 실제로 임베딩되는 내용이에요.")

    if len(chunks) <= 30:
        st.bar_chart(pd.DataFrame({"청크": [c["id"] for c in chunks], "글자 수": lens}),
                     x="청크", y="글자 수", horizontal=True)
        for c in chunks:
            with st.container(border=True):
                show_chunk(c)
    else:
        # 조각이 많으면 표로 훑어보고, 하나를 골라 자세히 본다.
        st.dataframe(pd.DataFrame({
            "청크": [c["id"] for c in chunks],
            "쪽": [c["page"] for c in chunks],
            "글자 수": lens,
            "표시": [c["note"] for c in chunks],
            "앞부분": [c["embed_text"][:60].replace("\n", " ") for c in chunks],
        }), hide_index=True, height=320)
        pick = st.selectbox("자세히 볼 청크", range(len(chunks)),
                            format_func=lambda i: f"{i + 1}. {chunks[i]['id']}")
        with st.container(border=True):
            show_chunk(chunks[pick])

# ── 탭 2: 방식별 검색 비교 ─────────────────────────────
with tab2:
    st.markdown("같은 질문으로 **여러 자르기 방식을 동시에** 검색해요. 사이드바 설정(크기·겹침·제목 붙이기)이 그대로 적용돼요.")
    cmp = st.multiselect("비교할 자르기 방식", methods_here, default=methods_here)
    st.radio("검색 방식", SEARCH_MODES, key="lv6_mode", horizontal=True,
             help="의미: 임베딩 유사도 / 키워드: 글자가 겹치는 정도(BM25) / 하이브리드: 두 등수를 RRF로 합침")
    q_col, k_col = st.columns([4, 1])
    query = q_col.text_input("질문", key="lv6_query")
    k = k_col.number_input("몇 개", 1, 10, key="lv6_k")
    total = sum(len(make_chunks(doc, m, size, overlap, prefix)) for m in cmp)
    st.caption(f"임베딩할 조각: 모두 {total:,}개 (처음 한 번만 오래 걸리고, 같은 조각은 기억해 둬요)")

    if (st.button("검색 비교", type="primary") or st.session_state.pop("lv6_run_search", False)) and query and cmp:
        mode = st.session_state["lv6_mode"]
        all_chunks = {m: make_chunks(doc, m, size, overlap, prefix) for m in cmp}
        q_vec = get_embeddings().embed_query(query)
        vecs = {m: embed_all([c["embed_text"] for c in cs], m) for m, cs in all_chunks.items()}
        cols = st.columns(min(len(cmp), 2))
        for i, m in enumerate(cmp):
            cs = all_chunks[m]
            sem = [cosine(q_vec, v) for v in vecs[m]]                        # 의미 점수 (0~1)
            kw = BM25([c["embed_text"] for c in cs]).scores(query)           # 키워드 점수 (0~수십)
            sem_rank, kw_rank = ranks(sem), ranks(kw)
            final = {"의미 (임베딩)": sem, "키워드 (BM25)": kw, "하이브리드 (RRF)": rrf(sem_rank, kw_rank)}[mode]
            order = sorted(range(len(cs)), key=lambda j: final[j], reverse=True)[:k]
            with cols[i % len(cols)]:
                st.markdown(f"#### {m}")
                st.caption(f"조각 {len(cs)}개 중 상위 {len(order)} · {mode}")
                for rank, j in enumerate(order, 1):
                    c = cs[j]
                    with st.container(border=True):
                        warn = " · ⚠️ 0.45 미만" if sem[j] < MIN_RELEVANCE else ""
                        page = f" · p.{c['page']}" if c["page"] else ""
                        kw_txt = f"{kw[j]:.2f} ({kw_rank[j]}등)" if kw_rank[j] else "0 (단어 안 겹침)"
                        st.markdown(f"**{rank}. {c['id']}**{page}  \n"
                                    f"의미 `{sem[j]:.3f}` ({sem_rank[j]}등){warn} · 키워드 `{kw_txt}`")
                        st.text(c["embed_text"][:220] + ("…" if len(c["embed_text"]) > 220 else ""))
                        if c["parent"] is not None:
                            with st.expander(f"📄 부모 복원: {c['parent']['key']} 전체"):
                                st.text(c["parent"]["text"])
        st.info(f"⚠️ 0.45 미만은 sproute-lex라면 '수록 자료에 없음'으로 처리했을 **의미 점수**예요. "
                "키워드 점수는 단위가 달라서 그대로 비교할 수 없고, 하이브리드는 두 등수만 보고 합쳐요. "
                "lex는 HuggingFace로, 여기는 Ollama로 같은 bge-m3를 돌려서 점수가 조금 다를 수 있어요.")
