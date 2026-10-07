"""🧠 동료의 기억 — my_notes 폴더의 메모를 벡터 저장소에 넣고 검색 도구로 만든다 (Lv4, Lv5)

RAG의 두 흐름:
  ① 넣기: 메모 파일 → 조각내기 → 임베딩 → 벡터 저장소   (load_chunks, build_store)
  ② 찾기: 질문 → 임베딩 → 비슷한 조각 3개 → LLM에게 전달  (make_search_tool)
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

NOTES_DIR = Path(__file__).parent / "my_notes"


def note_files() -> list[Path]:
    return sorted(p for p in NOTES_DIR.glob("*") if p.suffix in {".md", ".txt"})


def notes_signature() -> tuple:
    """파일 이름과 수정 시각 목록. 메모가 바뀌면 값이 달라져서 저장소를 새로 만든다."""
    return tuple((p.name, p.stat().st_mtime) for p in note_files())


def load_chunks() -> list[Document]:
    """메모 파일을 빈 줄 기준으로 조각낸다. 각 조각 앞에 소속 제목(#)을 붙여 문맥을 살린다."""
    docs = []
    for path in note_files():
        heading = path.stem
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#") and "\n" not in block:  # 제목만 있는 줄
                heading = block.lstrip("# ").strip()
                continue
            docs.append(Document(page_content=f"[{heading}] {block}", metadata={"source": path.name}))
    return docs


def build_store(embeddings) -> InMemoryVectorStore:
    """메모 조각을 전부 임베딩해서 벡터 저장소에 넣는다 (RAG ① 넣기)."""
    store = InMemoryVectorStore(embedding=embeddings)
    chunks = load_chunks()
    if chunks:
        store.add_documents(chunks)
    return store


def make_search_tool(store: InMemoryVectorStore):
    @tool
    def search_my_notes(query: str) -> list[dict]:
        """사용자의 메모장(팀 규칙, 맛집, 서버 메모, 동료 프로필, 예전에 저장한 메모 등)에서
        질문과 의미가 비슷한 내용을 최대 3개 찾는다.
        사용자나 팀, 회사, 예전에 기억해 달라고 한 것에 대한 질문이면 답하기 전에 반드시 먼저 사용한다.

        Args:
            query: 찾고 싶은 내용 (사용자 질문을 그대로 넣어도 된다).
        """
        hits = store.similarity_search_with_score(query, k=3)
        return [{"source": d.metadata["source"], "content": d.page_content, "similarity": round(s, 3)}
                for d, s in hits]

    return search_my_notes
