"""✂️ 청킹(문서 자르기) 함수 모음 — Lv6 청킹 실험실이 사용

LLM도 임베딩도 쓰지 않는 순수 문자열 처리예요. 그래서 따로 떼어 두었어요.
sproute-lex의 backend/core/chunking.py 와 같은 역할입니다.

지원하는 원문 모양
  - 법령·고시: `제3조(대출)`, `### 제1조(목적)` 같은 조 제목 + `①②` 항
  - 안내서·기술규격: `## 2. 개요`, `### 2.1 정의` 같은 마크다운 제목
  - PDF를 변환한 MD의 머리 정보(--- ... ---)와 `<!-- page: N -->` 페이지 표시
"""

import re
from dataclasses import dataclass, field

METHODS = ["조 단위", "항 단위", "제목 단위", "글자 수 단위"]
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

FRONT_RE = re.compile(r"\A\s*---\n.*?\n---\n", re.S)
PAGE_RE = re.compile(r"[ \t]*<!--\s*page:\s*(\d+)\s*-->[ \t]*\n?")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ART_RE = re.compile(r"^(#{1,6}\s+)?\**\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)\n]*)\))?(.*)$")


# ── 1. 원문 다듬기 ───────────────────────────────────────
@dataclass
class Doc:
    text: str                                         # 머리 정보·페이지 표시를 뺀 본문
    pages: list[tuple[int, int]] = field(default_factory=list)  # (본문 위치, 페이지 번호)
    removed_front: bool = False

    def page_at(self, pos: int) -> int | None:
        page = self.pages[0][1] if self.pages else None   # 첫 표시 앞부분은 첫 페이지로 본다
        for p, n in self.pages:
            if p > pos:
                break
            page = n
        return page


def prepare(raw: str) -> Doc:
    """머리 정보(--- ... ---)를 떼고, 페이지 표시는 지우되 '몇 쪽인지'는 위치로 기억한다."""
    text = raw.replace("\r\n", "\n")
    front = FRONT_RE.match(text)
    if front:
        text = text[front.end():]
    out, pages, last = [], [], 0
    for m in PAGE_RE.finditer(text):
        out.append(text[last:m.start()])
        pages.append((sum(len(s) for s in out), int(m.group(1))))
        last = m.end()
    out.append(text[last:])
    body = "".join(out)
    lead = len(body) - len(body.lstrip())
    pages = [(max(p - lead, 0), n) for p, n in pages]
    return Doc(text=body.strip(), pages=pages, removed_front=bool(front))


# ── 2. 구조 읽기: 조문 덩어리 / 제목 덩어리 ─────────────────
def _article(line: str):
    """이 줄이 조 제목이면 (제목 수준, 조 번호, 조 제목, 같은 줄 나머지) 를, 아니면 None."""
    m = ART_RE.match(line)
    if not m:
        return None
    hashes, no, title, rest = m.groups()
    rest = (rest or "").strip().strip("*").strip()
    # 본문 속 "제22조에 따른 …" 같은 인용 문장은 조 제목이 아니다.
    if not (hashes or title is not None or rest == "" or rest[0] in CIRCLED):
        return None
    level = len(hashes.strip()) if hashes else 99
    return level, no, (title or "").strip(), rest


def _lines(text: str):
    """(위치, 줄, 코드블록 안인지) 를 하나씩 돌려준다.

    ``` 코드블록 안의 `# 주석` 은 마크다운 제목이 아니므로 구분해야 한다.
    (기술규격의 예시 코드 `# authorization_code` 를 제목으로 잘못 읽는 문제)
    """
    pos, fence = 0, False
    for line in text.split("\n"):
        is_fence = line.lstrip().startswith(("```", "~~~"))
        yield pos, line, fence or is_fence
        if is_fence:
            fence = not fence
        pos += len(line) + 1


def blocks(doc: Doc) -> list[dict]:
    """본문 → 덩어리 목록. kind='article'(조문) 또는 'other'(조 밖 글: 머리말·장 제목·설명).

    각 덩어리 = {kind, key, no, title, path(위쪽 제목들), start, text}
    """
    out, cur, path = [], None, {}

    def close():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            out.append(cur)
        cur = None

    def where():
        return " > ".join(path[k] for k in sorted(path))

    for pos, line, code in _lines(doc.text):
        art, head = (None, None) if code else (_article(line), HEAD_RE.match(line))
        if art:
            close()
            level, no, title, rest = art
            cur = {"kind": "article", "no": no, "title": title, "path": where(),
                   "level": level, "start": pos, "text": line + "\n"}
        elif head:
            level = len(head.group(1))
            if cur and cur["kind"] == "article" and level > cur["level"]:
                cur["text"] += line + "\n"          # 조 안의 소제목
                continue
            close()
            for k in [k for k in path if k >= level]:
                del path[k]
            path[level] = head.group(2).strip("* ")
            cur = {"kind": "other", "no": "", "title": "", "path": where(),
                   "level": level, "start": pos, "text": line + "\n"}
        else:
            if cur is None:
                cur = {"kind": "other", "no": "", "title": "", "path": where(),
                       "level": 0, "start": pos, "text": ""}
            cur["text"] += line + "\n"
    close()

    # 제목만 있고 내용이 없는 덩어리(장 제목 등)는 버린다.
    out = [b for b in out if b["kind"] == "article" or
           any(l.strip() and not HEAD_RE.match(l) for l in b["text"].split("\n"))]

    # 덩어리 이름 붙이기. 같은 조 번호가 또 나오면(부칙 제1조 등) 위쪽 제목으로 구분한다.
    seen: dict[str, int] = {}
    for b in out:
        if b["kind"] == "article":
            base = f"{b['no']}({b['title']})" if b["title"] else b["no"]
            n = seen.get(base, 0)
            seen[base] = n + 1
            last = b["path"].split(" > ")[-1] if b["path"] else ""
            if "부칙" in last:                           # 부칙 조문은 항상 '부칙'을 붙여 본문과 구분
                b["key"] = f"부칙 {base}"
            elif n == 0:
                b["key"] = base
            else:
                b["key"] = f"{last} {base}" if last else f"{base} #{n + 1}"
        else:
            b["key"] = "조 밖: " + (b["path"] or "머리말")
    return out


def sections(doc: Doc, max_level: int = 4) -> list[dict]:
    """본문 → 마크다운 제목(#~####)마다 한 덩어리. 안내서·기술규격처럼 조가 없는 문서용."""
    out, cur, path = [], None, {}
    for pos, line, code in _lines(doc.text):
        head = None if code else HEAD_RE.match(line)
        if head and len(head.group(1)) <= max_level:
            if cur:
                out.append(cur)
            level = len(head.group(1))
            for k in [k for k in path if k >= level]:
                del path[k]
            path[level] = head.group(2).strip("* ")
            cur = {"key": " > ".join(path[k] for k in sorted(path)), "start": pos, "text": line + "\n"}
        else:
            if cur is None:
                cur = {"key": "머리말", "start": pos, "text": ""}
            cur["text"] += line + "\n"
    if cur:
        out.append(cur)
    for s in out:
        s["text"] = s["text"].strip()
    return [s for s in out if any(l.strip() and not HEAD_RE.match(l) for l in s["text"].split("\n"))]


def split_clauses(text: str) -> list[tuple[str, str]]:
    """조문 → [(항 번호, 항 본문)]. 줄 맨 앞이나 조 제목 바로 뒤의 ①②… 를 항의 시작으로 본다."""
    starts = [i for i, ch in enumerate(text) if ch in CIRCLED and (i == 0 or text[i - 1] in "\n )")]
    if not starts:
        return [("", text)]
    return [(text[s], text[s:e].strip()) for s, e in zip(starts, starts[1:] + [len(text)])]


# ── 3. 자르기 ───────────────────────────────────────────
def _chunk(cid, text, where, start, parent=None, note=""):
    return {"id": cid, "text": text, "where": where, "start": start,
            "parent": parent, "note": note, "overlap": ""}


def make_chunks(doc: Doc, method: str, size: int = 200, overlap: int = 0,
                prefix: bool = False) -> list[dict]:
    """청크 = {id, text(원래 조각), embed_text(실제로 임베딩할 글), where, page, parent, note, overlap}

    parent: 부모 복원 때 보여 줄 '조문 전체' 또는 '절 전체' (없으면 None)
    """
    chunks: list[dict] = []
    if method == "조 단위":
        for b in blocks(doc):
            chunks.append(_chunk(b["key"], b["text"], b["path"], b["start"]))
    elif method == "항 단위":
        for b in blocks(doc):
            parts = split_clauses(b["text"]) if b["kind"] == "article" else [("", b["text"])]
            for no, body in parts:
                pos = doc.text.find(body[:40], b["start"])
                # 항으로 쪼갠 조각에는 조 제목이 없으니 경로에 조 이름까지 넣는다.
                where = " > ".join(x for x in (b["path"], b["key"] if no else "") if x)
                chunks.append(_chunk(f"{b['key']} {no}".strip(), body, where,
                                     pos if pos >= 0 else b["start"],
                                     parent=b if len(parts) > 1 else None))
    elif method == "제목 단위":
        for s in sections(doc):
            chunks.append(_chunk(s["key"], s["text"], s["key"], s["start"]))
    else:  # 글자 수 단위
        flat, bl = doc.text, blocks(doc)
        step = max(size - overlap, 1)
        for start in range(0, len(flat), step):
            piece = flat[start:start + size]
            end = start + len(piece)
            touched = [b for b in bl if b["start"] < end and b["start"] + len(b["text"]) > start]
            first = next((b for b in reversed(bl) if b["start"] <= start), bl[0] if bl else None)
            note = ""
            if len(touched) > 1:
                note = "⚠️ 경계를 넘음: " + ", ".join(t["key"] for t in touched[:3]) + \
                       (" …" if len(touched) > 3 else "")
            where = (" > ".join(x for x in (first["path"], first["key"]) if x)) if first else ""
            c = _chunk(f"#{len(chunks) + 1} ({start}~{end}자)", piece, where, start,
                       parent=first, note=note)
            if start and overlap:
                c["overlap"] = flat[start:start + overlap]
            chunks.append(c)
            if end >= len(flat):
                break

    for c in chunks:
        c["page"] = doc.page_at(c["start"])
        add = prefix and method in ("항 단위", "제목 단위", "글자 수 단위") and c["where"]
        # 조 단위·제목 단위 조각은 이미 제목으로 시작하니, 위쪽 경로만 붙일 때 의미가 있다.
        c["embed_text"] = f"[{c['where']}] {c['text']}" if add else c["text"]
    return chunks


def available_methods(doc: Doc) -> list[str]:
    """이 원문에 의미 있는 자르기 방식만 고른다 (조가 없으면 조·항 단위는 뺀다)."""
    has_art = any(b["kind"] == "article" for b in blocks(doc))
    has_head = any(not code and HEAD_RE.match(l) for _, l, code in _lines(doc.text))
    return [m for m in METHODS if (m in ("조 단위", "항 단위") and has_art)
            or (m == "제목 단위" and has_head) or m == "글자 수 단위"]
