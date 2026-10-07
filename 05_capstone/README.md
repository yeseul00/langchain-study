# 5단계 · 종합 실습: 원본 AI Incident Copilot 분석

원본: https://github.com/rahulwagh/ai-incident-copilot

4단계에서 만든 미니 Copilot과 **구조가 같습니다.** 이번 단계의 목표는 새로 배우는 것보다,
"실무용으로 만들면 무엇이 더 붙는가"를 보는 것입니다.

## 1. 내가 만든 것 ↔ 원본 파일 대응표

원본 코드는 `apps/ai-incident-copilot/src/copilot/` 아래에 있습니다.

| 개념 | 이 실습 | 원본 |
| --- | --- | --- |
| LLM 연결 | `ChatOllama(...)` | `agent/copilot.py` → `_model()` |
| 에이전트 생성 | `create_agent(...)` (2·4단계) | `agent/copilot.py` → `_agent()` |
| 시스템 프롬프트 | `SYSTEM_PROMPT` (4-3) | `agent/copilot.py` → `SYSTEM_PROMPT` |
| 동작 기록 출력 | `[에이전트 동작 기록]` | `agent/copilot.py` → `_flatten()` |
| 도구 정의 | `@tool` 3개 (4-3) | `agent/tools.py` → `@tool` 4개 |
| 필터 허용 목록 | `SERVICES`, `REGIONS` 검사 | `agent/tools.py` → `_ALLOWED_REGIONS` 등 |
| 임베딩 | `OllamaEmbeddings(bge-m3)`, 1024차원 | `rag/embedder.py` (MiniLM, 384차원) |
| 벡터 저장소 | `InMemoryVectorStore` | Oracle 26ai `VECTOR` 컬럼 + HNSW 인덱스 (`db/schema.sql`) |
| 데이터 넣기 | `knowledge.py` | `db/seed.py` |
| 장애 기록·런북 | JSON 파일 | DB 테이블 (`incidents`, `runbooks`, `services`, `incident_runbooks`) |

## 2. 원본에만 있는 것 (실무용으로 더 붙는 것)

| 추가 요소 | 왜 필요한가 |
| --- | --- |
| **진짜 DB (Oracle 26ai)** | 메모리 저장소는 프로그램이 꺼지면 사라진다. 수만 건 이상이면 인덱스(HNSW)가 필요하다. |
| **SQL 한 줄로 필터 + 의미 검색** | `WHERE service = ... ORDER BY VECTOR_DISTANCE(...)`. 4단계의 `filter=lambda`를 DB가 대신한다. |
| **FastAPI** (`api/main.py`) | 다른 프로그램(슬랙 봇, 모니터링 알림 등)이 HTTP로 호출할 수 있게 한다. |
| **Streamlit UI** (`ui/app.py`) | 브라우저 화면. 도구 호출 기록을 펼쳐 볼 수 있다. |
| **테스트** (`tests/`) | LLM 없이 검색 함수만 따로 검증 (`_impl` 함수를 분리한 이유). |
| **Docker** | DB와 앱을 어느 PC에서든 같은 환경으로 띄운다. |

## 3. 코드 읽는 순서 (추천)

1. `agent/copilot.py` (97줄) — 4-3과 거의 같다. 차이점만 찾아보기.
2. `agent/tools.py` (243줄) — 도구 4개. 특히 `find_similar_incidents_filtered`의 SQL.
3. `db/schema.sql` — `VECTOR(384, FLOAT32)` 컬럼과 인덱스 정의.
4. `db/seed.py` — 4단계 `knowledge.py`가 하는 일을 DB에 하는 버전.
5. `api/main.py`, `ui/app.py` — 에이전트를 감싸는 껍데기.

## 4. 직접 띄워 보기 전 체크

- 디스크 **약 15GB** 필요 (Oracle 이미지 약 9GB + 모델)
- Docker Desktop 필요
- 원본의 `installation-instructions.md` 10.2절(`vector_memory_size` 설정)을 빼먹으면 인덱스 생성에서 실패한다.
- 원본은 임베딩을 `sentence-transformers`(파이썬 패키지, PyTorch 포함)로 하므로 설치가 오래 걸린다.
- 원본 데이터는 영어다. 한국어로 질문하면 검색 품질이 떨어질 수 있다 (MiniLM은 영어 중심 모델).

## 5. 도전 과제

1. 4단계 미니 Copilot에 FastAPI를 붙여 `POST /diagnose`로 호출되게 만들기 (원본 `api/main.py` 참고).
2. `InMemoryVectorStore`를 디스크에 저장되는 Chroma(`pip install langchain-chroma`)로 바꾸기.
3. 원본 데이터에 한국어 장애 기록을 추가하고, 임베딩 모델을 다국어 모델로 바꾸면 무엇을 고쳐야 하는지 찾아보기 (힌트: 벡터 차원 384 → ?).
