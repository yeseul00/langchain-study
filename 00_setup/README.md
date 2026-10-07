# 0단계 · 환경 준비

## 필요한 것

- Python 3.10 이상
- Ollama (설치 완료 상태 가정)
- 디스크 약 7GB (모델 2개)

## 1. 모델 내려받기

터미널에서 아래 두 개를 받습니다.

```bash
ollama pull llama3.1:8b        # 대화·판단을 맡는 LLM (약 4.9GB)
ollama pull bge-m3             # 문장을 숫자(벡터)로 바꾸는 임베딩 모델, 한국어 지원 (약 1.2GB)
```

> RAM이 8GB인 PC라면 `llama3.1:8b` 대신 `llama3.2:3b`를 받고,
> 실행 전에 `export OLLAMA_MODEL=llama3.2:3b` 를 입력하세요.
> 작은 모델일수록 빠르지만 도구를 잘못 부르는 경우가 늘어납니다.

Ollama가 켜져 있는지 확인:

```bash
ollama list          # 받은 모델 목록이 보이면 OK
```

목록이 안 나오면 Ollama 앱을 실행하거나 `ollama serve`를 다른 터미널에서 켜 두세요.

## 2. 파이썬 가상환경 + 패키지 설치

`langchain-study` 폴더에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

이후 실습할 때마다 터미널을 새로 열면 `source .venv/bin/activate` 를 먼저 해 주세요.

## 3. 점검 스크립트 실행

```bash
python 00_setup/check_env.py
```

모든 항목에 `[OK]`가 나오면 1단계로 넘어가세요.

## 모델 바꾸기

모든 예제는 아래 환경변수를 읽습니다. 안 정하면 기본값을 씁니다.

| 환경변수 | 기본값 | 용도 |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.1:8b` | 대화용 LLM |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | 임베딩 모델 |
