"""0단계: 실습 환경 점검.

실행: python 00_setup/check_env.py
"""

import os
import sys
import json
import urllib.request

CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

ok = True


def report(passed: bool, msg: str, hint: str = "") -> None:
    global ok
    print(("[OK]   " if passed else "[FAIL] ") + msg)
    if not passed:
        ok = False
        if hint:
            print("       → " + hint)


# 1) 파이썬 버전
report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}",
       "Python 3.10 이상이 필요합니다.")

# 2) 패키지
try:
    import langchain
    import langchain_ollama  # noqa: F401
    report(True, f"langchain {langchain.__version__} / langchain-ollama 설치됨")
except ImportError as e:
    report(False, f"패키지 없음: {e.name}", "pip install -r requirements.txt")

# 3) Ollama 서버 + 모델
try:
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
        names = [m["name"] for m in json.load(r)["models"]]
    report(True, f"Ollama 서버 응답 ({OLLAMA_URL})")
    for model in (CHAT_MODEL, EMBED_MODEL):
        found = any(n == model or n.split(":")[0] == model for n in names)
        report(found, f"모델 '{model}'", f"ollama pull {model}")
except Exception as e:  # 연결 실패
    report(False, f"Ollama 서버 연결 실패: {e}", "Ollama 앱을 켜거나 `ollama serve` 실행")

print()
print("준비 완료! 01_llm_basics 로 넘어가세요." if ok else "위 [FAIL] 항목을 해결한 뒤 다시 실행하세요.")
