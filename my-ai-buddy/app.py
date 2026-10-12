"""🐹 내 AI 동료 만들기 — 모든 레벨을 한 화면에서

실행: python -m streamlit run my-ai-buddy/app.py

- 왼쪽 사이드바 맨 위에서 레벨을 골라 이동할 수 있어요.
- 튜토리얼을 완료하면 '다음 레벨로' 버튼이 나와요.
- lv7_xxx.py 처럼 'lv숫자_이름.py' 파일을 새로 만들면 자동으로 목록에 추가돼요.

(이 파일은 화면 연결용이라 학습 대상은 아니에요.)
"""

import re
from pathlib import Path

import streamlit as st

import common

HERE = Path(__file__).parent
BUILTIN_TITLES = {
    1: ("말하는 동료", "💬"),
    2: ("손이 생긴 동료", "🔧"),
    3: ("눈치 있는 동료", "😶"),
    4: ("기억하는 동료", "🧠"),
    5: ("출근하는 동료", "💼"),
    6: ("청킹 실험실", "✂️"),
}

# 레벨을 옮겨 다녀도 사이드바 설정·미션 체크가 지워지지 않게 붙잡아 둔다.
# (Streamlit은 화면에 없는 위젯의 값을 지우기 때문)
_KEEP = re.compile(r"^lv\d+_(name|avatar|role|personality|speech|remember|temp|tool_\w+|query|feeling|probe|text|source|method|size|overlap|prefix|k|m\d+)$")
for k in [k for k in st.session_state.keys() if _KEEP.match(str(k))]:
    st.session_state[k] = st.session_state[k]

# lv숫자_이름.py 파일을 찾아서 페이지로 등록 (템플릿은 제외)
pages = {}
for path in sorted(HERE.glob("lv*_*.py"), key=lambda p: int(re.match(r"lv(\d+)_", p.name).group(1))
                   if re.match(r"lv(\d+)_", p.name) else 999):
    m = re.match(r"lv(\d+)_(.+)\.py$", path.name)
    if not m:
        continue
    n = int(m.group(1))
    title, icon = BUILTIN_TITLES.get(n, (m.group(2).replace("_", " "), "🧩"))
    pages[path.name] = st.Page(str(path), title=f"Lv{n} · {title}", icon=icon, default=(n == 1))

common.PAGES = pages  # 튜토리얼 완료 화면의 '다음 레벨로' 버튼이 사용

st.navigation(list(pages.values()), position="sidebar").run()
