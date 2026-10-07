"""🔧 동료가 쓸 수 있는 도구 모음 (Lv2, Lv5에서 사용)

도구 = 설명(docstring)이 붙은 파이썬 함수.
LLM은 함수 '코드'는 못 보고, 이름·설명·인자만 보고 언제 쓸지 판단한다.
→ 새 도구를 직접 만들어 TOOLS에 추가해 보세요!
"""

import ast
import operator
import random
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

NOTES_DIR = Path(__file__).parent / "my_notes"
MEMO_FILE = NOTES_DIR / "메모장.md"


@tool
def get_now() -> str:
    """지금 날짜, 요일, 시각을 알려준다. 시간·날짜·요일을 물으면 사용한다."""
    now = datetime.now()
    weekday = "월화수목금토일"[now.weekday()]
    return now.strftime(f"%Y-%m-%d ({weekday}) %H:%M")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("숫자와 + - * / ** % 괄호만 쓸 수 있어요")


@tool
def calculate(expression: str) -> str:
    """수식을 정확하게 계산한다. 숫자 계산이 필요하면 암산하지 말고 반드시 이 도구를 쓴다.

    Args:
        expression: 계산할 수식. 예) "(12000 * 3) / 4", "2 ** 10"
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"계산 실패: {e}"


LUNCH_MENU = {
    "한식": ["김치찌개", "제육볶음", "순두부찌개", "비빔밥", "국밥"],
    "중식": ["짜장면", "짬뽕", "마라탕", "볶음밥"],
    "일식": ["돈카츠", "라멘", "초밥", "규동"],
    "양식": ["파스타", "햄버거", "샐러드", "피자"],
}


@tool
def pick_lunch(category: str = "아무거나") -> str:
    """점심 메뉴를 무작위로 하나 골라준다. 메뉴 추천이나 '뭐 먹지' 질문에 사용한다.

    Args:
        category: 한식, 중식, 일식, 양식, 아무거나 중 하나.
    """
    pool = LUNCH_MENU.get(category) or [m for ms in LUNCH_MENU.values() for m in ms]
    return random.choice(pool)


@tool
def roll_dice(sides: int = 6) -> int:
    """주사위를 굴린다. 운에 맡기거나 무작위로 정할 때 사용한다.

    Args:
        sides: 주사위 면의 수 (기본 6).
    """
    return random.randint(1, max(2, sides))


@tool
def save_memo(content: str) -> str:
    """사용자가 '기억해 줘', '메모해 줘'라고 한 내용을 메모장 파일에 저장한다.

    Args:
        content: 저장할 내용 (한 문장으로 요약).
    """
    NOTES_DIR.mkdir(exist_ok=True)
    if not MEMO_FILE.exists():
        MEMO_FILE.write_text("# 메모장\n\n동료가 대화 중에 저장한 메모\n", encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with MEMO_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n- [{stamp}] {content}\n")
    return f"메모장에 저장함: {content}"


TOOLS = [get_now, calculate, pick_lunch, roll_dice, save_memo]
