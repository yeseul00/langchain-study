"""2단계에서 쓰는 도구 모음.

실제 회사라면 DB·사내 API를 조회하겠지만, 여기서는 개념에 집중하려고
파이썬 딕셔너리를 '가짜 DB'로 쓴다.
"""

from datetime import datetime

from langchain_core.tools import tool

# 가짜 DB: 서비스 → 담당 정보
SERVICES = {
    "payment-service": {"team": "결제팀", "oncall": "@kim-payments", "tier": "critical"},
    "order-service": {"team": "주문팀", "oncall": "@lee-orders", "tier": "critical"},
    "search-service": {"team": "검색팀", "oncall": "@park-search", "tier": "high"},
    "notification-service": {"team": "플랫폼팀", "oncall": "@choi-platform", "tier": "medium"},
}


@tool
def get_service_owner(service_name: str) -> dict:
    """서비스 이름으로 담당 팀, 온콜(당직) 담당자, 중요도 등급을 조회한다.

    Args:
        service_name: 서비스 이름. 예) payment-service, order-service,
            search-service, notification-service
    """
    info = SERVICES.get(service_name)
    if info is None:
        return {"error": f"'{service_name}' 서비스를 찾을 수 없음", "known_services": list(SERVICES)}
    return {"service": service_name, **info}


@tool
def get_current_time() -> str:
    """현재 날짜와 시각을 'YYYY-MM-DD HH:MM' 형식으로 돌려준다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


TOOLS = [get_service_owner, get_current_time]
