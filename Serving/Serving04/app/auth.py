"""API 키 인증. 인증과 화면 노트에서 만든 것에 부서 필드 하나를 더했습니다.

진짜 사내 서비스에서는 이 표가 사내 계정 시스템(LDAP, SSO)에서 옵니다.
키를 코드에 적어 두는 것은 실습에서만 하는 일이고, 사내에 올릴 때는
환경 변수나 시크릿 저장소에서 읽습니다.
"""
import json
import os

from fastapi import Header, HTTPException

# 기본값은 실습용입니다. 사내에서는 D52_API_KEYS 환경 변수로 덮어씁니다
_DEFAULT_KEYS = {
    "dev-key-001": {"사용자": "김개발", "부서": ["공개", "개발"]},
    "sales-key-002": {"사용자": "이영업", "부서": ["공개", "영업"]},
    "hr-key-003": {"사용자": "박인사", "부서": ["공개", "인사"]},
    "partner-key-004": {"사용자": "협력사 계정", "부서": ["공개"]},
}

API_KEYS = json.loads(os.environ["D52_API_KEYS"]) if os.environ.get("D52_API_KEYS") \
    else _DEFAULT_KEYS

async def verify_api_key(x_api_key: str = Header(None)) -> dict:
    """헤더의 API 키를 확인하고 사용자 정보를 돌려줍니다.

    돌려주는 것 : {"사용자": 이름, "부서": [열람 가능한 부서 목록]}
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API 키가 필요합니다. X-API-Key 헤더를 포함해 주십시오.",
        )
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")
    return API_KEYS[x_api_key]
