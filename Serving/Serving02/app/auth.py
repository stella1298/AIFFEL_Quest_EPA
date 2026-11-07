"""API 키 인증 (인증과 화면 노트 → auth_hashed 방식).

- 키 원문은 서버가 켜질 때 SHA-256 해시로 바꾸고 원문은 메모리에 두지 않습니다.
- 비교는 hmac.compare_digest 로 합니다 (비교 시간으로 키를 추측하는 공격 방지).
- 키가 없거나 틀리면 401. 모델에 닿기 전에 막습니다.
"""
import hashlib
import hmac

from fastapi import Header, Request

from app.config import settings
from app.errors import ApiError


def _h(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


KEY_HASHES: dict[str, str] = {_h(k): owner for k, owner in settings.api_keys.items()}


async def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    if not x_api_key:
        raise ApiError(401, "missing_api_key", "API 키가 필요합니다. X-API-Key 헤더를 넣어 주세요.",
                       headers={"WWW-Authenticate": "ApiKey"})
    digest = _h(x_api_key)
    owner = next((o for h, o in KEY_HASHES.items() if hmac.compare_digest(h, digest)), None)
    if owner is None:
        raise ApiError(401, "invalid_api_key", "유효하지 않은 API 키입니다.",
                       headers={"WWW-Authenticate": "ApiKey"})
    request.state.user = owner          # 로그·레이트 리밋이 '누가' 불렀는지 알 수 있게
    return owner
