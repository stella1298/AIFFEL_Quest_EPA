"""키마다 토큰 버킷으로 요청 속도를 제한합니다 (게이트웨이 노트의 TokenBucket).

버킷에는 최대 burst 개의 토큰이 있고 초당 rate/60 개씩 다시 찹니다.
요청 하나가 토큰 하나를 씁니다. 비어 있으면 429 + Retry-After(몇 초 뒤에 다시).

원래는 서버 '앞'의 게이트웨이(Nginx·클라우드 API Gateway)가 하는 일입니다.
이 프로젝트는 컨테이너 하나로도 외부에 공개할 수 있게 서버 안에 함께 넣었습니다.
서버를 여러 대 띄우면 버킷이 서버마다 따로 생기므로, 그때는 Redis 같은 공용 저장소로 옮깁니다.
"""
import math
import threading
import time

from fastapi import Depends, Request

from app.auth import verify_api_key
from app.config import settings
from app.errors import ApiError


class TokenBucket:
    def __init__(self, rate_per_min: int, burst: int):
        self.rate = rate_per_min / 60.0
        self.capacity = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self, cost: float = 1.0) -> float:
        """토큰을 쓰면 0, 모자라면 기다려야 할 초를 돌려줍니다."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= cost:
                self.tokens -= cost
                return 0.0
            return (cost - self.tokens) / self.rate


_buckets: dict[str, TokenBucket] = {}


def reset_buckets():
    _buckets.clear()


async def rate_limited_user(request: Request, user: str = Depends(verify_api_key)) -> str:
    """인증(401) 다음에 속도 제한(429)을 검사하는 의존성. 엔드포인트는 이것 하나만 붙입니다."""
    bucket = _buckets.setdefault(user, TokenBucket(settings.rate_per_min, settings.rate_burst))
    wait = bucket.take()
    if wait > 0:
        raise ApiError(429, "rate_limited", f"요청이 너무 많습니다. {math.ceil(wait)}초 뒤에 다시 시도해 주세요.",
                       headers={"Retry-After": str(math.ceil(wait))})
    return user
