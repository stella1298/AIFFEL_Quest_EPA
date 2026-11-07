"""모든 요청에 요청 ID 와 처리 시간을 붙이고 한 줄 로그를 남깁니다 (비동기 처리 노트의 middleware·logger).

- X-Request-ID : 클라이언트가 보내면 그대로, 없으면 새로 만듭니다. 에러 응답에도 같은 ID 가 들어가
                 "이 요청 왜 실패했나요?"를 로그에서 바로 찾을 수 있습니다.
- X-Process-Time-ms : 서버 안에서 걸린 시간
"""
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger("mnist_api")


def setup_logging():
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
        response.headers["X-Model-Version"] = settings.model_version
        if request.url.path not in ("/health",):          # 헬스체크는 너무 잦아서 생략
            logger.info("rid=%s user=%s %s %s -> %s (%.0fms)", rid, getattr(request.state, "user", "-"),
                        request.method, request.url.path, response.status_code, ms)
        return response
