"""에러 응답을 한 가지 모양으로 통일합니다 (비동기 처리 노트의 error_handlers).

모든 실패 응답 : {"success": false, "error": {"code": "...", "message": "..."}, "request_id": "..."}
화면은 code 로 분기하고, message 는 사람에게 그대로 보여 줍니다.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("mnist_api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def _body(request: Request, code: str, message: str, **extra):
    return {"success": False, "error": {"code": code, "message": message, **extra},
            "request_id": getattr(request.state, "request_id", None)}


def register_error_handlers(app: FastAPI):
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(_body(request, exc.code, exc.message), exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI 기본 422 는 loc/type/input 이 섞여 있어 화면에 보여 주기 어렵습니다 → 필드별 한 줄로
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": e["msg"]}
                   for e in exc.errors()]
        return JSONResponse(_body(request, "validation_error", "입력값이 올바르지 않습니다.",
                                  details=details), 422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(_body(request, f"http_{exc.status_code}", str(exc.detail)),
                            exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # 상세 원인은 서버 로그에만. 응답에는 내부 정보를 노출하지 않습니다
        logger.exception("처리되지 않은 오류 rid=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(_body(request, "internal_error", "서버 내부 오류가 발생했습니다."), 500)
