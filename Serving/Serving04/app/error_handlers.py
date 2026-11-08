"""글로벌 에러 핸들러. 상세는 로그에, 요청자에게는 한 줄."""
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ml_api")


def register_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.error(
            f"처리하지 못한 예외 : {type(exc).__name__}: {exc}\n"
            f"  경로 : {request.method} {request.url.path}\n"
            f"{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "서버 내부 오류가 발생했습니다"},
        )
