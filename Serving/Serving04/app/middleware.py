"""요청 로깅 미들웨어 (순수 ASGI).

BaseHTTPMiddleware 를 쓰지 않는 이유는 비동기 처리 노트에 적어 두었습니다.
스트리밍 도중 사용자가 떠날 때 정리 코드가 제때 돌지 않습니다.
"""
import logging
import time

logger = logging.getLogger("ml_api")


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        info = {"status": 0, "머리까지": None, "완료": False}

        async def send_and_watch(message):
            if message["type"] == "http.response.start":
                info["status"] = message["status"]
                info["머리까지"] = time.perf_counter() - started
                message["headers"].append(
                    (b"x-process-time", f"{info['머리까지']:.3f}".encode()))
            elif (message["type"] == "http.response.body"
                  and not message.get("more_body", False)):
                info["완료"] = True
            await send(message)

        label = f"{scope['method']} {scope['path']}"
        failed = None
        try:
            await self.app(scope, receive, send_and_watch)
        except Exception as exc:
            failed = type(exc).__name__
            raise
        finally:
            total = time.perf_counter() - started
            if failed:
                logger.error(f"{label} -> 예외 {failed} ({total:.2f}초)")
            elif info["머리까지"] is None:
                logger.warning(f"{label} -> 응답을 시작하지 못했습니다 ({total:.2f}초)")
            else:
                끝 = "완료" if info["완료"] else "중간에 끊김"
                line = (f"{label} -> {info['status']} "
                        f"(머리까지 {info['머리까지']:.2f}초, "
                        f"끝까지 {total:.2f}초, {끝})")
                if info["status"] >= 400 or not info["완료"]:
                    logger.warning(line)
                else:
                    logger.info(line)
