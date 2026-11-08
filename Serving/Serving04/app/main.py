# 실습 3 : 서버 조립 (POST /ask, POST /ask/sse)
"""사내 문서 질의응답 서버.

참고 : FastAPI 기초 노트의 경로 장, 비동기 처리 노트 7장의 스트리밍 서버.
"""
import os

from contextlib import aclosing, asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from app.auth import verify_api_key
from app.error_handlers import register_error_handlers
from app.generation import STATS, guarded_tokens, new_budget, new_report, sse
from app.inference import load_docs
from app.logger_config import setup_logger
from app.middleware import RequestLoggingMiddleware
from app.schemas import AskRequest, AskResponse
from app.service import in_scope, resolve_docs, to_sources

logger = setup_logger("ml_api")

STATE = {"준비됨": False, "문서 수": 0}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 뜰 때. 진짜 서비스라면 여기서 모델과 어댑터를 올립니다
    logger.info("사내 문서를 읽습니다")
    STATE["문서 수"] = len(load_docs())
    STATE["준비됨"] = True
    logger.info(f"준비 완료. 문서 {STATE['문서 수']}건")
    yield
    STATE["준비됨"] = False
    logger.info("서버를 내립니다")

app = FastAPI(
    title="사내 문서 질의응답",
    description="사내망 안에서만 도는 질의응답 서비스입니다. 부서별로 열람 범위가 다릅니다.",
    version="1.0.0",
    lifespan=lifespan,
    # 사내 서버에서는 문서 공개 범위를 환경 변수로 닫습니다
    docs_url=os.environ.get("D52_DOCS_URL", "/docs"),
)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

def _prepare(req, user):
    """근거를 고르고 마지막으로 한 번 더 확인합니다."""
    # 1) resolve_docs 로 열람 범위 안에서 근거 문서를 고릅니다
    docs = resolve_docs(req.question, user)
    # 2) in_scope 로 한 번 더 확인합니다. 범위 밖이면 500 을 내지 말고 근거를 비운 채 진행합니다
    #    (권한 문제를 사용자 잘못으로 만들지 않습니다)
    if not in_scope(docs, user):
        # 여기까지 왔다는 것은 우리 코드에 구멍이 있다는 뜻입니다.
        # 요청자를 탓하지 말고, 근거를 비우고 기록을 남깁니다
        logger.error(f"열람 범위 밖 문서가 검색되었습니다 : user={user.get('사용자')}")
        docs = []
    return docs, new_budget(req.max_tokens, req.deadline_seconds)


@app.get("/health", tags=["System"])
async def health():
    """게이트웨이가 부르는 경로. 절대 무거운 일을 하지 않습니다."""
    return {"status": "healthy" if STATE["준비됨"] else "loading", **STATE}


@app.get("/stats", tags=["System"])
async def stats():
    """운영 지표. 방금 무슨 일이 있었는지 봅니다."""
    return STATS

@app.post("/ask", response_model=AskResponse, tags=["Inference"])
async def ask(req: AskRequest, request: Request,
              user: dict = Depends(verify_api_key)):
    """단일 응답. 답을 한 번에 돌려줍니다."""
    docs, budget = _prepare(req, user)
    report = new_report()
    parts = []
    # 1) aclosing 으로 감쌉니다. 감싸지 않으면 안쪽 제너레이터가
    #    자동으로 닫히지 않습니다 (비동기 처리 노트 7.5절)
    async with aclosing(
            guarded_tokens(req.question, docs, request, budget, report)) as stream:
        # 2) 토큰을 하나씩 받아 parts 에 모읍니다
        async for token in stream:
            parts.append(token)
    return AskResponse(
        success=True,
        answer="".join(parts),
        sources=to_sources(docs),
        user=user["사용자"],
        report=report,
    )

@app.post("/ask/sse", tags=["Inference"])
async def ask_sse(req: AskRequest, request: Request,
                  user: dict = Depends(verify_api_key)):
    """스트리밍 응답. 만들어지는 대로 내보냅니다."""
    docs, budget = _prepare(req, user)
    report = new_report()

    async def events():
        yield sse("start", {"question": req.question,
                            "sources": to_sources(docs),
                            "user": user["사용자"],
                            "예산": budget})
        try:
            async with aclosing(
                    guarded_tokens(req.question, docs, request, budget, report)) as stream:
                async for token in stream:
                    yield sse("token", {"text": token})
        except Exception as exc:
            logger.error(f"스트리밍 중 오류 : {type(exc).__name__}: {exc}")
            # 1) 스트리밍 도중의 예외는 상태 코드로 알릴 수 없습니다. 이미 200 이 나갔기 때문입니다
            yield sse("error", {"error": "생성 중 오류가 발생했습니다"})
            return
        yield sse("done", report)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ask/broken", tags=["System"])
async def ask_broken(req: AskRequest, user: dict = Depends(verify_api_key)):
    """일부러 터뜨립니다. 글로벌 핸들러가 받는지 확인하는 용도입니다."""
    raise RuntimeError(f"모델 가중치를 읽지 못했습니다 (질문 : {req.question})")
