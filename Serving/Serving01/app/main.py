# [3.5 app/main.py] FastAPI 서버 조립
#   lifespan 에서 모델 1회 로드 / Depends(verify_api_key) 인증 / run_in_executor 비동기 추론
#   상태 코드 : 200 정상, 401 키 없음·틀림, 422 입력 위반, 503 모델 로드 전, 500 내부 오류(정보 숨김)
#   (startup 이벤트 대신 lifespan 을 썼습니다. 둘 다 가능)
"""
뉴스 섹션 자동 분류 FastAPI 서버.

엔드포인트
  GET  /health          서버/모델 상태 (인증 없음 : 게이트웨이·모니터링용)
  GET  /labels          분류 가능한 섹션 목록 (인증 없음)
  POST /predict         제목 한 건 분류 (API Key 필요)
  POST /predict/batch   제목 여러 건 일괄 분류 (API Key 필요)

상태 코드
  200 정상 / 401 API Key 없음·틀림 / 422 입력 규격 위반 / 503 모델 준비 전 / 500 추론 실패
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.auth import verify_api_key
from app.model_service import (MODEL_ID, MODEL_REVISION, REVIEW_THRESHOLD,
                               load_model, predict, predict_batch)
from app.schemas import (SECTIONS, BatchPredictRequest, BatchPredictResponse,
                         PredictRequest, PredictResponse)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [news_api] %(message)s")
logger = logging.getLogger("news_api")

# 추론 전용 스레드풀 : 무거운 동기 추론이 이벤트 루프를 막지 않게 합니다
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="news-infer")

STATE = {"model": None, "loaded_at": None, "load_seconds": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 : 모델을 한 번만 올립니다. 다운로드·로드도 무거워서 스레드에서 돌립니다
    logger.info(f"모델 로드 시작 : {MODEL_ID} ({MODEL_REVISION})")
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        STATE["model"] = await loop.run_in_executor(inference_executor, load_model)
        STATE["load_seconds"] = round(time.perf_counter() - start, 1)
        STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"모델 로드 완료 ({STATE['load_seconds']}초)")
    except Exception:
        # 로드 실패로 서버가 죽지 않게 합니다. /health 가 loading 을, /predict 가 503 을 알립니다
        logger.exception("모델 로드 실패")
    yield
    # 종료 : 스레드풀 정리
    inference_executor.shutdown(wait=False, cancel_futures=True)
    logger.info("서버 종료")


app = FastAPI(
    title="뉴스 섹션 자동 분류 API",
    description=(
        "뉴스 제목을 받아 KLUE-YNAT 7개 섹션(IT과학·경제·사회·생활문화·세계·스포츠·정치) "
        "중 하나로 분류합니다. 확률이 낮은 제목은 `needs_review` 로 표시해 편집자 확인을 돕습니다."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ===== 미들웨어 : 처리 시간 헤더 + 요청 로그 =====
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed:.1f}"
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.0f}ms)")
    return response


# ===== 에러 핸들러 =====
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 응답을 사람이 읽기 쉬운 형태로 바꿉니다. (어느 필드가 왜 틀렸는지)"""
    errors = [
        {
            # 본문이 JSON 이 아니면 loc 이 ("body", 위치) 라서 필드 이름 대신 body 로 표시합니다
            "field": "body" if e["type"] == "json_invalid"
            else ".".join(str(p) for p in e["loc"][1:]) or "body",
            "message": e["msg"],
        }
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "입력값이 올바르지 않습니다.", "details": errors},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """예상 못 한 에러 : 상세는 로그에만 남기고 요청자에게는 내부 정보를 숨깁니다."""
    logger.exception(f"처리되지 않은 에러 : {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "서버 내부 오류가 발생했습니다."},
    )


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="모델을 불러오는 중입니다. 잠시 후 다시 시도해 주세요.")
    return STATE["model"]


# ===== 엔드포인트 =====
@app.get("/health", tags=["System"])
async def health():
    ready = STATE["model"] is not None
    return {
        "status": "healthy" if ready else "loading",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "loaded_at": STATE["loaded_at"],
        "load_seconds": STATE["load_seconds"],
    }


@app.get("/labels", tags=["System"])
async def labels():
    return {"sections": SECTIONS, "review_threshold": REVIEW_THRESHOLD}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_one(req: PredictRequest, user: str = Depends(verify_api_key)):
    """뉴스 제목 한 건을 분류합니다."""
    model = _require_model()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(inference_executor, predict, model, req.title, req.top_k)
    except Exception as e:
        logger.exception("추론 실패")
        raise HTTPException(status_code=500, detail=f"추론에 실패했습니다 ({type(e).__name__}).")
    return PredictResponse(success=True, user=user, **result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_many(req: BatchPredictRequest, user: str = Depends(verify_api_key)):
    """뉴스 제목 여러 건(최대 20건)을 한 번에 분류합니다."""
    model = _require_model()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(inference_executor, predict_batch, model, req.titles)
    except Exception as e:
        logger.exception("일괄 추론 실패")
        raise HTTPException(status_code=500, detail=f"추론에 실패했습니다 ({type(e).__name__}).")
    return BatchPredictResponse(success=True, user=user, **result)
