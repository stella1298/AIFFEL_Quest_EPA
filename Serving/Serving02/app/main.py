"""MNIST 손글씨 숫자 인식 API — 여섯 모델(로지스틱 회귀·MLP·LeNet-5·CNN·LSTM·ViT)을 함께 서빙하고,
투표 모델(기본 LeNet-5·CNN·LSTM)의 확률 평균으로 최종 답과 '사람 확인 필요' 여부를 정합니다.

엔드포인트
  GET  /                         그림판 화면 (브라우저)
  GET  /docs                     Swagger 문서 (자동 생성)
  GET  /health                   살아 있는지 (게이트웨이·Docker 헬스체크용, 인증 없음)
  GET  /ready                    모델까지 준비됐는지 (준비 전 503)
  GET  /v1/models                모델 정보 · 학습 지표 (인증 없음)
  POST /v1/predict               base64 이미지 1장            🔑
  POST /v1/predict/file          파일 업로드(multipart) 1장   🔑
  POST /v1/predict/batch         여러 장 한 번에              🔑
  POST /v1/predict/batch/stream  여러 장을 한 장씩 SSE 로 흘려보냄 🔑

요청이 지나는 순서 : 미들웨어(요청 ID) → 인증(401) → 속도 제한(429) → 입력 검증(422)
                   → 이미지 검사(400·413·415) → 동시 처리 자리(503) → 스레드풀에서 추론 → 200
"""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import preprocess as pp
from app.config import settings
from app.errors import ApiError, logger, register_error_handlers
from app.inference import ensemble, registry
from app.model_catalog import ORDER, info
from app.middleware import RequestContextMiddleware, setup_logging
from app.rate_limit import rate_limited_user
from app.schemas import (BatchItem, BatchPredictRequest, BatchPredictResponse, PredictRequest,
                         PredictResponse)

STATIC = Path(__file__).parent / "static"

# 추론 전용 스레드풀 : 동기 추론이 이벤트 루프를 막지 않게 (비동기 처리 노트의 run_in_executor)
executor = ThreadPoolExecutor(max_workers=settings.infer_workers, thread_name_prefix="infer")
# 동시에 처리할 요청 수 상한 ('슬롯'). 다 차면 queue_timeout 동안 기다렸다가 503
slots = asyncio.Semaphore(settings.max_inflight)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("모델 로드 시작 (backend=%s, models=%s)", settings.backend, settings.enabled_models)
    try:
        await asyncio.get_running_loop().run_in_executor(executor, registry.load)
        logger.info("모델 로드 완료 (%.2f초)", registry.load_seconds)
    except Exception:
        # 로드에 실패해도 서버는 뜹니다. /ready 가 503 으로 알려 줍니다
        logger.exception("모델 로드 실패")
    yield
    executor.shutdown(wait=False, cancel_futures=True)
    logger.info("서버 종료")


app = FastAPI(
    title="MNIST 손글씨 숫자 인식 API",
    description="손글씨 숫자 이미지를 **CNN(합성곱)** 과 **MLP(다층 퍼셉트론)** 로 동시에 분류해 비교합니다. "
                "🔑 표시 엔드포인트는 `X-API-Key` 헤더가 필요합니다.",
    version=settings.model_version,
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
                   expose_headers=["X-Request-ID", "X-Process-Time-ms", "X-Model-Version"])
register_error_handlers(app)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ===== 공통 =====
def _require_ready():
    if not registry.ready:
        raise ApiError(503, "model_loading", "모델을 불러오는 중입니다. 잠시 후 다시 시도해 주세요.",
                       headers={"Retry-After": "5"})


async def run_in_slot(fn, *args):
    """자리(슬롯)를 얻은 뒤 스레드풀에서 fn 을 실행합니다. 자리가 안 나면 503."""
    try:
        await asyncio.wait_for(slots.acquire(), timeout=settings.queue_timeout_sec)
    except asyncio.TimeoutError:
        raise ApiError(503, "server_busy", "요청이 몰려 있습니다. 잠시 후 다시 시도해 주세요.",
                       headers={"Retry-After": "1"})
    try:
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)
    finally:
        slots.release()


def _predict_raw(raw: bytes, models: list[str], want_preview: bool) -> dict:
    """(스레드에서 실행) 바이트 → 검사 → MNIST 형식 → 추론."""
    img = pp.open_image(raw, settings.max_upload_bytes, settings.max_pixels)
    x, preview = pp.to_mnist(img)
    results = {k: v[0] for k, v in registry.predict(x, models).items()}
    ens = ensemble(results, list(settings.ensemble_models))
    # 확인 필요 : 평균 확률이 낮거나(모두 망설임), 최종 답에 동의한 모델이 적을 때(의견이 갈림)
    review = ens["confidence"] < settings.review_threshold or ens["agreement"] < settings.min_agreement
    agree = len({r["pred"] for r in results.values()}) == 1
    return {"results": results, "ensemble": ens, "agree": agree, "needs_review": review,
            "preview": pp.preview_png(preview) if want_preview else None}


def _predict_b64(b64: str, models: list[str], want_preview: bool) -> dict:
    # base64 해석도 CPU 작업이라 이벤트 루프가 아니라 스레드에서 합니다
    return _predict_raw(pp.decode_base64(b64), models, want_preview)


def _response(request: Request, user: str, out: dict) -> dict:
    return {"success": True, "request_id": request.state.request_id, "user": user,
            "backend": settings.backend, "model_version": settings.model_version, **out}


def _resolve_models(models: list[str] | None) -> list[str]:
    """요청에 모델을 안 적으면 서버에 켜진 모델 전부. 적었는데 꺼진 모델이 있으면 400."""
    if not models:
        return list(registry.models)
    missing = [m for m in models if m not in registry.models]
    if missing:
        raise ApiError(400, "model_not_enabled", f"이 서버에서 켜지지 않은 모델입니다 : {missing}")
    return models


# ===== 화면·상태 =====
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["System"], summary="살아 있는지 (liveness)")
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["System"], summary="모델까지 준비됐는지 (readiness)")
async def ready():
    if not registry.ready:
        return JSONResponse({"status": "loading"}, 503)
    return {"status": "ready", "backend": settings.backend, "models": list(registry.models),
            "loaded_at": registry.loaded_at, "load_seconds": registry.load_seconds}


@app.get("/ui-config", include_in_schema=False)
async def ui_config():
    # 공개 데모 키를 화면에 내려 줍니다. 키가 브라우저에 보이므로 '속도 제한이 걸린 데모 키'만 씁니다
    return {"demo_key": settings.public_demo_key or None, "review_threshold": settings.review_threshold,
            "min_agreement": settings.min_agreement}


@app.get("/v1/models", tags=["System"], summary="모델 정보와 학습 지표")
async def models_info():
    m = registry.metrics
    trained = m.get("models", {})
    keys = [k for k in ORDER if k in registry.models]          # 화면에 보여 줄 순서 = 카탈로그 순서
    return {"backend": settings.backend, "model_version": settings.model_version,
            "enabled": keys, "review_threshold": settings.review_threshold,
            "min_agreement": settings.min_agreement,
            "ensemble_members": [k for k in settings.ensemble_models if k in registry.models],
            "trained_at": m.get("trained_at"), "environment": m.get("environment"),
            "ensemble": m.get("ensemble"), "ensemble_all": m.get("ensemble_all"),
            "ensemble_search": m.get("ensemble_search"),
            "models": {k: {**info(k), **{kk: trained.get(k, {}).get(kk) for kk in (
                "params", "epochs", "train_sec", "test_acc", "shift4px_acc", "rotate20_acc",
                "file_bytes", "latency_ms")}} for k in keys}}


# ===== 추론 =====
@app.post("/v1/predict", response_model=PredictResponse, tags=["Prediction"],
          summary="🔑 base64 이미지 1장 분류")
async def predict(req: PredictRequest, request: Request, user: str = Depends(rate_limited_user)):
    _require_ready()
    models = _resolve_models(req.models)
    out = await run_in_slot(_predict_b64, req.image, models, req.return_preview)
    return _response(request, user, out)


@app.post("/v1/predict/file", response_model=PredictResponse, tags=["Prediction"],
          summary="🔑 이미지 파일 업로드(multipart) 1장 분류")
async def predict_file(request: Request, file: UploadFile = File(..., description="PNG/JPEG 등"),
                       models: list[str] | None = Query(default=None, description="비우면 켜진 모델 전부"),
                       user: str = Depends(rate_limited_user)):
    _require_ready()
    models = _resolve_models(models)
    raw = await file.read(settings.max_upload_bytes + 1)   # 한도+1 만 읽어 큰 파일을 끝까지 받지 않음
    out = await run_in_slot(_predict_raw, raw, models, True)
    return _response(request, user, out)


def _predict_batch(images: list[str], models: list[str]) -> list[dict]:
    items = []
    for i, b64 in enumerate(images):
        try:
            out = _predict_raw(pp.decode_base64(b64), models, False)
            items.append({"index": i, "results": out["results"], "ensemble": out["ensemble"]})
        except ApiError as e:                      # 한 장이 잘못돼도 나머지는 계속
            items.append({"index": i, "error": e.message})
    return items


def _accuracy(items: list[dict], labels: list[int] | None, models: list[str]):
    if not labels:
        return None
    done = [it for it in items if it.get("results")]
    acc = {}
    for m in models:
        ok = [it["results"][m]["pred"] == labels[it["index"]] for it in done]
        acc[m] = round(sum(ok) / len(ok), 4) if ok else 0.0
    ok = [it["ensemble"]["pred"] == labels[it["index"]] for it in done]
    acc["ensemble"] = round(sum(ok) / len(ok), 4) if ok else 0.0
    return acc


@app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"],
          summary="🔑 여러 장 한 번에 분류")
async def predict_batch(req: BatchPredictRequest, request: Request, user: str = Depends(rate_limited_user)):
    _require_ready()
    models = _resolve_models(req.models)
    t0 = time.perf_counter()
    items = await run_in_slot(_predict_batch, req.images, models)
    for it in items:
        it["label"] = req.labels[it["index"]] if req.labels else None
    return {"success": True, "request_id": request.state.request_id, "user": user, "count": len(items),
            "items": [BatchItem(**it) for it in items], "accuracy": _accuracy(items, req.labels, models),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/v1/predict/batch/stream", tags=["Prediction"],
          summary="🔑 여러 장을 한 장씩 결과가 나오는 대로 보내기 (Server-Sent Events)")
async def predict_batch_stream(req: BatchPredictRequest, request: Request,
                               user: str = Depends(rate_limited_user)):
    """결과를 다 모을 때까지 기다리지 않고 한 장씩 `event: item` 으로 보냅니다.
    클라이언트가 연결을 끊으면 남은 추론을 멈춥니다 (비동기 처리 노트의 '취소')."""
    _require_ready()
    models = _resolve_models(req.models)

    async def events():
        done = []
        t0 = time.perf_counter()
        yield f"event: start\ndata: {json.dumps({'count': len(req.images), 'models': models})}\n\n"
        for i, b64 in enumerate(req.images):
            if await request.is_disconnected():
                logger.info("rid=%s 클라이언트 연결 끊김 → %d/%d 에서 중단",
                            request.state.request_id, i, len(req.images))
                return
            item = (await run_in_slot(_predict_batch, [b64], models))[0]
            item["index"] = i
            item["label"] = req.labels[i] if req.labels else None
            done.append(item)
            yield f"event: item\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        summary = {"count": len(done), "accuracy": _accuracy(done, req.labels, models),
                   "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        yield f"event: done\ndata: {json.dumps(summary)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
