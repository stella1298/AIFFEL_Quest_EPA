"""정형 데이터 예측 서비스 (선택 트랙).

3장의 질의응답 서버와 구조가 같습니다. 모델이 가벼워서 스트리밍도
동시 통과 제한도 필요 없습니다. 그래서 층이 몇 개 빠졌을 뿐입니다.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.error_handlers import register_error_handlers
from app.logger_config import setup_logger
from app.middleware import RequestLoggingMiddleware

logger = setup_logger("ml_api")

MODEL_PATH = Path(os.environ.get("D52_TABULAR_MODEL", "models/housing_pipeline.joblib"))
META_PATH = Path(os.environ.get("D52_TABULAR_META", "models/housing_meta.json"))

STATE = {"준비됨": False, "mae": None}
MODEL = {"pipe": None, "features": []}

class PriceRequest(BaseModel):
    """주택 정보. 범위를 벗어난 값은 여기서 막힙니다."""
    MedInc: float = Field(..., gt=0, le=20, description="중위 소득")
    HouseAge: float = Field(..., ge=0, le=100, description="주택 연식(년)")
    AveRooms: float = Field(..., gt=0, le=50, description="평균 방 수")
    AveBedrms: float = Field(..., gt=0, le=20, description="평균 침실 수")
    Population: float = Field(..., gt=0, le=50000, description="인구")
    AveOccup: float = Field(..., gt=0, le=20, description="평균 거주 인원")
    Latitude: float = Field(..., ge=32, le=42, description="위도 (캘리포니아 범위)")
    Longitude: float = Field(..., ge=-125, le=-114, description="경도 (캘리포니아 범위)")

    model_config = {
        "json_schema_extra": {
            "examples": [{"MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 5.0,
                          "AveBedrms": 1.0, "Population": 1500.0, "AveOccup": 3.0,
                          "Latitude": 37.5, "Longitude": -122.0}]
        }
    }

class PriceResponse(BaseModel):
    success: bool = True
    predicted_price: float = Field(description="예측 가격 (10만 달러 단위)")
    predicted_price_usd: int = Field(description="예측 가격 (달러)")
    user: str = Field(description="인증된 사용자")
    model_mae: float | None = Field(default=None, description="이 모델의 테스트 MAE")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists() and META_PATH.exists():
        MODEL["pipe"] = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        MODEL["features"] = meta["features"]
        STATE["mae"] = meta.get("mae")
        STATE["준비됨"] = True
        logger.info(f"모델 로드 완료. 피처 {len(MODEL['features'])}개, MAE {STATE['mae']}")
    else:
        # 실패를 예외로 만들지 않습니다. 서버는 뜨고 헬스체크가 loading 을 알립니다
        logger.error(f"모델 파일이 없습니다 : {MODEL_PATH}")
    yield
    STATE["준비됨"] = False


app = FastAPI(title="주택 가격 예측", version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy" if STATE["준비됨"] else "loading", **STATE}

# 추론이 1밀리초 미만이라 def 로 둡니다. 비동기 처리 노트의 판단 기준 그대로입니다
@app.post("/predict", response_model=PriceResponse, tags=["Prediction"])
def predict(req: PriceRequest, user: dict = Depends(verify_api_key)):
    if MODEL["pipe"] is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    데이터 = req.model_dump()
    # 피처 순서를 모델이 학습한 순서에 맞춥니다. 순서가 어긋나면 조용히 틀립니다
    행 = np.array([[데이터[이름] for 이름 in MODEL["features"]]], dtype=np.float64)
    값 = float(MODEL["pipe"].predict(행)[0])
    값 = max(값, 0.0)

    return PriceResponse(
        predicted_price=round(값, 4),
        predicted_price_usd=int(값 * 100000),
        user=user["사용자"],
        model_mae=STATE["mae"],
    )
