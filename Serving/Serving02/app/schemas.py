"""요청·응답 규격 (FastAPI 기초 노트의 Pydantic).

규격을 어기면 모델에 닿기 전에 FastAPI 가 422 로 거절합니다.
응답 모델은 /docs(Swagger) 에 그대로 문서로 나옵니다.
"""
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.model_catalog import CATALOG, ORDER

_MODELS_DESC = f"예측할 모델 ({', '.join(ORDER)}). 비우면 서버에 켜진 모델 전부"


def _check_model_names(v):
    if v is None:
        return v
    unknown = [m for m in v if m not in CATALOG]
    if unknown:
        raise ValueError(f"알 수 없는 모델 : {unknown}. 가능한 값 : {ORDER}")
    return list(dict.fromkeys(v))               # 중복 제거, 순서 유지


class PredictRequest(BaseModel):
    image: str = Field(..., min_length=20, description="PNG/JPEG 를 base64 로 (data URL 형식도 가능)")
    models: list[str] | None = Field(default=None, min_length=1, max_length=len(CATALOG), description=_MODELS_DESC)
    return_preview: bool = Field(default=True, description="모델이 실제로 본 28x28 이미지를 돌려줄지")

    @field_validator("models")
    @classmethod
    def known_models(cls, v):
        return _check_model_names(v)

    model_config = {"json_schema_extra": {"examples": [{
        "image": "data:image/png;base64,iVBORw0KGgo...", "models": ["cnn", "lstm", "vit"], "return_preview": True}]}}


class BatchPredictRequest(BaseModel):
    images: list[str] = Field(..., min_length=1, max_length=settings.max_batch,
                              description=f"base64 이미지 목록 (1~{settings.max_batch}장)")
    models: list[str] | None = Field(default=None, min_length=1, max_length=len(CATALOG), description=_MODELS_DESC)
    labels: list[int] | None = Field(default=None, description="정답을 알면 함께 보내 정확도를 받습니다")

    @field_validator("models")
    @classmethod
    def known_models(cls, v):
        return _check_model_names(v)

    @field_validator("labels")
    @classmethod
    def check_labels(cls, v, info):
        if v is not None:
            if len(v) != len(info.data.get("images", [])):
                raise ValueError("labels 는 images 와 개수가 같아야 합니다.")
            if any(not 0 <= x <= 9 for x in v):
                raise ValueError("labels 는 0~9 사이여야 합니다.")
        return v


class ModelResult(BaseModel):
    pred: int = Field(description="예측한 숫자")
    confidence: float = Field(description="그 숫자의 확률 (0~1)")
    probs: list[float] = Field(description="0~9 각각의 확률")
    infer_ms: float = Field(description="이 모델의 추론 시간 (이미지 1장, ms)")


class EnsembleResult(BaseModel):
    pred: int = Field(description="최종 답 : 투표 모델들의 확률을 평균(soft voting)해 가장 높은 숫자")
    confidence: float = Field(description="최종 답의 평균 확률")
    probs: list[float] = Field(description="0~9 각각의 평균 확률")
    votes: dict[str, int] = Field(description="숫자별로 그 숫자를 고른 모델 수")
    agreement: float = Field(description="투표 모델 중 최종 답과 같은 숫자를 고른 비율 (0~1)")
    members: list[str] = Field(description="투표에 참여한 모델")


class PredictResponse(BaseModel):
    success: bool = True
    request_id: str | None
    user: str
    results: dict[str, ModelResult] = Field(description="모델별 결과")
    ensemble: EnsembleResult
    agree: bool = Field(description="요청한 모든 모델(참고 모델 포함)이 같은 숫자를 골랐는지")
    needs_review: bool = Field(description="투표 모델의 평균 확률이 기준보다 낮거나 의견이 갈려 사람이 확인할 필요가 있는지")
    preview: str | None = Field(default=None, description="모델 입력 28x28 미리보기 (PNG data URL)")
    backend: str
    model_version: str


class BatchItem(BaseModel):
    index: int
    label: int | None = None
    results: dict[str, ModelResult] | None = None
    ensemble: EnsembleResult | None = None
    error: str | None = None


class BatchPredictResponse(BaseModel):
    success: bool = True
    request_id: str | None
    user: str
    count: int
    items: list[BatchItem]
    accuracy: dict[str, float] | None = Field(default=None, description="labels 를 보냈을 때 모델별·앙상블 정확도")
    elapsed_ms: float
