"""설정은 전부 환경 변수에서 읽습니다.

같은 코드가 내 PC · Docker · 클라우드 VM · Hugging Face Spaces 에서 그대로 돌아야 하므로
주소·포트·키·정책 값을 코드에 직접 적지 않습니다. (서빙 프로젝트 노트의 원칙)
기본값은 '내 PC 에서 바로 실행' 기준입니다. 운영에서는 .env 로 덮어씁니다.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.model_catalog import DEFAULT_ENSEMBLE, ORDER

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get(name, default)


def _parse_keys(raw: str) -> dict[str, str]:
    """'이름:키,이름:키' → {키: 이름}"""
    keys = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        owner, _, key = pair.partition(":")
        if owner and key:
            keys[key.strip()] = owner.strip()
    return keys


@dataclass(frozen=True)
class Settings:
    # 모델
    model_dir: Path = Path(_env("MODEL_DIR", str(ROOT / "models")))
    backend: str = _env("MODEL_BACKEND", "onnx")                 # onnx | torchscript | pytorch
    enabled_models: tuple = tuple(_env("ENABLED_MODELS", ",".join(ORDER)).split(","))
    # 최종 답(확률 평균) 투표에 참여할 모델. 나머지 켜진 모델은 결과만 함께 보여 줍니다
    ensemble_models: tuple = tuple(_env("ENSEMBLE_MODELS", ",".join(DEFAULT_ENSEMBLE)).split(","))
    model_version: str = _env("MODEL_VERSION", "1.1.0")
    # '확인 필요' 판단 : 앙상블(확률 평균)의 확률이 이보다 낮거나,
    review_threshold: float = float(_env("REVIEW_THRESHOLD", "0.7"))
    #                  투표 모델 중 최종 답에 동의한 비율이 이보다 낮으면 (3개 중 2개 = 0.67 은 통과, 6개 중 3개 = 0.5 는 확인)
    min_agreement: float = float(_env("MIN_AGREEMENT", "0.66"))

    # 인증 : 'admin:키,demo:키' 형태. 서버는 받는 즉시 해시만 남기고 원문은 버립니다
    api_keys: dict = field(default_factory=lambda: _parse_keys(
        _env("API_KEYS", "admin:dev-admin-key,demo:demo-key")))
    # 브라우저 그림판에 자동으로 채워 줄 키 (공개 데모용. 비우면 화면에서 직접 입력)
    public_demo_key: str = _env("PUBLIC_DEMO_KEY", "demo-key")

    # 동시성 : 추론 스레드 수 / 동시에 처리할 요청 수 / 자리가 날 때까지 기다릴 시간
    infer_workers: int = int(_env("INFER_WORKERS", "2"))
    max_inflight: int = int(_env("MAX_INFLIGHT", "8"))
    queue_timeout_sec: float = float(_env("QUEUE_TIMEOUT_SEC", "2.0"))

    # 레이트 리밋 (키마다 토큰 버킷) : 분당 요청 수와 순간 허용량
    rate_per_min: int = int(_env("RATE_LIMIT_PER_MIN", "120"))
    rate_burst: int = int(_env("RATE_LIMIT_BURST", "20"))

    # 입력 이미지 제한
    max_upload_bytes: int = int(_env("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))   # 2MB
    max_pixels: int = int(_env("MAX_PIXELS", str(4_000_000)))                     # 압축 폭탄 방지
    max_batch: int = int(_env("MAX_BATCH", "32"))

    # 외부에서 다른 주소의 화면이 부를 수 있게 허용할 출처 (쉼표 구분, * 는 전체)
    cors_origins: tuple = tuple(filter(None, _env("CORS_ORIGINS", "*").split(",")))
    log_level: str = _env("LOG_LEVEL", "INFO")


settings = Settings()
