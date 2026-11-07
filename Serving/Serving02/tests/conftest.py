import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 앱을 import 하기 전에 테스트용 설정을 넣습니다 (설정은 환경 변수에서 읽으므로)
os.environ.update({
    "API_KEYS": "admin:test-admin,demo:test-demo,tiny:test-tiny",
    "PUBLIC_DEMO_KEY": "test-demo",
    "LOG_LEVEL": "WARNING",
    "RATE_LIMIT_BURST": "10000",       # 속도 제한은 test_rate_limit 에서만 따로 확인합니다
})

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SAMPLES = np.load(ROOT / "samples" / "mnist_test_200.npz")


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def digit_canvas(i: int, size: int = 120, offset=(80, 80)) -> Image.Image:
    """MNIST 샘플을 280x280 '그림판' 위 임의 위치에 붙여, 사람이 그린 것처럼 만듭니다."""
    canvas = Image.new("L", (280, 280), 0)
    canvas.paste(Image.fromarray(SAMPLES["images"][i]).resize((size, size)), offset)
    return canvas


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:         # with 로 열어야 lifespan(모델 로드)이 실행됩니다
        yield c


@pytest.fixture
def admin():
    return {"X-API-Key": "test-admin"}
