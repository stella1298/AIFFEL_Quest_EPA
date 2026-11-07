"""이미지 → 모델 입력 변환. torch 없이 numpy·Pillow 만 씁니다 (서빙 이미지를 가볍게 하려고).

1) 안전하게 읽기   : 용량·픽셀 수·형식을 먼저 확인합니다 (인증과 화면 노트의 '안전한 이미지 처리')
2) MNIST 처럼 만들기 : 학습 데이터와 같은 모양이어야 모델이 제 실력을 냅니다
"""
import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.errors import ApiError

MEAN, STD = 0.1307, 0.3081
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF"}


def decode_base64(data: str) -> bytes:
    """'data:image/png;base64,....' 또는 순수 base64 문자열을 bytes 로."""
    try:
        return base64.b64decode(data.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid_base64", "base64 이미지를 해석할 수 없습니다.")


def open_image(raw: bytes, max_bytes: int, max_pixels: int) -> Image.Image:
    """바이트를 이미지로 엽니다. 디코딩 전에 크기부터 확인합니다."""
    if not raw:
        raise ApiError(400, "empty_file", "빈 파일입니다.")
    if len(raw) > max_bytes:
        raise ApiError(413, "too_large", f"파일이 너무 큽니다 (최대 {max_bytes // 1024}KB).")
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format
        w, h = img.size                      # 헤더만 읽은 상태. 아직 픽셀을 풀지 않았습니다
    except (UnidentifiedImageError, OSError):
        raise ApiError(415, "not_an_image", "이미지 파일이 아닙니다 (PNG·JPEG·WEBP·BMP·GIF).")
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(415, "unsupported_format", f"지원하지 않는 형식입니다 ({fmt}).")
    if w * h > max_pixels:
        # 작은 파일이 거대한 픽셀로 풀리는 '압축 폭탄'을 여기서 막습니다
        raise ApiError(413, "too_many_pixels", f"이미지 해상도가 너무 큽니다 ({w}x{h}).")
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))   # 투명 부분은 검은색(=배경)으로
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def to_mnist(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """흑백 이미지를 MNIST 형식(검은 배경·흰 글씨, 20px 맞춤, 무게중심 정렬)으로 바꿉니다.

    반환 : (모델 입력 float32 (1,1,28,28), 미리보기 uint8 (28,28))
    이 단계를 빼고 그냥 28x28 로 줄이면, 작게 그리거나 구석에 그린 숫자를 잘 못 맞힙니다.
    """
    a = np.asarray(img, dtype=np.float32)
    if a.mean() > 127:                          # 흰 종이에 검은 글씨(사진·스캔) → 반전
        a = 255.0 - a
    a[a < 30] = 0                               # 옅은 잡티 제거
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        raise ApiError(400, "blank_image", "그림이 비어 있습니다. 숫자를 그려 주세요.")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]          # 글씨 부분만
    h, w = a.shape
    scale = 20.0 / max(h, w)
    small = Image.fromarray(a.astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    b = np.asarray(small, dtype=np.float32)
    bh, bw = b.shape
    # 무게중심이 (14,14) 에 오도록 놓을 자리를 계산합니다.
    # 주의 : np.roll 로 옮기면, 무게가 한쪽에 쏠린 글씨(두꺼운 윗변 + 가는 꼬리 등)는
    #        끝부분이 반대편 가장자리로 넘어갑니다. 그래서 자리를 계산한 뒤 28x28 안으로 범위를 제한합니다.
    #        (tests/test_api.py::test_preprocess_never_wraps_around)
    total = b.sum()
    cy = (b.sum(1) * np.arange(bh)).sum() / total
    cx = (b.sum(0) * np.arange(bw)).sum() / total
    top = int(np.clip(round(14 - cy), 0, 28 - bh))
    left = int(np.clip(round(14 - cx), 0, 28 - bw))
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + bh, left:left + bw] = b
    x = ((canvas / 255.0 - MEAN) / STD).astype(np.float32).reshape(1, 1, 28, 28)
    return x, canvas.clip(0, 255).astype(np.uint8)


def from_mnist_array(arr: np.ndarray) -> np.ndarray:
    """이미 28x28 MNIST 형식인 배열(0~255)을 모델 입력으로. 샘플 테스트용."""
    return ((arr.astype(np.float32) / 255.0 - MEAN) / STD).reshape(1, 1, 28, 28)


def preview_png(arr: np.ndarray, size: int = 112) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).resize((size, size), Image.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
