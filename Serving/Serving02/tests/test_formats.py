"""직렬화 형식 3가지(state_dict · TorchScript · ONNX)가 같은 답을 내는지 확인합니다.

torch 가 없는 서빙 환경(Docker 의 API 이미지)에서는 자동으로 건너뜁니다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import inference  # noqa: E402
from app.model_catalog import ORDER  # noqa: E402
from app.preprocess import from_mnist_array  # noqa: E402
from conftest import SAMPLES  # noqa: E402


@pytest.mark.parametrize("name", ORDER)
def test_three_formats_agree(name):
    x = np.concatenate([from_mnist_array(img) for img in SAMPLES["images"][:50]])
    outs = {b: inference._load_one(name, b)(x) for b in ("onnx", "torchscript", "pytorch")}
    for b in ("torchscript", "pytorch"):
        assert np.abs(outs[b] - outs["onnx"]).max() < 1e-4
        assert (outs[b].argmax(1) == outs["onnx"].argmax(1)).all()
