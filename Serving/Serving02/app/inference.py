"""모델 로드와 추론. 형식(backend)은 환경 변수 MODEL_BACKEND 로 고릅니다.

| backend     | 읽는 파일 | 필요한 것                          | 쓰임                       |
|-------------|-----------|------------------------------------|----------------------------|
| onnx (기본) | .onnx     | onnxruntime 만 (torch 불필요)       | 서빙 · Docker 이미지 경량화 |
| torchscript | .pt       | torch                              | 클래스 코드 없이 torch 로   |
| pytorch     | .pth      | torch + 모델 클래스 코드(training/) | 학습 코드와 같은 환경       |

세 형식 모두 같은 답을 내는지는 training/train.py 가 저장할 때 확인합니다.
"""
import json
import logging
import time

import numpy as np

from app.config import settings
from app.model_catalog import CATALOG

logger = logging.getLogger("mnist_api")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # 요청 동시성은 스레드풀이 맡으므로 모델 하나는 1스레드로
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: x})[0]


class TorchModel:
    def __init__(self, module):
        import torch
        self.torch = torch
        self.m = module.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.m(self.torch.from_numpy(x)).numpy()


def _load_one(name: str, backend: str):
    d = settings.model_dir
    if backend == "onnx":
        path = d / f"mnist_{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(path)
        return OnnxModel(path)
    import torch
    if backend == "torchscript":
        return TorchModel(torch.jit.load(str(d / f"mnist_{name}.pt"), map_location="cpu"))
    if backend == "pytorch":
        from training.models import MODELS        # state_dict 는 클래스 코드가 있어야 열립니다
        m = MODELS[name]()
        m.load_state_dict(torch.load(d / f"mnist_{name}.pth", map_location="cpu", weights_only=True))
        return TorchModel(m)
    raise ValueError(f"알 수 없는 MODEL_BACKEND : {backend}")


class ModelRegistry:
    """서버가 켜질 때 한 번 만들고 모든 요청이 같이 씁니다 (요청마다 로드하지 않음)."""

    def __init__(self):
        self.models: dict = {}
        self.metrics: dict = {}
        self.loaded_at = None
        self.load_seconds = None

    def load(self):
        t0 = time.perf_counter()
        for name in settings.enabled_models:
            if name not in CATALOG:
                logger.warning("카탈로그에 없는 모델 이름이라 건너뜁니다 : %s", name)
                continue
            try:
                self.models[name] = _load_one(name, settings.backend)
                logger.info("모델 로드 : %s (%s)", name, settings.backend)
            except (FileNotFoundError, OSError) as e:
                # 파일이 없는 모델만 빼고 나머지로 서비스합니다 (예 : 일부만 학습한 경우)
                logger.warning("모델 파일이 없어 건너뜁니다 : %s (%s)", name, type(e).__name__)
        # 워밍업 : 첫 요청만 느린 현상을 서버 시작 때 미리 치릅니다
        dummy = np.zeros((1, 1, 28, 28), dtype=np.float32)
        for m in self.models.values():
            m(dummy)
        mpath = settings.model_dir / "metrics.json"
        self.metrics = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def predict(self, x: np.ndarray, which: list[str]) -> dict:
        """x : (N,1,28,28) float32. 동기 함수 → 서버에서는 스레드풀에서 부릅니다."""
        out = {}
        for name in which:
            t0 = time.perf_counter()
            probs = softmax(self.models[name](x))
            ms = (time.perf_counter() - t0) * 1000
            out[name] = [{"pred": int(p.argmax()), "confidence": round(float(p.max()), 4),
                          "probs": [round(float(v), 4) for v in p], "infer_ms": round(ms / len(x), 3)}
                         for p in probs]
        return out


def ensemble(results: dict, members: list[str] | None = None) -> dict:
    """모델별 결과 → 확률 평균(soft voting) 과 투표 현황.

    members : 투표에 참여할 모델. 요청한 모델 중 투표 모델이 하나도 없으면 요청한 모델 전부가 투표합니다.
    agreement : 투표 모델 중 최종 답과 같은 숫자를 고른 비율 (3개 중 2개면 0.67)
    """
    voters = [m for m in (members or []) if m in results] or list(results)
    results = {m: results[m] for m in voters}
    probs = np.mean([r["probs"] for r in results.values()], axis=0)
    pred = int(probs.argmax())
    votes: dict[str, int] = {}
    for r in results.values():
        votes[str(r["pred"])] = votes.get(str(r["pred"]), 0) + 1
    votes = dict(sorted(votes.items(), key=lambda kv: -kv[1]))
    return {"pred": pred, "confidence": round(float(probs[pred]), 4),
            "probs": [round(float(v), 4) for v in probs],
            "votes": votes, "agreement": round(votes.get(str(pred), 0) / len(results), 4),
            "members": voters}


registry = ModelRegistry()
