"""저장된 ONNX 모델들로 앙상블(확률 평균)의 모든 조합을 채점해 metrics.json 에 기록합니다.

실행 : python -m training.ensemble_eval --data ../data
- 모델 6개 → 조합 63가지를 원본·4px 이동·20도 회전 테스트셋(각 1만 장)으로 채점
- metrics.json 의 "ensemble" 에 기본 투표 조합(app/model_catalog.py 의 DEFAULT_ENSEMBLE),
  "ensemble_all" 에 6개 전부, "ensemble_search" 에 상위 조합을 남깁니다

⚠️ 조합을 '테스트셋 점수'로 고르면 그 점수는 약간 낙관적입니다(테스트셋에 맞춘 선택).
   제대로 하려면 학습셋에서 검증셋을 떼어 내 그걸로 고르고, 테스트셋은 마지막 확인에만 써야 합니다.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from app.model_catalog import DEFAULT_ENSEMBLE, ORDER

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def collect_probs(data_dir: str) -> tuple[dict, dict, list[str]]:
    from training.train import rotate_tf, shift_tf, test_tf
    names = [n for n in ORDER if (MODEL_DIR / f"mnist_{n}.onnx").exists()]
    sessions = {n: ort.InferenceSession(str(MODEL_DIR / f"mnist_{n}.onnx"), providers=["CPUExecutionProvider"])
                for n in names}
    probs, labels = {}, {}
    for split, tf in (("test", test_tf), ("shift", shift_tf), ("rotate", rotate_tf)):
        torch.manual_seed(0)                          # 학습 스크립트의 견고성 평가와 같은 변형
        xs, ys = [], []
        for x, y in DataLoader(datasets.MNIST(data_dir, train=False, download=True, transform=tf), batch_size=1000):
            xs.append(x.numpy())
            ys.append(y.numpy())
        x_all, labels[split] = np.concatenate(xs), np.concatenate(ys)
        for n, s in sessions.items():
            z = np.concatenate([s.run(None, {"image": x_all[i:i + 1000]})[0] for i in range(0, len(x_all), 1000)])
            z = np.exp(z - z.max(1, keepdims=True))
            probs[(n, split)] = z / z.sum(1, keepdims=True)
    return probs, labels, names


def score(members, probs, labels) -> dict:
    out = {"members": list(members)}
    for split, key in (("test", "test_acc"), ("shift", "shift4px_acc"), ("rotate", "rotate20_acc")):
        p = np.mean([probs[(m, split)] for m in members], axis=0)
        out[key] = round(float((p.argmax(1) == labels[split]).mean()), 4)
    return out


def run(data_dir: str, top: int = 8):
    probs, labels, names = collect_probs(data_dir)
    results = [score(c, probs, labels) for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]
    results.sort(key=lambda d: (d["test_acc"], d["shift4px_acc"], d["rotate20_acc"]), reverse=True)

    default = [m for m in DEFAULT_ENSEMBLE if m in names]
    metrics_path = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["ensemble"] = score(default, probs, labels)
    metrics["ensemble_all"] = score(names, probs, labels)
    metrics["ensemble_search"] = {"combinations": len(results), "top": results[:top],
                                  "note": "테스트셋으로 고른 조합이라 점수가 약간 낙관적입니다"}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"조합 {len(results)}가지 채점 (원본 / 4px 이동 / 20도 회전)")
    for d in results[:top]:
        print(f"  {d['test_acc']:.4f} / {d['shift4px_acc']:.4f} / {d['rotate20_acc']:.4f}  {d['members']}")
    print(f"기본 투표 조합 {default} : {metrics['ensemble']}")
    print(f"6개 전부 : {metrics['ensemble_all']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    run(args.data, args.top)


if __name__ == "__main__":
    main()
