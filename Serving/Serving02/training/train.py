"""MNIST 로 여섯 모델(로지스틱 회귀·MLP·LeNet-5·CNN·LSTM·ViT)을 학습하고,
세 가지 형식으로 저장해 서로 같은 답을 내는지 검증합니다. 마지막에 여섯 모델의 앙상블 정확도도 잽니다.

실행 (mnist-project 폴더에서):
    python -m training.train --epochs 3
    python -m training.train --epochs 3 --data ../data      # 이미 받아 둔 MNIST 가 있으면
    python -m training.train --models vit lstm              # 일부만 다시 학습 (나머지 결과는 유지)

만들어지는 파일 (models/)
    mnist_<키>.pth    state_dict      : 가중치만. 불러오려면 모델 클래스 코드가 필요
    mnist_<키>.pt     TorchScript     : 구조+가중치. 클래스 코드 없이 torch 만 있으면 됨
    mnist_<키>.onnx   ONNX            : 구조+가중치. torch 없이 onnxruntime 만으로 추론 → 서빙 기본값
    metrics.json      정확도·견고성·파일 크기·형식별 지연 시간·앙상블 정확도
samples/mnist_test_200.npz : 테스트셋 200장 (화면·테스트·부하 테스트에서 씀)
"""
import argparse
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.model_catalog import ORDER
from training.models import MODELS

# torch.jit 은 새 버전에서 폐기 예정 경고를 냅니다. 과정에서 배운 형식이라 그대로 쓰고 경고만 숨깁니다
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "samples"
MEAN, STD = 0.1307, 0.3081          # MNIST 학습셋의 평균·표준편차. 서빙 전처리와 반드시 같아야 함
normalize = transforms.Normalize((MEAN,), (STD,))

# 학습 때만 살짝 돌리고 밀어 줍니다. 사람이 그린 숫자는 MNIST 보다 위치·기울기가 제각각이기 때문입니다
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), normalize,
])
test_tf = transforms.Compose([transforms.ToTensor(), normalize])
# 견고성 확인용 : 4픽셀 이동 / 20도 회전한 테스트셋
shift_tf = transforms.Compose([transforms.RandomAffine(degrees=0, translate=(4 / 28, 4 / 28)),
                               transforms.ToTensor(), normalize])
rotate_tf = transforms.Compose([transforms.RandomAffine(degrees=(20, 20)), transforms.ToTensor(), normalize])


def evaluate(model, loader, seed=0):
    torch.manual_seed(seed)          # 무작위 변형을 쓰는 테스트셋도 매번 같은 결과가 나오게
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return round(correct / total, 4)


def median_ms(fn, n=200):
    for _ in range(10):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)


def export_and_verify(name, model, sample):
    """state_dict / TorchScript / ONNX 로 저장하고, 세 형식의 출력이 같은지 확인합니다."""
    model.eval()
    paths = {k: MODEL_DIR / f"mnist_{name}.{k}" for k in ("pth", "pt", "onnx")}

    torch.save(model.state_dict(), paths["pth"])

    traced = torch.jit.trace(model, sample[:1])
    traced.save(str(paths["pt"]))

    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.onnx.export(model, (sample[:2],), paths["onnx"], dynamo=True,
                      input_names=["image"], output_names=["logits"],
                      dynamic_shapes={"x": {0: batch}}, external_data=False, verbose=False)

    # --- 같은 입력 → 같은 출력인지 (형식을 바꿔도 모델이 바뀌면 안 됨) ---
    loaded_ts = torch.jit.load(str(paths["pt"]))
    sess = ort.InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(sample).numpy()
        ts_out = loaded_ts(sample).numpy()
    onnx_out = sess.run(None, {"image": sample.numpy()})[0]
    max_diff = {"torchscript": float(np.abs(ref - ts_out).max()),
                "onnx": float(np.abs(ref - onnx_out).max())}
    same_pred = {"torchscript": bool((ref.argmax(1) == ts_out.argmax(1)).all()),
                 "onnx": bool((ref.argmax(1) == onnx_out.argmax(1)).all())}

    # --- 이미지 1장 지연 시간 (CPU, 중앙값) ---
    one = sample[:1]
    one_np = one.numpy()
    with torch.no_grad():
        latency = {
            "pytorch": median_ms(lambda: model(one)),
            "torchscript": median_ms(lambda: loaded_ts(one)),
            "onnx": median_ms(lambda: sess.run(None, {"image": one_np})),
        }
    sizes = {k: p.stat().st_size for k, p in paths.items()}
    return {"files": {k: p.name for k, p in paths.items()}, "file_bytes": sizes,
            "max_abs_diff": max_diff, "same_prediction": same_pred, "latency_ms": latency}


def train_one(name, epochs, loaders):
    torch.manual_seed(0)
    model = MODELS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] 파라미터 {n_params:,}개")
    t0 = time.time()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        acc = evaluate(model, loaders["test"])
        loss_avg = running / len(loaders["train"].dataset)
        history.append({"epoch": ep, "loss": round(loss_avg, 4), "test_acc": acc})
        print(f"  epoch {ep}  loss {loss_avg:.4f}  test acc {acc:.2%}")
    train_sec = round(time.time() - t0, 1)
    result = {
        "params": n_params, "epochs": epochs, "train_sec": train_sec, "history": history,
        "test_acc": history[-1]["test_acc"],
        "shift4px_acc": evaluate(model, loaders["shift"]),
        "rotate20_acc": evaluate(model, loaders["rotate"]),
    }
    print(f"  견고성 : 4px 이동 {result['shift4px_acc']:.2%} / 20도 회전 {result['rotate20_acc']:.2%}")
    return model, result


def main():
    ap = argparse.ArgumentParser(description="MNIST 여섯 모델 학습과 3가지 형식 내보내기")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data"), help="MNIST 를 받을(또는 이미 있는) 폴더")
    ap.add_argument("--models", nargs="+", default=ORDER, choices=list(MODELS))
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)
    ds = lambda tf, train=False: datasets.MNIST(args.data, train=train, download=True, transform=tf)
    loaders = {
        "train": DataLoader(ds(train_tf, True), batch_size=128, shuffle=True),
        "test": DataLoader(ds(test_tf), batch_size=1000),
        "shift": DataLoader(ds(shift_tf), batch_size=1000),
        "rotate": DataLoader(ds(rotate_tf), batch_size=1000),
    }
    print(f"학습 {len(loaders['train'].dataset)}장 / 테스트 {len(loaders['test'].dataset)}장")

    # 화면·테스트에서 쓸 샘플 200장 (원본 픽셀 0~255 그대로)
    raw_test = datasets.MNIST(args.data, train=False, download=True)
    idx = np.random.default_rng(0).choice(len(raw_test), 200, replace=False)
    np.savez_compressed(SAMPLE_DIR / "mnist_test_200.npz",
                        images=raw_test.data[idx].numpy().astype(np.uint8),
                        labels=raw_test.targets[idx].numpy().astype(np.int64))

    sample = next(iter(loaders["test"]))[0][:64]
    metrics_path = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"models": {}}
    for name in args.models:
        model, result = train_one(name, args.epochs, loaders)
        result.update(export_and_verify(name, model, sample))
        metrics["models"][name] = result
        print(f"  형식별 최대 오차 {result['max_abs_diff']}  /  1장 지연(ms) {result['latency_ms']}")
    # 서버의 응답 순서와 맞추려고 카탈로그 순서로 다시 정렬합니다
    metrics["models"] = {k: metrics["models"][k] for k in ORDER if k in metrics["models"]}
    metrics["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                              "onnxruntime": ort.__version__, "cpu": platform.processor() or platform.machine()}
    metrics["preprocess"] = {"mean": MEAN, "std": STD}
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장 완료 : {MODEL_DIR}")

    # 앙상블 : 저장된 모델들의 모든 조합(6개면 63가지)을 채점해 metrics.json 에 덧붙입니다
    from training.ensemble_eval import run as eval_ensembles
    print("\n[앙상블] 모든 조합 채점")
    eval_ensembles(args.data)


if __name__ == "__main__":
    main()
