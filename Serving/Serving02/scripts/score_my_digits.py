"""그림판 '내 손글씨 채점'에서 내려받은 JSON 을 다시 채점하거나 학습용 파일로 바꿉니다.

쓰임
- 모델을 다시 학습·교체한 뒤, 같은 손글씨 세트로 전후를 비교합니다 (실제 사람 글씨 회귀 테스트).
- 모은 손글씨를 .npz(images, labels)로 저장해 추가 학습·분석에 씁니다.

예
  python scripts/score_my_digits.py my_digits_2026-11-07.json
  python scripts/score_my_digits.py my_digits.json --api http://localhost:8000 --key dev-admin-key
  python scripts/score_my_digits.py my_digits.json --npz my_digits.npz
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

BATCH = 32   # 서버의 MAX_BATCH 기본값


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "mnist-my-digits/1":
        raise ValueError("그림판 '내 손글씨 채점'에서 내려받은 파일이 아닙니다.")
    recs = data["records"]
    images = np.array([r["pixels"] for r in recs], dtype=np.uint8).reshape(-1, 28, 28)
    labels = np.array([r["label"] for r in recs], dtype=np.int64)
    return images, labels, recs


def accuracy(recs: list[dict], labels: np.ndarray, key: str = "preds") -> dict[str, float]:
    """기록마다 {'preds': {모델: 답}, 'ensemble': 답} 형태에서 모델별·앙상블 정확도."""
    models = sorted({m for r in recs for m in r[key]})
    acc = {m: float(np.mean([r[key].get(m) == y for r, y in zip(recs, labels)])) for m in models}
    acc["ensemble"] = float(np.mean([r["ensemble"] == y for r, y in zip(recs, labels)]))
    return acc


def _b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def rescore(images: np.ndarray, post) -> list[dict]:
    """지금 서버로 다시 예측합니다. post(body) -> 응답 dict (requests 든 TestClient 든 상관없게)."""
    out = []
    for s in range(0, len(images), BATCH):
        res = post({"images": [_b64(x) for x in images[s:s + BATCH]]})
        for it in res["items"]:
            if "results" not in it:
                raise RuntimeError(f"{s + it['index']}번 이미지 실패 : {it.get('error')}")
            out.append({"preds": {m: v["pred"] for m, v in it["results"].items()}, "ensemble": it["ensemble"]["pred"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--api", help="주면 이 서버로 다시 채점해 기록 당시와 비교합니다")
    ap.add_argument("--key", default="dev-admin-key")
    ap.add_argument("--npz", help="images(N,28,28 uint8)·labels 로 저장할 경로")
    args = ap.parse_args()

    images, labels, recs = load(args.file)
    print(f"{len(labels)}장 · 숫자별 개수 {np.bincount(labels, minlength=10).tolist()}")
    then = accuracy(recs, labels)
    now = None
    if args.api:
        import requests

        def post(body):
            r = requests.post(f"{args.api.rstrip('/')}/v1/predict/batch", json=body,
                              headers={"X-API-Key": args.key}, timeout=60)
            r.raise_for_status()
            return r.json()
        now = accuracy(rescore(images, post), labels)

    print(f"\n{'모델':<10}{'기록 당시':>10}" + (f"{'지금 서버':>10}{'차이':>9}" if now else ""))
    for m in ["ensemble"] + [k for k in then if k != "ensemble"]:
        line = f"{m:<10}{then[m]:>10.1%}"
        if now and m in now:
            line += f"{now[m]:>10.1%}{(now[m] - then[m]) * 100:>+8.1f}p"
        print(line)

    if args.npz:
        np.savez_compressed(args.npz, images=images, labels=labels)
        print(f"\n저장 : {args.npz}")


if __name__ == "__main__":
    main()
