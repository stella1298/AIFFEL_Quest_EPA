"""그림판 '내 손글씨 채점' 내려받기 파일과 다시 채점 스크립트를 확인합니다."""
import json
import sys

import numpy as np
import pytest

from conftest import ROOT, SAMPLES

sys.path.insert(0, str(ROOT / "scripts"))
import score_my_digits as smd  # noqa: E402


def _export(tmp_path, n=40):
    """그림판이 내려주는 형식 그대로 : 모델이 본 28×28 픽셀 + 정답 + 기록 당시 답."""
    imgs, labels = SAMPLES["images"][:n], SAMPLES["labels"][:n]
    recs = [{"label": int(y), "ensemble": int(y), "needs_review": False, "preds": {"cnn": int(y)},
             "conf": {"cnn": 0.99}, "pixels": img.reshape(-1).tolist(), "t": 0} for img, y in zip(imgs, labels)]
    recs[0]["ensemble"] = (recs[0]["label"] + 1) % 10           # 하나는 틀린 기록
    path = tmp_path / "my_digits.json"
    path.write_text(json.dumps({"format": "mnist-my-digits/1", "n": n, "records": recs}), encoding="utf-8")
    return path


def test_ui_has_score_mode(client):
    html = client.get("/").text
    for el in ('id="m-score"', 'id="rec"', 'id="sc-export"', "mnist-my-digits/1"):
        assert el in html


def test_load_and_accuracy(tmp_path):
    images, labels, recs = smd.load(_export(tmp_path))
    assert images.shape == (40, 28, 28) and images.dtype == np.uint8
    assert (images == SAMPLES["images"][:40]).all()
    acc = smd.accuracy(recs, labels)
    assert acc["cnn"] == 1.0 and acc["ensemble"] == pytest.approx(39 / 40)


def test_rejects_other_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        smd.load(p)


def test_rescore_against_api(client, admin, tmp_path):
    """32장 묶음 경계를 넘는 40장 : 다시 채점한 결과가 장 수·순서 모두 맞아야 합니다."""
    images, labels, _ = smd.load(_export(tmp_path))
    post = lambda body: client.post("/v1/predict/batch", json=body, headers=admin).json()  # noqa: E731
    now = smd.rescore(images, post)
    assert len(now) == 40 and set(now[0]["preds"]) >= {"lenet", "cnn", "lstm"}
    acc = smd.accuracy(now, labels)
    assert acc["ensemble"] >= 0.9                                # 이미 MNIST 형식인 샘플이라 거의 다 맞음
