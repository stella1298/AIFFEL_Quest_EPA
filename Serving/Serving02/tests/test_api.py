"""API 채점표. 정상 경로보다 실패 경로를 더 많이 확인합니다.

실행 : python -m pytest -q
"""
import base64
import io
import json

import numpy as np
from PIL import Image

from app import rate_limit
from app.inference import ensemble
from app.model_catalog import ORDER
from app.config import settings
from conftest import SAMPLES, digit_canvas, png_b64


# ---------- 상태 ----------
def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and set(r.json()["models"]) == set(ORDER)


def test_models_info_is_public(client):
    m = client.get("/v1/models").json()
    assert m["backend"] == "onnx" and m["enabled"] == ORDER
    assert m["models"]["cnn"]["test_acc"] > 0.97 and m["models"]["vit"]["family"] == "트랜스포머"
    assert m["ensemble"]["test_acc"] > 0.97


def test_ui_page_and_config(client):
    assert "<canvas" in client.get("/").text
    assert client.get("/ui-config").json()["demo_key"] == "test-demo"


# ---------- 200 ----------
def test_predict_digits(client, admin):
    hits = ens_hits = 0
    for i in range(20):
        r = client.post("/v1/predict", headers=admin, json={"image": png_b64(digit_canvas(i, offset=(10 + i * 7, 40)))})
        assert r.status_code == 200
        body = r.json()
        assert list(body["results"]) == ORDER            # 기본값 = 켜진 모델 전부, 카탈로그 순서
        assert 0 < body["ensemble"]["agreement"] <= 1
        assert len(body["results"]["cnn"]["probs"]) == 10
        assert body["preview"].startswith("data:image/png;base64,")
        hits += body["results"]["cnn"]["pred"] == int(SAMPLES["labels"][i])
        ens_hits += body["ensemble"]["pred"] == int(SAMPLES["labels"][i])
    assert hits >= 18, f"CNN 이 20장 중 {hits}장만 맞힘"
    assert ens_hits >= 18, f"앙상블이 20장 중 {ens_hits}장만 맞힘"


def test_request_id_is_echoed(client, admin):
    r = client.post("/v1/predict", headers={**admin, "X-Request-ID": "abc123"},
                    json={"image": png_b64(digit_canvas(0))})
    assert r.headers["X-Request-ID"] == "abc123" and r.json()["request_id"] == "abc123"
    assert float(r.headers["X-Process-Time-ms"]) >= 0


def test_single_model(client, admin):
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(digit_canvas(1)), "models": ["mlp"]})
    assert list(r.json()["results"]) == ["mlp"] and r.json()["agree"] is True
    assert r.json()["ensemble"]["pred"] == r.json()["results"]["mlp"]["pred"]


def test_model_subset(client, admin):
    r = client.post("/v1/predict", headers=admin,
                    json={"image": png_b64(digit_canvas(4)), "models": ["vit", "lstm", "vit"]})
    assert list(r.json()["results"]) == ["vit", "lstm"]          # 중복 제거, 순서 유지


def test_ensemble_rule():
    """확률 평균으로 최종 답을 정하고, 동의한 모델 비율을 셉니다."""
    def res(pred, conf):
        p = [(1 - conf) / 9] * 10
        p[pred] = conf
        return {"pred": pred, "confidence": conf, "probs": p}
    # 6개 중 5개가 4, 1개가 9 → 최종 4, 동의 5/6
    e = ensemble({f"m{i}": res(4, 0.9) for i in range(5)} | {"x": res(9, 0.99)})
    assert e["pred"] == 4 and e["votes"] == {"4": 5, "9": 1} and abs(e["agreement"] - 5 / 6) < 1e-3
    # 다수결이 아니라 '확률 평균' : 3개는 약하게 7, 3개는 강하게 1 → 1
    e = ensemble({f"a{i}": res(7, 0.4) for i in range(3)} | {f"b{i}": res(1, 0.95) for i in range(3)})
    assert e["pred"] == 1 and e["agreement"] == 0.5


def test_white_background_photo_is_inverted(client, admin):
    """흰 종이에 검은 글씨(사진·스캔)도 반전해서 맞혀야 합니다."""
    img = Image.eval(digit_canvas(2), lambda v: 255 - v)
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(img)})
    assert r.json()["results"]["cnn"]["pred"] == int(SAMPLES["labels"][2])


def test_file_upload(client, admin):
    buf = io.BytesIO()
    digit_canvas(3).save(buf, format="PNG")
    r = client.post("/v1/predict/file", headers=admin, files={"file": ("d.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["results"]["cnn"]["pred"] == int(SAMPLES["labels"][3])


# ---------- 401 ----------
def test_missing_key(client):
    r = client.post("/v1/predict", json={"image": png_b64(digit_canvas(0))})
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"


def test_wrong_key(client):
    r = client.post("/v1/predict", headers={"X-API-Key": "nope"}, json={"image": png_b64(digit_canvas(0))})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


# ---------- 422 ----------
def test_validation_errors(client, admin):
    r = client.post("/v1/predict", headers=admin, json={})
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "image"
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(digit_canvas(0)), "models": ["rnn"]})
    assert r.status_code == 422
    r = client.post("/v1/predict/batch", headers=admin,
                    json={"images": [png_b64(digit_canvas(0))] * (settings.max_batch + 1)})
    assert r.status_code == 422


def test_labels_must_match_images(client, admin):
    r = client.post("/v1/predict/batch", headers=admin, json={"images": [png_b64(digit_canvas(0))], "labels": [1, 2]})
    assert r.status_code == 422


# ---------- 400 · 413 · 415 ----------
def test_bad_base64(client, admin):
    r = client.post("/v1/predict", headers=admin, json={"image": "data:image/png;base64,@@@@@@@@@@@@@@@@@@@@"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_base64"


def test_blank_canvas(client, admin):
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(Image.new("L", (280, 280), 0))})
    assert r.status_code == 400 and r.json()["error"]["code"] == "blank_image"


def test_not_an_image(client, admin):
    fake = "data:image/png;base64," + base64.b64encode(b"hello, this is not an image at all").decode()
    r = client.post("/v1/predict", headers=admin, json={"image": fake})
    assert r.status_code == 415


def test_too_large_file(client, admin):
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1000, 1000, 3), dtype=np.uint8))
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(noise)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_decompression_bomb(client, admin):
    """파일은 작지만(단색) 픽셀이 900만 개인 이미지 → 풀기 전에 413."""
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(Image.new("L", (3000, 3000), 0))})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_pixels"


# ---------- 일괄 · 스트리밍 ----------
def test_batch_with_labels_and_bad_item(client, admin):
    images = [png_b64(digit_canvas(i)) for i in range(10)] + ["data:image/png;base64,AAAA"]
    labels = [int(x) for x in SAMPLES["labels"][:10]] + [0]
    r = client.post("/v1/predict/batch", headers=admin, json={"images": images, "labels": labels})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 11
    assert body["items"][-1]["error"]                     # 잘못된 한 장은 그 항목만 실패
    assert body["accuracy"]["cnn"] >= 0.9 and body["accuracy"]["ensemble"] >= 0.9
    it = body["items"][1]                                  # 앙상블 답 = 투표 모델 확률 평균의 최댓값
    members = it["ensemble"]["members"]
    mean = np.mean([it["results"][m]["probs"] for m in members], axis=0)
    assert members == ["lenet", "cnn", "lstm"] and it["ensemble"]["pred"] == int(mean.argmax())


def test_split_opinion_needs_review(client, admin):
    """MNIST 테스트 샘플 0번(정답 0)은 6처럼 기울어진 0 이라 모델들의 답이 1·6·0·8 로 갈립니다.
    투표 모델(LeNet·CNN·LSTM)도 6·6·0 으로 갈리고 평균 확률이 낮아 '확인 필요'여야 합니다."""
    r = client.post("/v1/predict", headers=admin, json={"image": png_b64(digit_canvas(0))}).json()
    assert r["agree"] is False and r["needs_review"] is True
    assert r["ensemble"]["confidence"] < settings.review_threshold


def test_stream_events(client, admin):
    images = [png_b64(digit_canvas(i)) for i in range(3)]
    events = []
    with client.stream("POST", "/v1/predict/batch/stream", headers=admin, json={"images": images}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:") and events[-1] == "done":
                assert json.loads(line[5:])["count"] == 3
    assert events == ["start", "item", "item", "item", "done"]


# ---------- 429 ----------
def test_rate_limit(client):
    rate_limit._buckets["tiny"] = rate_limit.TokenBucket(rate_per_min=1, burst=2)
    headers = {"X-API-Key": "test-tiny"}
    body = {"image": png_b64(digit_canvas(0)), "return_preview": False}
    codes = [client.post("/v1/predict", headers=headers, json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/v1/predict", headers=headers, json=body)
    assert int(r.headers["Retry-After"]) > 0 and r.json()["error"]["code"] == "rate_limited"


# ---------- 전처리 ----------
def test_preprocess_never_wraps_around():
    """무게가 한쪽에 쏠린 글씨(두꺼운 윗변 + 가는 꼬리)도 정렬할 때 반대편으로 넘어가면 안 됩니다.

    np.roll 로 무게중심을 옮기던 첫 구현에서는 이 그림의 꼬리 끝이 28x28 의 맨 위로 넘어갔습니다.
    """
    from PIL import ImageDraw
    from app.preprocess import to_mnist
    img = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(img)
    d.rectangle([60, 40, 220, 110], fill=255)
    d.line([(140, 110), (140, 250)], fill=255, width=4)
    _, arr = to_mnist(img)
    for axis in (0, 1):
        lines = np.nonzero(arr.max(axis=axis) > 30)[0]
        assert (np.diff(lines) == 1).all(), f"글씨가 가장자리에서 잘려 반대편으로 넘어감 (axis={axis})"
