"""MNIST 서빙 대시보드 (Streamlit). 그림판(/)과 별도로, 운영자·발표용 화면입니다.

- 탭 1 이미지 업로드 : 사진·스캔한 숫자 파일을 올려 모델들과 앙상블 비교 (multipart → /v1/predict/file)
- 탭 2 샘플 채점     : MNIST 테스트 샘플을 스트리밍(SSE)으로 보내 한 장씩 채점, 틀린 것 모아 보기
- 탭 3 모델 비교     : 학습 지표·견고성·형식별 파일 크기와 지연 시간

실행 : streamlit run frontend/streamlit_app.py
API 주소 : API_BASE_URL → localhost:8000 → 둘 다 없으면 API 를 이 프로세스 안에 직접 띄웁니다 (내장 모드).
          내장 모드 덕분에 Streamlit Community Cloud(xxx.streamlit.app)에 이 파일 하나만 올려도 동작합니다.
모델 목록·이름·색은 서버의 /v1/models 에서 받아 옵니다. 모델을 추가해도 이 파일은 고치지 않아도 됩니다.
API 키는 이 화면의 서버 쪽(Python)에서 붙여 보냅니다. 브라우저에는 노출되지 않습니다.
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY = os.environ.get("UI_API_KEY", "dev-admin-key")
SAMPLES = ROOT / "samples" / "mnist_test_200.npz"
MAX_BATCH = 32
ENSEMBLE_COLOR = "#1d2433"

st.set_page_config(page_title="MNIST 서빙 대시보드", page_icon="✍️", layout="wide")


def _alive(base: str) -> bool:
    try:
        return requests.get(f"{base}/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="모델 6개를 불러오는 중입니다…")
def start_embedded_api() -> str:
    """API 서버를 이 프로세스 안의 스레드로 띄웁니다 (Streamlit Community Cloud 처럼 화면 하나만 올릴 수 있는 곳용).

    FastAPI 코드는 그대로 쓰고, 127.0.0.1 의 빈 포트에만 열어 외부에서는 API 에 직접 닿지 않습니다.
    화면은 여전히 HTTP 로 API 를 부르므로, 따로 띄웠을 때와 같은 경로(인증·속도 제한·검증)를 지납니다.
    """
    import socket
    import sys
    import threading
    import time

    import uvicorn

    sys.path.insert(0, str(ROOT))
    from app.main import app as api_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api_app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="embedded-api").start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(150):                       # 최대 30초 : 모델 로딩 + 워밍업
        try:
            if requests.get(f"{base}/ready", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return base


# API 주소 정하기 : API_BASE_URL 이 있으면 그 서버, 없으면 localhost:8000, 그것도 없으면 내장 API
if os.environ.get("API_BASE_URL"):
    API_BASE, EMBEDDED = os.environ["API_BASE_URL"].rstrip("/"), False
elif _alive("http://localhost:8000"):
    API_BASE, EMBEDDED = "http://localhost:8000", False
else:
    try:
        API_BASE, EMBEDDED = start_embedded_api(), True
    except ImportError as e:                   # 대시보드 전용 이미지처럼 app/ 이나 서버 패키지가 없을 때
        st.error(f"API 서버도 없고, 내장 API 를 띄울 패키지도 없습니다 : {e}")
        st.stop()


@st.cache_data
def load_samples():
    d = np.load(SAMPLES)
    return d["images"], d["labels"]


@st.cache_data(ttl=30)
def model_info(api_base: str) -> dict:
    try:
        return requests.get(f"{api_base}/v1/models", timeout=5).json()
    except requests.RequestException:
        return {}


def to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def explain(resp: requests.Response) -> str:
    """상태 코드별 안내 문구 (인증과 화면 노트)."""
    try:
        err = resp.json().get("error", {})
    except ValueError:
        err = {}
    msg = err.get("message", resp.text[:200])
    hint = {401: "🔑 사이드바의 API 키를 확인해 주세요.", 413: "📦 파일이 너무 큽니다.",
            415: "🖼️ 이미지 파일만 올릴 수 있습니다.", 422: "✏️ 입력 형식이 맞지 않습니다.",
            429: "⏳ 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
            503: "⏳ 서버가 바쁘거나 모델을 불러오는 중입니다."}.get(resp.status_code, "❌ 오류")
    return f"{hint} (HTTP {resp.status_code}) {msg}"


INFO = model_info(API_BASE)
KEYS = INFO.get("enabled", [])
META = INFO.get("models", {})
NAME = {k: META[k]["name"] for k in KEYS}
MEMBERS = INFO.get("ensemble_members", [])
COLOR = {k: META[k]["color"] for k in KEYS}

# ===== 사이드바 =====
with st.sidebar:
    st.header("⚙️ 연결")
    api_key = st.text_input("API 키", value=DEFAULT_KEY, type="password")
    st.caption("API : 이 앱 안에서 함께 실행 중 (내장 모드)" if EMBEDDED else f"API 주소 : `{API_BASE}`")
    try:
        ready = requests.get(f"{API_BASE}/ready", timeout=3)
        if ready.status_code == 200:
            st.success(f"🟢 준비 완료 · {ready.json()['backend']} · 모델 {len(ready.json()['models'])}개")
        else:
            st.warning("🟡 모델을 불러오는 중입니다.")
    except requests.RequestException:
        st.error("🔴 API 서버에 연결할 수 없습니다.")
    if KEYS:
        chosen = st.multiselect("참여할 모델", KEYS, default=KEYS, format_func=lambda k: NAME[k])
    else:
        chosen = []
    st.divider()
    if not EMBEDDED:                           # 내장 모드의 API 는 127.0.0.1 에만 열려 있어 링크가 동작하지 않습니다
        st.markdown(f"[✍️ 그림판 열기]({API_BASE}/) · [📘 API 문서]({API_BASE}/docs)")

st.title("✍️ MNIST 손글씨 인식 · 모델 6개 비교")
if not KEYS:
    st.error("서버에서 모델 목록을 받지 못했습니다. API 서버가 떠 있는지 확인해 주세요.")
    st.stop()
if not chosen:
    st.warning("사이드바에서 참여할 모델을 하나 이상 고르세요.")
    st.stop()

tab_up, tab_eval, tab_cmp = st.tabs(["🖼️ 이미지 업로드", "🧪 샘플 채점 (스트리밍)", "📊 모델 비교"])

# ===== 탭 1 : 업로드 =====
with tab_up:
    st.write("종이에 쓴 숫자를 찍은 사진이나 그림 파일을 올려 보세요. 흰 배경·검은 글씨도 알아서 뒤집습니다.")
    up = st.file_uploader("이미지 파일", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if up is not None:
        r = requests.post(f"{API_BASE}/v1/predict/file", headers={"X-API-Key": api_key},
                          params=[("models", m) for m in chosen],
                          files={"file": (up.name, up.getvalue(), up.type)}, timeout=30)
        if r.status_code != 200:
            st.error(explain(r))
        else:
            d = r.json()
            e = d["ensemble"]
            c0, c1, c2 = st.columns([1, 1, 2])
            c0.image(up, caption="올린 이미지", width="stretch")
            c1.image(d["preview"], caption="모델 입력 28×28", width=112)
            c1.metric("최종 답 (투표 모델 확률 평균)", e["pred"], f"평균 확률 {e['confidence']:.1%}", delta_color="off")
            c1.caption(f"투표 ({'·'.join(NAME[m] for m in e['members'])}) : "
                       + " · ".join(f"{k} → {v}표" for k, v in e["votes"].items()))
            if d["needs_review"]:
                c2.warning(f"⚠️ 사람이 확인할 대상입니다 (투표 모델 {len(e['members'])}개 중 "
                           f"{round(e['agreement'] * len(e['members']))}개 동의 · 평균 확률 {e['confidence']:.1%}).")
            else:
                c2.success("✅ 투표 모델이 같은 답을 냈고 확신도 충분합니다.")
            table = pd.DataFrame({NAME[k]: res["probs"] for k, res in d["results"].items()},
                                 index=[str(i) for i in range(10)])
            c2.bar_chart(table, stack=False, color=[COLOR[k] for k in d["results"]], height=260,
                         x_label="숫자", y_label="확률")
            cols = st.columns(len(d["results"]))
            for col, (k, res) in zip(cols, d["results"].items()):
                col.metric(NAME[k] + (" · 투표" if k in e["members"] else " · 참고"), res["pred"], f"{res['confidence']:.1%}",
                           delta_color="normal" if res["pred"] == e["pred"] else "inverse")
            st.caption(f"요청 ID {d['request_id']} · 서버 처리 {r.headers.get('X-Process-Time-ms')}ms")

# ===== 탭 2 : 샘플 채점 (SSE) =====
with tab_eval:
    images, labels = load_samples()
    st.write("MNIST 테스트셋에서 뽑아 둔 샘플을 서버로 보내 채점합니다. "
             "결과를 다 기다리지 않고 **한 장씩 도착하는 대로** 표시합니다 (Server-Sent Events).")
    n = st.slider("채점할 장 수", 10, len(images), 100, step=10)
    if st.button("🧪 채점 시작", type="primary"):
        bar = st.progress(0.0, text="채점 중...")
        live = st.empty()
        rows, failed = [], None
        for start in range(0, n, MAX_BATCH):        # 서버의 한 번 최대 장 수에 맞춰 나눠 보냄
            idx = list(range(start, min(start + MAX_BATCH, n)))
            body = {"images": [to_b64(images[i]) for i in idx], "labels": [int(labels[i]) for i in idx],
                    "models": chosen}
            with requests.post(f"{API_BASE}/v1/predict/batch/stream", headers={"X-API-Key": api_key},
                               json=body, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    failed = explain(r)
                    break
                event = None
                for line in r.iter_lines(decode_unicode=True):
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event == "item":
                        it = json.loads(line[5:])
                        if not it.get("results"):
                            continue
                        i = idx[it["index"]]
                        row = {"i": i, "정답": int(labels[i]), "앙상블": it["ensemble"]["pred"]}
                        row.update({k: it["results"][k]["pred"] for k in chosen})
                        rows.append(row)
                        df = pd.DataFrame(rows)
                        bar.progress(len(rows) / n, text=f"{len(rows)}/{n}장")
                        live.markdown(" · ".join(f"{NAME.get(k, k)} **{(df[k] == df['정답']).mean():.1%}**"
                                                 for k in chosen + ["앙상블"]) + " (지금까지)")
        if failed:
            st.error(failed)
        if rows:
            st.session_state["eval"] = (pd.DataFrame(rows), list(chosen))

    if "eval" in st.session_state:
        df, used = st.session_state["eval"]
        cols = st.columns(len(used) + 1)
        for col, k in zip(cols, used + ["앙상블"]):
            col.metric(NAME.get(k, k), f"{(df[k] == df['정답']).mean():.1%}")
        voters = [m for m in MEMBERS if m in used] or used
        wrong_vote = df[df[voters].ne(df["정답"], axis=0).any(axis=1)]
        wrong_ens = df[df["앙상블"] != df["정답"]]
        st.subheader(f"투표 모델 중 하나라도 틀린 샘플 {len(wrong_vote)}장 · 최종 답이 틀린 샘플 {len(wrong_ens)}장")
        st.caption(f"{len(df)}장은 표본이 작아 한두 장 차이로 순위가 바뀝니다. 1만 장 기준 정확도는 '모델 비교' 탭을 보세요. "
                   "그림 아래 : 정답 → 최종 답, 틀린 모델")
        short = {"logreg": "LR", "lenet": "LeNet"}
        grid = st.columns(8)
        for k, (_, row) in enumerate(wrong_vote.head(24).iterrows()):
            missed = [short.get(m, NAME.get(m, m)) for m in used if row[m] != row["정답"]]
            mark = "❌" if row["앙상블"] != row["정답"] else "✅"
            grid[k % 8].image(images[int(row["i"])], width=56,
                              caption=f"{int(row['정답'])} → {int(row['앙상블'])} {mark}  틀림: {'·'.join(missed)}")
        with st.expander("전체 결과 표"):
            st.dataframe(df.rename(columns=NAME), hide_index=True, width="stretch")

# ===== 탭 3 : 모델 비교 =====
with tab_cmp:
    ens = INFO.get("ensemble") or {}
    ens_label = "앙상블 (" + "·".join(NAME.get(m, m) for m in ens.get("members", [])) + ")"
    st.caption(f"학습 시각 {INFO.get('trained_at')} · 서빙 형식 `{INFO.get('backend')}` · 버전 {INFO.get('model_version')}")
    rows = {NAME[k]: META[k] for k in KEYS if META[k].get("test_acc") is not None}
    # 정확도 99% 와 97% 는 막대로 보면 차이가 안 보이므로 '오답률'로 그립니다
    err = pd.DataFrame({name: {"① 원본": 100 * (1 - v["test_acc"]), "② 4px 이동": 100 * (1 - v["shift4px_acc"]),
                               "③ 20도 회전": 100 * (1 - v["rotate20_acc"])} for name, v in rows.items()})
    colors = [COLOR[k] for k in KEYS if NAME[k] in rows]
    if ens.get("test_acc"):
        err[ens_label] = [100 * (1 - ens["test_acc"]), 100 * (1 - ens["shift4px_acc"]), 100 * (1 - ens["rotate20_acc"])]
        colors.append(ENSEMBLE_COLOR)
    st.write("**오답률 (%) · 낮을수록 좋음** (테스트셋 1만 장, 같은 조건으로 3 epoch 학습)")
    st.bar_chart(err, stack=False, color=colors, y_label="오답률 %", height=320)
    st.caption("숫자를 조금만 옮기거나 돌리면 픽셀 위치를 외우는 모델(로지스틱·MLP)의 오답이 크게 늘어납니다. "
               "합성곱은 이웃 픽셀의 모양을 보기 때문에 덜 흔들립니다. 구조가 다른 세 모델(합성곱 2 + 순환 1)을 "
               "섞은 앙상블이 세 항목 모두에서 가장 낮습니다.")
    search = INFO.get("ensemble_search") or {}
    c1, c2 = st.columns(2)
    lat = pd.DataFrame({name: v["latency_ms"] for name, v in rows.items()})
    c1.write("**형식별 1장 추론 시간 (ms, CPU) · 낮을수록 좋음**")
    c1.bar_chart(lat, stack=False, color=colors[:len(rows)], y_label="ms", height=280)
    params = pd.DataFrame({"파라미터 (천 개)": {name: v["params"] / 1000 for name, v in rows.items()}})
    c2.write("**모델 크기 (파라미터 수)**")
    c2.bar_chart(params, horizontal=True, height=280, color=ENSEMBLE_COLOR)
    table = pd.DataFrame({name: {"계열": v["family"], "파라미터": f"{v['params']:,}",
                                 "학습 시간": f"{v['train_sec']:.0f}초" if v.get("train_sec") else "-",
                                 "테스트 정확도": f"{v['test_acc']:.2%}",
                                 "ONNX 파일": f"{v['file_bytes']['onnx'] / 1e6:.2f}MB",
                                 "ONNX 1장": f"{v['latency_ms']['onnx']:.3f}ms",
                                 "설명": v["description"]} for name, v in rows.items()})
    st.dataframe(table.T, width="stretch")

    if search.get("top"):
        st.write(f"**앙상블 조합 탐색** — 모델 {len(KEYS)}개로 만들 수 있는 조합 {search['combinations']}가지 중 상위")
        rows_s = [{"조합": " + ".join(NAME.get(m, m) for m in d["members"]), "원본": d["test_acc"],
                   "4px 이동": d["shift4px_acc"], "20도 회전": d["rotate20_acc"]} for d in search["top"]]
        if INFO.get("ensemble_all"):
            a = INFO["ensemble_all"]
            rows_s.append({"조합": "(참고) 6개 전부", "원본": a["test_acc"], "4px 이동": a["shift4px_acc"],
                           "20도 회전": a["rotate20_acc"]})
        st.dataframe(pd.DataFrame(rows_s).style.format({c: "{:.2%}" for c in ("원본", "4px 이동", "20도 회전")}),
                     hide_index=True, width="stretch")
        st.caption("⚠️ " + search.get("note", "") + ". 약한 모델까지 모두 섞으면 오히려 CNN 혼자보다 낮아집니다.")
