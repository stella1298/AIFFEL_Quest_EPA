# [3.6 frontend/app.py] Streamlit 화면
#   사이드바 : API Key 입력, 서버 상태 / 탭 1 : 제목 한 건 분류 / 탭 2 : 일괄 분류(CSV 업로드·내려받기)
#   실행 : streamlit run frontend/app.py  -> http://localhost:8501
"""
뉴스 섹션 자동 분류 화면 (Streamlit).

실행 : streamlit run frontend/app.py   (FastAPI 서버가 먼저 떠 있어야 합니다)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("NEWS_API_BASE", "http://localhost:8000")

SAMPLE_TITLES = [
    "삼성전자, 차세대 AI 반도체 HBM4 양산 돌입",
    "한국은행 기준금리 0.25%p 인하…가계대출 부담 줄어",
    "서울 지하철 노조 내일 총파업 예고…출근길 혼잡 우려",
    "올가을 단풍 절정 이번 주말…설악산 인파 몰려",
    "일본 도쿄서 규모 5.8 지진…인명 피해 없어",
    "손흥민 시즌 10호 골…토트넘 3연승",
    "여야, 내년도 예산안 처리 두고 국회 대치",
    "정부, AI 스타트업 육성에 1조 원 투입",        # IT과학/정치 사이 -> 검토 필요 예시
    "서울시장, 한강 리버버스 운항 개시",            # 사회/생활문화 사이 -> 검토 필요 예시
]

SECTION_ICON = {
    "IT과학": "💻", "경제": "💰", "사회": "🏙️", "생활문화": "🎨",
    "세계": "🌏", "스포츠": "⚽", "정치": "🏛️",
}

st.set_page_config(page_title="뉴스 섹션 자동 분류", page_icon="📰", layout="wide")


# ===== API 호출 =====
def call_api(path, payload=None, api_key=None, method="post"):
    """API 를 호출하고 (데이터, 에러 메시지) 를 돌려줍니다. 상태 코드별로 안내 문구를 다르게 씁니다."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        if method == "get":
            resp = requests.get(f"{API_BASE}{path}", timeout=10)
        else:
            resp = requests.post(f"{API_BASE}{path}", json=payload, headers=headers, timeout=60)
    except requests.exceptions.ConnectionError:
        return None, "🔌 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해 주세요."
    except requests.exceptions.Timeout:
        return None, "⏱️ 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."

    if resp.status_code == 200:
        return resp.json(), None
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code == 401:
        return None, f"🔑 인증 실패 (401) : {body.get('detail', '')} 사이드바의 API Key 를 확인해 주세요."
    if resp.status_code == 422:
        lines = [f"- `{d['field']}` : {d['message']}" for d in body.get("details", [])]
        return None, "✏️ 입력값 오류 (422)\n" + "\n".join(lines)
    if resp.status_code == 503:
        return None, "⏳ 모델을 불러오는 중입니다 (503). 잠시 후 다시 시도해 주세요."
    return None, f"❌ 서버 오류 (HTTP {resp.status_code}) : {body.get('error') or body.get('detail', '')}"


# ===== 사이드바 =====
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("API Key", value="test-key-001", type="password",
                            help="실습용 키 : test-key-001, test-key-002")

    health, err = call_api("/health", method="get")
    if health and health["status"] == "healthy":
        st.success("🟢 서버 연결됨 · 모델 준비 완료")
        st.caption(f"모델 : `{health['model']}`")
        st.caption(f"로드 시각 : {health['loaded_at']} ({health['load_seconds']}초)")
    elif health:
        st.warning("🟡 서버는 떠 있지만 모델을 불러오는 중입니다.")
    else:
        st.error("🔴 서버 연결 실패")

    labels, _ = call_api("/labels", method="get")
    if labels:
        st.divider()
        st.caption("분류 가능한 섹션")
        st.write(" ".join(f"{SECTION_ICON.get(s, '')}{s}" for s in labels["sections"]))
        st.caption(f"확률 {labels['review_threshold']:.0%} 미만은 '검토 필요' 로 표시합니다.")


# ===== 메인 =====
st.title("📰 뉴스 섹션 자동 분류")
st.write("기사 제목을 넣으면 KLUE-YNAT 7개 섹션 중 어디에 실을지 추천합니다. "
         "확신이 낮은 제목은 편집자가 확인하도록 표시합니다.")

tab_one, tab_batch = st.tabs(["📝 제목 한 건 분류", "📋 여러 건 일괄 분류"])

# ----- 탭 1 : 한 건 -----
with tab_one:
    col_in, col_out = st.columns([1, 1])
    with col_in:
        sample = st.selectbox("예시 제목 (선택하면 아래 입력란에 채워집니다)", ["직접 입력"] + SAMPLE_TITLES)
        title = st.text_input("기사 제목", value="" if sample == "직접 입력" else sample,
                              max_chars=200, placeholder="예) 코스피 2,800선 회복, 외국인 순매수 이어져")
        top_k = st.slider("후보 섹션 수 (top_k)", 1, 7, 3)
        run_one = st.button("🔍 분류하기", type="primary", width="stretch")

    with col_out:
        if run_one:
            with st.spinner("분류 중..."):
                data, err = call_api("/predict", {"title": title, "top_k": top_k}, api_key)
            st.session_state["one_result"] = (data, err)

        if "one_result" in st.session_state:
            data, err = st.session_state["one_result"]
            if err:
                st.error(err)
            else:
                icon = SECTION_ICON.get(data["section"], "")
                c1, c2 = st.columns(2)
                c1.metric("추천 섹션", f"{icon} {data['section']}")
                c2.metric("확률", f"{data['confidence']:.1%}")
                if data["needs_review"]:
                    st.warning("⚠️ 확신이 낮습니다. 편집자 확인이 필요합니다.")
                else:
                    st.success("✅ 자동 배정 가능")
                chart = pd.DataFrame(data["candidates"]).set_index("section")
                st.bar_chart(chart, horizontal=True)
                st.caption(f"요청자 : {data['user']} · 추론 {data['elapsed_ms']}ms")

# ----- 탭 2 : 여러 건 -----
with tab_batch:
    st.write("한 줄에 제목 하나씩 입력하거나, `title` 열이 있는 CSV 를 올립니다. (최대 20건)")
    text = st.text_area("기사 제목 목록", value="\n".join(SAMPLE_TITLES), height=200)
    uploaded = st.file_uploader("또는 CSV 업로드", type=["csv"])

    if st.button("📋 일괄 분류", type="primary"):
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                titles = df_in["title"].dropna().astype(str).tolist()
            except Exception:
                titles = None
                st.error("CSV 를 읽지 못했습니다. `title` 열이 있는지 확인해 주세요.")
        else:
            titles = [line for line in text.splitlines() if line.strip()]

        if titles is not None:
            with st.spinner(f"{len(titles)}건 분류 중..."):
                data, err = call_api("/predict/batch", {"titles": titles}, api_key)
            st.session_state["batch_result"] = (data, err)

    if "batch_result" in st.session_state:
        data, err = st.session_state["batch_result"]
        if err:
            st.error(err)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("분류한 제목", f"{data['count']}건")
            c2.metric("검토 필요", f"{data['review_count']}건")
            c3.metric("처리 시간", f"{data['elapsed_ms'] / 1000:.1f}초")

            df = pd.DataFrame(data["results"])
            df["confidence"] = (df["confidence"] * 100).round(1)
            df["needs_review"] = df["needs_review"].map({True: "⚠️ 검토", False: "✅"})
            df.columns = ["제목", "섹션", "확률(%)", "상태"]

            left, right = st.columns([2, 1])
            left.dataframe(df, width="stretch", hide_index=True)
            right.write("**섹션별 건수**")
            right.bar_chart(pd.Series(data["section_counts"], name="건수"))
            st.download_button("⬇️ 결과 CSV 내려받기", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="news_sections.csv", mime="text/csv")
