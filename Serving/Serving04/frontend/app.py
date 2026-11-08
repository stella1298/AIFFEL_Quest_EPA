# 실습 : 화면 (API 키 입력, 스트리밍 답, 근거 표시)
"""사내 문서 질의응답 화면.

실행 : streamlit run frontend/app.py
참고 : 인증과 화면 노트의 화면 장, 챗봇 프로젝트 부록 4장.
"""
import json
import os

import httpx
import streamlit as st

API_BASE = os.environ.get("D52_API_BASE", "http://127.0.0.1:8452")

st.set_page_config(page_title="사내 문서 질의응답", page_icon="📄", layout="centered")

def stream_answer(question, api_key):
    """SSE 를 받아 토큰을 하나씩 스트리밍으로 보냅니다. 근거는 st.session_state 에 담습니다."""
    st.session_state["마지막 근거"] = []
    try:
        with httpx.stream("POST", f"{API_BASE}/ask/sse",
                          json={"question": question},
                          headers={"X-API-Key": api_key}, timeout=180) as r:
            if r.status_code == 401:
                yield "인증에 실패했습니다. 사이드바의 API 키를 확인해 주십시오."
                return
            if r.status_code >= 400:
                yield f"요청이 거부되었습니다 (HTTP {r.status_code})."
                return
            event = "message"
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1])
                    if event == "start":
                        st.session_state["마지막 근거"] = data.get("sources", [])
                    elif event == "token":
                        yield data["text"]
                    elif event == "error":
                        yield f"\n\n({data.get('error', '오류')})"
    except httpx.ConnectError:
        yield "서버에 연결할 수 없습니다. 서버가 떠 있는지 확인해 주십시오."

# ===== 사이드바 =====
with st.sidebar:
    st.header("설정")
    # 1) 사이드바에 API 키 입력란을 만듭니다. 기본값은 실습용 키 dev-key-001
    api_key = st.text_input("API 키", value="dev-key-001", type="password")

    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3).json()
        if health.get("status") == "healthy":
            st.success(f"서버 연결됨 (문서 {health.get('문서 수', 0)}건)")
        else:
            st.warning("서버가 아직 준비 중입니다")
    except Exception:
        st.error("서버에 연결할 수 없습니다")

    if st.button("대화 지우기"):
        st.session_state["기록"] = []
        st.rerun()

# ===== 본문 =====
st.title("사내 문서 질의응답")
st.caption("사내망 안에서만 동작합니다. 계정에 따라 볼 수 있는 문서가 다릅니다.")

if "기록" not in st.session_state:
    st.session_state["기록"] = []

for 말 in st.session_state["기록"]:
    with st.chat_message(말["role"]):
        st.write(말["content"])
        for s in 말.get("sources", []):
            st.caption(f"근거 : {s['제목']} ({s['부서']})")

질문 = st.chat_input("사내 규정에 대해 물어보십시오")

if 질문:
    st.session_state["기록"].append({"role": "user", "content": 질문})
    with st.chat_message("user"):
        st.write(질문)

    with st.chat_message("assistant"):
        # 1) stream_answer 제너레이터를 st.write_stream 에 넘깁니다. 이어 붙인 전체 답이 돌아옵니다
        답 = st.write_stream(stream_answer(질문, api_key))
        # 2) start 이벤트에서 담아 둔 근거를 꺼내 제목과 부서를 답 아래에 표시합니다
        근거 = st.session_state.get("마지막 근거", [])
        for s in 근거:
            st.caption(f"근거 : {s['제목']} ({s['부서']})")

    st.session_state["기록"].append(
        {"role": "assistant", "content": 답, "sources": 근거})
