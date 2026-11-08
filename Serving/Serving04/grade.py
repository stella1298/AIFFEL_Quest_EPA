"""통과 기준 채점기.

1.3절의 열두 항목을 실제로 재서 통과 여부를 찍습니다.
실패를 예외로 만들지 않습니다. 한 항목이 터져도 나머지를 계속 잽니다.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

VALID = "hr-key-003"
SALES = "sales-key-002"
HR_Q = "인사 평가 등급은 어떻게 정합니까"
OPEN_Q = "재택근무는 며칠까지 가능합니까"


class _Abort(Exception):
    pass


def _post(base, path, body, key=VALID, timeout=180):
    headers = {"X-API-Key": key} if key else {}
    return httpx.post(base + path, json=body, headers=headers, timeout=timeout)

def _stream_tokens(base, body, stop_after=None):
    """SSE 를 받습니다. stop_after 초가 지나면 끊습니다. (첫 청크 시각, 청크 수)"""
    t0 = time.perf_counter()
    first, chunks = None, 0
    try:
        with httpx.stream("POST", base + "/ask/sse", json=body,
                          headers={"X-API-Key": VALID}, timeout=180) as r:
            event = "message"
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    if event == "token":
                        chunks += 1
                        if first is None:
                            first = time.perf_counter() - t0
                if stop_after and time.perf_counter() - t0 >= stop_after:
                    raise _Abort()
    except _Abort:
        pass
    return first, chunks


def run(base):
    결과 = []

    def 기록(번호, 항목, 기준, 측정, 통과):
        결과.append((번호, 항목, 기준, str(측정), bool(통과)))

    def 안전하게(번호, 항목, 기준, fn):
        try:
            측정, 통과 = fn()
        except Exception as exc:
            측정, 통과 = f"{type(exc).__name__}: {exc}"[:40], False
        기록(번호, 항목, 기준, 측정, 통과)

    # 1. 헬스체크
    def c1():
        t0 = time.perf_counter()
        r = httpx.get(base + "/health", timeout=10)
        걸림 = time.perf_counter() - t0
        return (f"{r.status_code}, {걸림:.2f}초",
                r.status_code == 200 and 걸림 < 0.5
                and r.json().get("status") == "healthy")
    안전하게(1, "헬스체크", "200, 0.5초 이내", c1)

    # 2. 정상 질의   3. 근거 표시
    상태 = {"r": None}

    def c2():
        r = _post(base, "/ask", {"question": OPEN_Q})
        상태["r"] = r
        답 = r.json().get("answer", "") if r.status_code == 200 else ""
        return f"{r.status_code}, {len(답)}자", r.status_code == 200 and len(답) >= 30
    안전하게(2, "정상 질의", "200, 답변 30자 이상", c2)

    def c3():
        r = 상태["r"]
        sources = r.json().get("sources", []) if r is not None else []
        return f"{len(sources)}건", len(sources) >= 1
    안전하게(3, "근거 표시", "sources 1건 이상", c3)

    # 4. 키 없음   5. 잘못된 키
    안전하게(4, "키 없음", "401", lambda: (
        (lambda s: (s, s == 401))(_post(base, "/ask", {"question": OPEN_Q},
                                        key=None).status_code)))
    # 키를 한글로 쓰지 마십시오. HTTP 헤더는 latin-1 로만 보낼 수 있어서
    # 서버에 닿기도 전에 클라이언트가 UnicodeEncodeError 로 터집니다
    안전하게(5, "잘못된 키", "401", lambda: (
        (lambda s: (s, s == 401))(_post(base, "/ask", {"question": OPEN_Q},
                                        key="wrong-key-999").status_code)))

    # 6. 부서 밖 문서 (두 쪽을 함께 봅니다)
    def c6():
        영업 = _post(base, "/ask", {"question": HR_Q, "max_tokens": 20}, key=SALES)
        인사 = _post(base, "/ask", {"question": HR_Q, "max_tokens": 20}, key=VALID)
        영업근거 = [s["부서"] for s in 영업.json().get("sources", [])]
        인사근거 = [s["부서"] for s in 인사.json().get("sources", [])]
        샘 = [d for d in 영업근거 if d == "인사"]
        return (f"영업 계정 인사 문서 {len(샘)}건, 인사 계정 근거 {len(인사근거)}건",
                len(샘) == 0 and len(인사근거) >= 1)
    안전하게(6, "부서 밖 문서", "영업 0건 / 인사 1건 이상", c6)

    # 7. 빈 질문   8. 너무 긴 질문
    안전하게(7, "빈 질문", "422", lambda: (
        (lambda s: (s, s == 422))(_post(base, "/ask", {"question": ""}).status_code)))
    안전하게(8, "너무 긴 질문", "422", lambda: (
        (lambda s: (s, s == 422))(_post(base, "/ask",
                                        {"question": "가" * 501}).status_code)))

    # 9. 내부 오류
    def c9():
        r = _post(base, "/ask/broken", {"question": OPEN_Q})
        본문 = r.text
        샘 = [단어 for 단어 in ["Traceback", ".py", "RuntimeError", "/app"]
             if 단어 in 본문]
        return f"{r.status_code}, 새는 단어 {샘}", r.status_code == 500 and not 샘
    안전하게(9, "내부 오류", "500, 내부 정보 0곳", c9)

    # 10. 동시 10건
    def c10():
        def 한건(i):
            return _post(base, "/ask",
                         {"question": OPEN_Q, "max_tokens": 40}).status_code
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as ex:
            코드들 = list(ex.map(한건, range(10)))
        걸림 = time.perf_counter() - t0
        살아있음 = httpx.get(base + "/health", timeout=10).status_code == 200
        성공 = sum(1 for c in 코드들 if c == 200)
        return (f"200 이 {성공}/10건, {걸림:.1f}초, 헬스체크 {'정상' if 살아있음 else '실패'}",
                성공 == 10 and 살아있음)
    안전하게(10, "동시 10건", "전부 200, 서버 생존", c10)

    # 11. 스트리밍 첫 청크
    def c11():
        first, chunks = _stream_tokens(base, {"question": OPEN_Q, "max_tokens": 40})
        return f"첫 청크 {first:.2f}초, 청크 {chunks}개", first is not None and first < 1.0
    안전하게(11, "스트리밍", "첫 청크 1.0초 이내", c11)

    # 12. 취소
    def c12():
        _stream_tokens(base, {"question": OPEN_Q})            # 끝까지
        전체 = httpx.get(base + "/stats", timeout=10).json()["마지막"]["토큰"]
        _stream_tokens(base, {"question": OPEN_Q}, stop_after=0.8)
        time.sleep(1.0)                                        # 서버가 정리할 시간
        끊김 = httpx.get(base + "/stats", timeout=10).json()["마지막"]
        return (f"전체 {전체}토큰 -> 끊었을 때 {끊김['토큰']}토큰 ({끊김['정지 사유']})",
                전체 > 0 and 끊김["토큰"] < 전체 / 2)
    안전하게(12, "취소", "만든 토큰이 절반 미만", c12)

    # ===== 출력 =====
    print("=" * 78)
    print(f"{'#':>2} {'항목':<14} {'기준':<24} {'측정':<26} 판정")
    print("-" * 78)
    for 번호, 항목, 기준, 측정, 통과 in 결과:
        print(f"{번호:>2} {항목:<14} {기준:<24} {측정:<26} {'통과' if 통과 else '불통과'}")
    print("-" * 78)
    통과수 = sum(1 for *_, ok in 결과 if ok)
    print(f"  {len(결과)}개 중 {통과수}개 통과")
    if 통과수 == len(결과):
        print("  전부 통과했습니다. 제출할 수 있습니다")
    else:
        안된것 = [str(번호) for 번호, *_, ok in 결과 if not ok]
        print(f"  통과하지 못한 항목 : {', '.join(안된것)}번. 1.3절 표에서 그 줄을 보십시오")
    return 통과수 == len(결과)
