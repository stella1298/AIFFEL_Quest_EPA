"""생성 층 : 예산과 이탈을 보면서 토큰을 내놓습니다.

비동기 처리 노트 7.2절의 guarded_tokens 를 그대로 가져왔습니다.
달라진 것은 둘입니다. 검색 결과(docs)를 밖에서 받고, 마지막 생성 결과를
STATS 에 남깁니다. STATS 는 운영 중에 "방금 무슨 일이 있었나" 를 보는 창입니다.
"""
import json
import logging
import os
import time

import anyio

from app.inference import generate_tokens

logger = logging.getLogger("ml_api")

# 상수를 박지 않습니다. 사내 서버마다 GPU도 정책도 다릅니다
MAX_TOKENS = int(os.environ.get("D52_MAX_TOKENS", "500"))
WALL_CLOCK = float(os.environ.get("D52_WALL_CLOCK", "30"))
CHECK_EVERY = int(os.environ.get("D52_CHECK_EVERY", "10"))
GPU_SLOTS = int(os.environ.get("D52_GPU_SLOTS", "2"))

gpu_limiter = anyio.CapacityLimiter(GPU_SLOTS)

STATS = {"동시 통과": GPU_SLOTS, "누적 요청": 0, "누적 토큰": 0, "마지막": None}

def new_report():
    return {"정지 사유": "진행 중", "토큰": 0, "대기 초": 0.0, "생성 초": 0.0}


def new_budget(max_tokens=None, deadline_seconds=None):
    """요청이 값을 주지 않으면 서버 기본값을 씁니다."""
    return {"max_tokens": max_tokens or MAX_TOKENS,
            "deadline": deadline_seconds or WALL_CLOCK}


def sse(event, data):
    """Server-Sent Events wire format 한 청크."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def guarded_tokens(question, docs, request, budget, report):
    """예산과 이탈을 보며 토큰을 내놓습니다. 정지 사유는 report 에 남깁니다."""
    it = generate_tokens(question, docs)
    done = object()
    queued_at = time.perf_counter()
    started = queued_at
    count = 0
    report["정지 사유"] = "진행 중"

    try:
        # 문지기. 동시에 GPU_SLOTS 건만 통과하고 나머지는 여기서 기다립니다
        async with gpu_limiter:
            started = time.perf_counter()
            report["대기 초"] = round(started - queued_at, 2)
            while True:
                token = await anyio.to_thread.run_sync(next, it, done)
                if token is done:
                    report["정지 사유"] = "완료"
                    break
                count += 1
                yield token
                if count >= budget["max_tokens"]:
                    report["정지 사유"] = "토큰 예산 소진"
                    break
                if time.perf_counter() - started >= budget["deadline"]:
                    report["정지 사유"] = "시간 예산 소진"
                    break
                if count % CHECK_EVERY == 0 and await request.is_disconnected():
                    report["정지 사유"] = "클라이언트 이탈"
                    break
    finally:
        it.close()                      # 여기가 GPU를 아끼는 한 줄입니다
        if report["정지 사유"] == "진행 중":
            report["정지 사유"] = "밖에서 끊김"
        report["토큰"] = count
        report["생성 초"] = round(time.perf_counter() - started, 2)
        STATS["누적 요청"] += 1
        STATS["누적 토큰"] += count
        STATS["마지막"] = dict(report)
        logger.info(f"생성 종료 : {report}")
