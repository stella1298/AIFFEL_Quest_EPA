"""부하 테스트 : 동시에 여러 요청을 보내 지연 시간과 상태 코드를 재고,
그 사이에 /health 가 계속 빨리 응답하는지(이벤트 루프가 막히지 않는지) 확인합니다.

주의 : /health 는 '별도 프로세스'에서 잽니다. 부하를 만드는 이 스크립트의 이벤트 루프 안에서 재면,
       서버가 아니라 이 스크립트 자신이 바빠서 늦어진 시간까지 섞여 100ms 가 넘게 나옵니다(직접 겪은 함정).

실행 : python scripts/load_test.py --url http://127.0.0.1:8000 --key dev-admin-key --concurrency 16 --requests 400
"""
import argparse
import asyncio
import multiprocessing as mp
import base64
import io
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def sample_images(n=50):
    data = np.load(ROOT / "samples" / "mnist_test_200.npz")
    out = []
    for arr in data["images"][:n]:
        canvas = Image.new("L", (280, 280), 0)
        canvas.paste(Image.fromarray(arr).resize((140, 140)), (70, 70))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def probe_health(url, stop, out):
    """(별도 프로세스) 0.1초마다 /health 를 부르고 걸린 시간을 모읍니다."""
    times = []
    with httpx.Client(base_url=url, timeout=10) as c:
        while not stop.is_set():
            t0 = time.perf_counter()
            c.get("/health")
            times.append((time.perf_counter() - t0) * 1000)
            time.sleep(0.1)
    out.put(times)


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))] if values else float("nan")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="dev-admin-key")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=400)
    args = ap.parse_args()

    images = sample_images()
    latencies, codes = [], Counter()
    queue = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait(i)

    stop, results = mp.Event(), mp.Queue()
    prober = mp.Process(target=probe_health, args=(args.url, stop, results))
    prober.start()

    async with httpx.AsyncClient(base_url=args.url, timeout=30,
                                 limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def worker():
            while not queue.empty():
                i = queue.get_nowait()
                t0 = time.perf_counter()
                r = await client.post("/v1/predict", headers={"X-API-Key": args.key},
                                      json={"image": images[i % len(images)], "return_preview": False})
                latencies.append((time.perf_counter() - t0) * 1000)
                codes[r.status_code] += 1

        await asyncio.sleep(0.3)                      # 측정 프로세스가 먼저 자리를 잡게
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        total = time.perf_counter() - t0
    stop.set()
    health_ms = results.get()
    prober.join()

    print(f"요청 {args.requests}건 · 동시 {args.concurrency} · 총 {total:.2f}초 · 처리량 {args.requests / total:.0f} req/s")
    print(f"상태 코드 : {dict(codes)}")
    print(f"/v1/predict 지연(ms) : p50 {statistics.median(latencies):.1f} · p95 {pct(latencies, .95):.1f} · 최대 {max(latencies):.1f}")
    print(f"부하 중 /health(ms, 별도 프로세스에서 측정) : p50 {statistics.median(health_ms):.1f} · p95 {pct(health_ms, .95):.1f} · {len(health_ms)}회")


if __name__ == "__main__":
    asyncio.run(main())
