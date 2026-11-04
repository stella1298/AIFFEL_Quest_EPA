# [3.4 app/model_service.py] 모델 로드 + 단건/일괄 추론 + '검토 필요' 판정 (1위 확률 < 0.6)
"""
모델 로드와 추론 함수.

모델 : chunwoolee0/klue_ynat_roberta_base_model (KLUE-RoBERTa-base, KLUE-YNAT 7개 섹션)
- 원본 저장소는 pytorch_model.bin 만 있어 torch 2.6 미만에서는 읽히지 않습니다.
  (transformers 가 torch.load 취약점 CVE-2025-32434 때문에 막음)
- 허깅페이스 변환 봇이 올린 safetensors 판(refs/pr/2)을 revision 으로 지정해 해결합니다.

모델 이름, revision, 검토 기준값은 환경 변수로 바꿀 수 있습니다.
"""
import os
import time

from transformers import pipeline

MODEL_ID = os.environ.get("NEWS_MODEL_ID", "chunwoolee0/klue_ynat_roberta_base_model")
MODEL_REVISION = os.environ.get("NEWS_MODEL_REVISION", "refs/pr/2")

# 1위 확률이 이 값보다 낮으면 '편집자 검토 필요' 로 표시합니다
REVIEW_THRESHOLD = float(os.environ.get("NEWS_REVIEW_THRESHOLD", "0.6"))


def load_model():
    """분류 파이프라인을 만듭니다. 서버 시작 시 한 번만 부릅니다."""
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        top_k=None,          # 7개 섹션 확률을 모두 받습니다 (확률 내림차순)
        device=-1,           # CPU
    )


def _to_result(title: str, scores: list[dict], top_k: int) -> dict:
    """파이프라인 출력 한 건을 서비스 응답 형태로 바꿉니다."""
    best = scores[0]
    return {
        "title": title,
        "section": best["label"],
        "confidence": round(best["score"], 4),
        "needs_review": best["score"] < REVIEW_THRESHOLD,
        "candidates": [
            {"section": s["label"], "score": round(s["score"], 4)}
            for s in scores[:top_k]
        ],
    }


def predict(model, title: str, top_k: int = 3) -> dict:
    """제목 한 건을 분류합니다. (동기 함수 -> 서버에서 별도 스레드로 실행)"""
    start = time.perf_counter()
    scores = model(title, truncation=True)[0]
    result = _to_result(title, scores, top_k)
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


def predict_batch(model, titles: list[str]) -> dict:
    """제목 여러 건을 한 번에 분류합니다. 파이프라인에 리스트로 넘겨 배치로 처리합니다."""
    start = time.perf_counter()
    outputs = model(titles, truncation=True, batch_size=8)
    results = []
    counts: dict[str, int] = {}
    for title, scores in zip(titles, outputs):
        r = _to_result(title, scores, top_k=1)
        results.append({k: r[k] for k in ("title", "section", "confidence", "needs_review")})
        counts[r["section"]] = counts.get(r["section"], 0) + 1
    return {
        "count": len(results),
        "results": results,
        "section_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "review_count": sum(r["needs_review"] for r in results),
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }
