# ===== API 서버 이미지 =====
# torch 를 넣지 않습니다. ONNX 모델 + onnxruntime 만으로 추론하므로 이미지가 작고 빨리 뜹니다
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

# 1) 패키지 목록만 먼저 → 코드가 바뀌어도 이 무거운 레이어는 캐시를 그대로 씁니다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) 모델을 이미지 안에 넣습니다 (첫 기동 때 내려받지 않음 → 어디서 띄워도 같은 모델)
COPY models/*.onnx models/metrics.json ./models/
# 3) 자주 바뀌는 코드는 맨 뒤에
COPY app/ ./app/

# root 가 아닌 사용자로 실행
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/ready', timeout=2)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log --proxy-headers --forwarded-allow-ips='*'"]
