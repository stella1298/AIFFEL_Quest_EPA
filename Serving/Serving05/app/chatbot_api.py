"""
Day 7 - 한국어 챗봇 FastAPI 서버
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException

from app.chatbot_schemas import ChatRequest, ChatResponse
from app.chatbot_model import ChatbotModel
from app.auth import verify_api_key
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


# ===== 설정 =====
logger = setup_logger("chatbot_api")

app = FastAPI(
    title="Korean Chatbot API",
    description="한국어 GPT 기반 멀티턴 챗봇 API (인증 필요)",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chatbot")

# ===== 모델 로드 =====
chatbot = None

@app.on_event("startup")
async def startup():
    global chatbot
    import torch
    model_name = "skt/ko-gpt-trinity-1.2B-v0.5" if torch.cuda.is_available() else "skt/kogpt2-base-v2"
    logger.info(f"챗봇 모델 로드 중: {model_name}")
    chatbot = ChatbotModel(model_name)                # ChatbotModel 인스턴스 생성
    logger.info("모델 로드 완료")

def run_chat(messages, max_new_tokens, temperature):
    """별도 스레드에서 실행되는 추론 함수"""
    if chatbot is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    return chatbot.generate_response(
        messages=[m.model_dump() for m in messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ===== 엔드포인트 =====

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if chatbot else "loading",
        "model": chatbot.model_name if chatbot else None,
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    user: dict = Depends(verify_api_key),              # 인증 적용
):
    """대화 기록을 받아 응답을 생성합니다."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    logger.info(f"채팅 요청 — 사용자: {user['사용자']}, 메시지 수: {len(request.messages)}")

    try:
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(    # 비동기 추론
            inference_executor,
            run_chat,
            request.messages,
            request.max_new_tokens,
            request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 실패: {str(e)}")

    return ChatResponse(
        success=True,
        response=response_text,
        model_name=chatbot.model_name,
        user=user,
    )
