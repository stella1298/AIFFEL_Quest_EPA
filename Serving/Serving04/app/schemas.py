# 실습 1 : 요청과 응답의 규격 (AskRequest, Source, AskResponse)
"""요청과 응답의 규격.

참고 : FastAPI 기초 노트의 Request Body 절과 문서를 친절하게 만드는 절.
"""
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    """질의 요청."""
    # 1) question : 필수. 1자 이상 500자 이하 (통과 기준 7번과 8번)
    question: str = Field(
        ..., min_length=1, max_length=500,
        description="사내 규정에 대한 질문. 길수록 비쌉니다",
    )
    # 2) max_tokens : 선택. 1 이상 2000 이하. 기본값 None
    max_tokens: int | None = Field(
        default=None, ge=1, le=2000,
        description="이 요청에만 적용할 토큰 예산. 없으면 서버 기본값",
    )
    # 3) deadline_seconds : 선택. 0 초과 300 이하. 기본값 None
    deadline_seconds: float | None = Field(
        default=None, gt=0, le=300,
        description="이 요청에만 적용할 시간 예산(초). 없으면 서버 기본값",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "재택근무는 며칠까지 가능합니까",
                 "max_tokens": 200,
                 "deadline_seconds": 20.0}
            ]
        }
    }

class Source(BaseModel):
    """근거 문서 한 건."""
    # 1) 제목 : 사내 문서 제목
    제목: str = Field(description="사내 문서 제목")
    # 2) 부서 : 그 문서를 소유한 부서
    부서: str = Field(description="그 문서를 소유한 부서")


class AskResponse(BaseModel):
    """질의 응답."""
    success: bool = Field(default=True, description="요청 처리 성공 여부")
    answer: str = Field(description="생성된 답변")
    sources: list[Source] = Field(default_factory=list, description="근거로 쓴 문서")
    user: str = Field(description="인증된 사용자 이름")
    report: dict = Field(default_factory=dict, description="정지 사유, 토큰 수, 대기 초, 생성 초")
