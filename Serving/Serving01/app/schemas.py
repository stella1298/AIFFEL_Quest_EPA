# [3.3 app/schemas.py] 입력/출력 규격 : 위반하면 추론 전에 FastAPI 가 422 로 거절
#   제목 2~200자(공백만 금지), top_k 1~7, 일괄 1~20건
"""
뉴스 섹션 분류 API 의 입력/출력 스키마.

모델(KLUE-YNAT)은 '뉴스 제목' 으로 학습되었습니다. 그래서 입력도 제목 길이에 맞춰
짧게 제한합니다. 본문 전체를 넣으면 길이만 늘고 정확도는 오르지 않습니다.
"""
from pydantic import BaseModel, Field, field_validator

# 학습 데이터(KLUE-YNAT)의 7개 섹션. /labels 와 화면에서 함께 씁니다
SECTIONS = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]

MAX_TITLE_LEN = 200      # 제목 하나의 최대 글자 수
MAX_BATCH = 20           # 한 번에 분류할 수 있는 제목 수


def _not_blank(value: str) -> str:
    """공백만 있는 문자열은 min_length 를 통과하므로 따로 막습니다."""
    value = value.strip()
    if not value:
        raise ValueError("공백만으로 된 제목은 분류할 수 없습니다.")
    return value


# ===== 요청 =====

class PredictRequest(BaseModel):
    """뉴스 제목 한 건 분류 요청"""
    title: str = Field(
        ..., min_length=2, max_length=MAX_TITLE_LEN,
        description="분류할 뉴스 제목 (2~200자)",
    )
    top_k: int = Field(
        default=3, ge=1, le=len(SECTIONS),
        description="함께 돌려받을 상위 후보 수 (1~7)",
    )

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        return _not_blank(value)

    model_config = {
        "json_schema_extra": {
            "examples": [{"title": "삼성전자, 차세대 AI 반도체 양산 돌입", "top_k": 3}]
        }
    }


class BatchPredictRequest(BaseModel):
    """뉴스 제목 여러 건 일괄 분류 요청"""
    titles: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH,
        description=f"분류할 뉴스 제목 목록 (1~{MAX_BATCH}건)",
    )

    @field_validator("titles")
    @classmethod
    def check_each_title(cls, titles: list[str]) -> list[str]:
        cleaned = []
        for i, t in enumerate(titles, start=1):
            t = t.strip()
            if len(t) < 2 or len(t) > MAX_TITLE_LEN:
                raise ValueError(f"{i}번째 제목의 길이가 2~{MAX_TITLE_LEN}자를 벗어났습니다.")
            cleaned.append(t)
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [{"titles": [
                "한국은행 기준금리 동결",
                "손흥민 시즌 10호 골",
                "여야, 예산안 두고 국회 대치",
            ]}]
        }
    }


# ===== 응답 =====

class SectionScore(BaseModel):
    """섹션 후보 하나와 그 확률"""
    section: str = Field(description="뉴스 섹션")
    score: float = Field(description="확률 (0~1)")


class PredictResponse(BaseModel):
    """한 건 분류 결과"""
    success: bool = True
    title: str = Field(description="분류한 제목 (앞뒤 공백 제거)")
    section: str = Field(description="가장 가능성 높은 섹션")
    confidence: float = Field(description="그 섹션의 확률 (0~1)")
    needs_review: bool = Field(description="확률이 기준보다 낮아 편집자 확인이 필요한지")
    candidates: list[SectionScore] = Field(description="상위 top_k 후보")
    user: str = Field(description="요청한 사용자 (API Key 소유자)")
    elapsed_ms: float = Field(description="추론에 걸린 시간 (밀리초)")


class BatchItem(BaseModel):
    title: str
    section: str
    confidence: float
    needs_review: bool


class BatchPredictResponse(BaseModel):
    """일괄 분류 결과"""
    success: bool = True
    count: int = Field(description="분류한 제목 수")
    results: list[BatchItem]
    section_counts: dict[str, int] = Field(description="섹션별 건수")
    review_count: int = Field(description="확인이 필요한 제목 수")
    user: str
    elapsed_ms: float
