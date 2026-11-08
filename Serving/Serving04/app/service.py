# 실습 2 : 서비스 층 (resolve_docs, to_sources, in_scope)
"""서비스 층 : 누가 무엇을 볼 수 있는지 정합니다.

참고 : 인증과 화면 노트의 Depends 절, 사내 RAG 노트의 검색 절.

판정 함수는 "모르면 불합격" 으로 만듭니다. 그 반대로 만들면
버그가 났을 때 조용히 통과합니다.
"""
from app.inference import retrieve

def resolve_docs(question, user):
    """사용자가 볼 수 있는 범위에서 근거 문서를 고릅니다."""
    # 1) user 의 부서 목록. user 가 None 이거나 부서 필드가 없으면 빈 목록
    부서 = (user or {}).get("부서") or []
    # 2) 부서 필드가 없는 user 는 빈 목록 (안전한 쪽)
    if not 부서:
        return []                      # 부서를 모르면 아무것도 보여주지 않습니다
    # 3) 부서 목록을 retrieve 에 넘겨 근거 문서를 받습니다
    return retrieve(question, 부서)


def to_sources(docs):
    """응답에 실을 근거 목록으로 바꿉니다. 본문은 빼고 제목과 부서만."""
    # 1) 본문은 넣지 않습니다. 응답이 무거워지고, 볼 필요도 없습니다
    return [{"제목": d["제목"], "부서": d["부서"]} for d in docs]


def in_scope(docs, user):
    """고른 문서가 전부 열람 범위 안인지 확인합니다. 기본값은 불합격입니다."""
    부서 = (user or {}).get("부서") or []
    if not 부서:
        return False
    # 1) 모든 문서의 부서가 user 의 부서 목록 안에 있는지
    return all(d.get("부서") in 부서 for d in docs)
