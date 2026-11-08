"""사내 문서 질의응답 : 검색과 생성.

진짜 서비스에서는 사내 RAG 노트에서 만든 벡터 검색기와, 사내 데이터로 학습한
어댑터를 붙인 모델이 여기 들어갑니다. 오늘의 주제는 서버 조립이라 생성부만
mock을 씁니다. 바꿔 끼우는 자리는 generate_tokens() 하나입니다.
"""
import json
import os
import time
from pathlib import Path

# 생성 속도. 진짜 모델이라면 GPU와 모델 크기가 정하는 값입니다
TOKENS_PER_SECOND = float(os.environ.get("D52_TPS", "20"))

# 한 토큰에 몇 글자로 볼 것인가. 한국어는 보통 한두 글자가 한 토큰입니다
CHARS_PER_TOKEN = 3

DOC_PATH = Path(os.environ.get("D52_DOCS", "data/사내문서.json"))

# 근거로 채택할 최소 겹침 수. 낮추면 엉뚱한 문서가 붙고 높이면 답을 못 합니다
MIN_HITS = int(os.environ.get("D52_MIN_HITS", "3"))


def load_docs():
    """사내 문서를 읽습니다."""
    return json.loads(DOC_PATH.read_text(encoding="utf-8"))

def _grams(text, n=2):
    """두 글자씩 잘라 집합으로 만듭니다."""
    t = "".join(text.split())
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def retrieve(question, allowed_depts, k=2):
    """볼 수 있는 부서의 문서 중에서 질문과 겹치는 청크를 고릅니다.

    부서로 거르는 일이 점수를 매기는 일보다 **먼저** 옵니다.
    순서가 뒤집히면 볼 수 없는 문서가 프롬프트에 들어갑니다.

    진짜 서비스에서는 이 자리에 사내 RAG 노트의 벡터 검색이 들어가고,
    부서 조건은 벡터DB 의 메타데이터 필터가 됩니다.
    """
    docs = [d for d in load_docs() if d["부서"] in allowed_depts]
    q = _grams(question)
    scored = []
    for d in docs:
        hits = len(q & _grams(d["제목"] + " " + d["본문"]))
        if hits >= MIN_HITS:
            scored.append((hits, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]

def _compose(question, docs):
    """근거를 붙여 답변 문장을 만듭니다."""
    if not docs:
        # 실패를 예외가 아니라 답변으로 다룹니다. 서버는 200 을 돌려주고,
        # 직원은 "권한이 없다" 가 아니라 "근거를 못 찾았다" 를 읽습니다
        return ("문의하신 내용은 지금 계정으로 열람할 수 있는 사내 문서에서 "
                "근거를 찾지 못했습니다. 문서 담당 부서에 열람 권한을 요청하시거나 "
                "질문을 더 구체적으로 적어 다시 문의해 주십시오.")
    근거 = " ".join(d["본문"] for d in docs)
    return (f"문의하신 '{question}'에 대해 사내 규정을 확인했습니다. {근거} "
            f"자세한 내용은 담당 부서에 문의하시기 바랍니다.")

def generate_tokens(question, docs):
    """답변을 토큰 하나씩 내놓는 동기 제너레이터.

    ===== 진짜 모델로 바꾸는 자리는 여기 하나입니다 =====
    transformers 의 TextIteratorStreamer 를 쓰면 model.generate 를 별도
    스레드에서 돌리면서 이 함수와 똑같이 토큰을 하나씩 받아올 수 있습니다.
    바깥에서 보는 모양(동기 제너레이터, next() 한 번이 토큰 하나)은 같습니다.
    프롬프트에는 docs 의 본문만 넣습니다. 부서 확인은 이미 끝나 있습니다.
    """
    text = _compose(question, docs)
    for i in range(0, len(text), CHARS_PER_TOKEN):
        time.sleep(1.0 / TOKENS_PER_SECOND)
        yield text[i:i + CHARS_PER_TOKEN]


def generate(question, docs):
    """답변 전체를 한 번에 돌려줍니다. model.generate() 자리입니다."""
    return "".join(generate_tokens(question, docs))
