"""모델 목록 (이름·계열·설명·화면 색). 학습 스크립트·서버·두 화면이 모두 이 파일 하나를 봅니다.

새 모델을 추가할 때
  1) training/models.py 에 클래스를 만들고 MODELS 에 등록
  2) 여기 CATALOG 에 한 줄 추가
  3) python -m training.train --models <키>  → models/mnist_<키>.onnx 가 생기면 서버가 자동으로 읽습니다
torch 를 import 하지 않으므로 서빙 이미지에서도 그대로 쓸 수 있습니다.
"""

CATALOG = {
    # 키 : (화면 이름, 계열, 한 줄 설명, 색)
    "logreg": ("로지스틱 회귀", "선형", "은닉층 없이 픽셀에 가중치를 곱해 더하기만 합니다. 다른 모델의 기준선", "#7a8599"),
    "mlp":    ("MLP", "완전연결", "784개 픽셀을 한 줄로 펼쳐 은닉층 2개로 판단합니다", "#d9701a"),
    "lenet":  ("LeNet-5", "합성곱 (1998)", "우편번호 인식에 쓰인 원조 CNN. 파라미터가 아주 적습니다", "#2a9d8f"),
    "cnn":    ("CNN", "합성곱", "3x3 필터 32·64개로 이웃 픽셀의 모양을 찾습니다", "#2f6fdf"),
    "lstm":   ("LSTM", "순환 (RNN)", "28줄을 위에서 아래로 한 줄씩, 문장을 읽듯 순서대로 읽습니다", "#9b59b6"),
    "vit":    ("ViT", "트랜스포머", "7x7 조각 16개를 토큰처럼 보고 셀프 어텐션으로 관계를 봅니다", "#c0397b"),
}

ORDER = list(CATALOG)

# 최종 답 투표에 참여하는 기본 모델. 나머지는 화면에 '참고'로만 보여 줍니다.
# training/ensemble_eval.py 로 모든 조합(63가지)을 채점해 고른 조합입니다 (구조가 서로 다른 상위 세 모델).
# 환경 변수 ENSEMBLE_MODELS 로 바꿀 수 있습니다.
DEFAULT_ENSEMBLE = ["lenet", "cnn", "lstm"]


def info(key: str) -> dict:
    name, family, desc, color = CATALOG[key]
    return {"key": key, "name": name, "family": family, "description": desc, "color": color}
