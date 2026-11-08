# [5. 모듈화 app/housing_model.py] 서버가 import 할 모델 파일
#   - HousingModel     : 학습 때와 똑같은 구조여야 state_dict 를 올릴 수 있습니다
#   - HousingPredictor : 모델 로드 + 전처리 + 추론을 한 클래스로 묶습니다 (서버는 predict() 만 부름)
"""
Day 5 - 주택 가격 예측 모델 정의 + 추론 함수
"""
import json
import torch
import torch.nn as nn
import numpy as np


class HousingModel(nn.Module):
    """캘리포니아 주택 가격 예측 모델"""
    def __init__(self, input_dim=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


class HousingPredictor:
    """모델 로드 + 전처리 + 추론을 캡슐화한 클래스"""

    def __init__(self, model_path: str, preprocessing_path: str):
        # 전처리 파라미터 로드
        with open(preprocessing_path, "r") as f:
            params = json.load(f)
        self.mean = np.array(params["mean"])
        self.std = np.array(params["std"])
        self.feature_names = params["feature_names"]

        # 모델 로드
        self.model = HousingModel(input_dim=len(self.feature_names))
        self.model.load_state_dict(
            # map_location="cpu" : GPU 에서 저장한 가중치도 CPU 서버에서 읽게 함
            # weights_only=True : 가중치만 읽어 임의 코드 실행을 막음 (보안)
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    # 추론 순서 : dict -> 학습 때 피처 순서로 배열 -> 학습 통계로 정규화 -> (1, 8) 텐서 -> 모델
    def predict(self, features: dict) -> dict:
        """
        피처 딕셔너리를 받아 가격을 예측합니다.

        Args:
            features: {"MedInc": 3.5, "HouseAge": 25, ...}
        Returns:
            {"predicted_price": 2.35, "predicted_price_usd": 235000}
        """
        # 피처를 올바른 순서로 배열
        values = [features[name] for name in self.feature_names]  # *your code* — 피처 순서 맞추기

        # 정규화
        values = np.array(values, dtype=np.float32)
        normalized = (values - self.mean) / self.std              # *your code* — 정규화 적용

        # 추론
        input_tensor = torch.FloatTensor(normalized).unsqueeze(0)  # (1, 8)
        with torch.no_grad():
            output = self.model(input_tensor)

        price = output.item()
        price = max(price, 0.0)  # 음수 방지

        return {
            "predicted_price": round(price, 4),
            "predicted_price_usd": int(price * 100000),
        }
