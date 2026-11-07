"""MNIST 숫자 분류 모델 여섯 가지. 같은 입력 (N, 1, 28, 28) → 같은 출력 (N, 10) 입니다.

| 키     | 계열       | 이미지를 보는 방식                                  |
|--------|------------|-----------------------------------------------------|
| logreg | 선형       | 픽셀마다 가중치를 곱해 더함 (은닉층 없음)            |
| mlp    | 완전연결   | 784개 픽셀을 한 줄로 펼침 → 위치를 하나하나 외움     |
| lenet  | 합성곱     | 5x5 필터 6·16개 (1998년 LeNet-5)                     |
| cnn    | 합성곱     | 3x3 필터 32·64개 (REST API 노트의 SimpleClassifier)  |
| lstm   | 순환       | 28줄을 순서대로 읽음 (한 줄 = 단어 하나처럼)         |
| vit    | 트랜스포머 | 7x7 조각 16개 = 토큰 16개, 셀프 어텐션               |

서빙 쪽(app/)은 이 파일을 import 하지 않습니다. ONNX 로 내보낸 파일만 읽기 때문입니다.
"""
import math

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """가장 단순한 기준선. 숫자마다 '어느 픽셀이 밝으면 이 숫자다'라는 가중치 한 장씩."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, num_classes))

    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    """다층 퍼셉트론. 픽셀의 '위치'를 하나하나 따로 외우므로 숫자가 몇 칸만 밀려도 약해집니다."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeNet5(nn.Module):
    """LeCun 등(1998)의 LeNet-5. 활성화만 tanh 대신 ReLU, 평균 풀링 대신 최대 풀링으로 바꿨습니다."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),             # 14 -> 10 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNClassifier(nn.Module):
    """합성곱 모델. REST API 노트의 SimpleClassifier 와 같은 구조입니다."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LSTMClassifier(nn.Module):
    """이미지를 '문장'처럼 읽습니다. 28줄 각각이 단어 하나(28차원)이고, 마지막 줄까지 읽은 뒤의 기억으로 판단합니다.

    NLP 에서 문장을 단어 순서대로 읽던 RNN 과 똑같은 구조입니다. 바뀐 것은 입력이 단어 벡터가 아니라 픽셀 한 줄이라는 점뿐.
    """

    def __init__(self, num_classes: int = 10, hidden: int = 128):
        super().__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        rows = x.reshape(x.shape[0], 28, 28)          # (N, 줄 28개, 줄마다 픽셀 28개) = (배치, 시퀀스 길이, 단어 차원)
        out, _ = self.rnn(rows)
        return self.fc(out[:, -1])                     # 마지막 줄까지 읽은 상태


class SelfAttention(nn.Module):
    """멀티헤드 셀프 어텐션 : Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V  (트랜스포머 노트의 수식 그대로)"""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.qkv(x).reshape(n, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (N, 헤드, 토큰, 토큰)
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(n, t, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN 트랜스포머 인코더 블록 : x + Attention(LN(x)) → x + FFN(LN(x))"""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinyViT(nn.Module):
    """작은 Vision Transformer. 28x28 을 7x7 조각 16개로 잘라 각각을 '토큰'으로 봅니다.

    BERT 처럼 맨 앞에 [CLS] 토큰을 붙이고, 그 자리의 출력으로 분류합니다.
    합성곱과 달리 '가까운 픽셀끼리 관련 있다'는 가정(귀납 편향)이 없어서, 같은 학습량에서는 CNN 보다 불리합니다.
    """

    def __init__(self, num_classes: int = 10, dim: int = 64, depth: int = 4, heads: int = 4,
                 mlp_dim: int = 128, patch: int = 7, dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        self.n_patches = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, dim)                     # 조각 → 토큰 벡터
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))               # [CLS]
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, dim) * 0.02)   # 위치 임베딩
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        n, g, p = x.shape[0], 28 // self.patch, self.patch
        patches = x.reshape(n, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(n, g * g, p * p)   # (N, 16, 49)
        tokens = torch.cat([self.cls.expand(n, -1, -1), self.embed(patches)], dim=1) + self.pos
        return self.head(self.norm(self.blocks(tokens)[:, 0]))


MODELS = {"logreg": LogisticRegression, "mlp": MLPClassifier, "lenet": LeNet5,
          "cnn": CNNClassifier, "lstm": LSTMClassifier, "vit": TinyViT}
