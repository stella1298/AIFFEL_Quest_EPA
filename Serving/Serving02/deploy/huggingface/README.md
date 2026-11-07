---
title: MNIST 6 Models
emoji: ✍️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: 손글씨 숫자를 모델 6개가 동시에 판단하는 서빙 데모
---

# ✍️ MNIST 손글씨 숫자 인식 — 모델 6개 비교

검은 칸에 숫자를 그리면 생각하는 방식이 다른 여섯 모델이 동시에 판단합니다.
로지스틱 회귀 · MLP · LeNet-5 · CNN · LSTM(이미지를 문장처럼) · ViT(트랜스포머)
확률을 평균해 최종 답을 내고, 의견이 갈리거나 확신이 낮으면 '사람이 확인할 대상'으로 표시합니다.

- 그림판 : 이 Space 의 첫 화면
- API 문서 : `/docs`
- 소스 · 설계 문서 : https://github.com/stella1298/AIFFEL_Quest_EPA/tree/master/Serving/Serving02
