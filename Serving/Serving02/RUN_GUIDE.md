# 구동 가이드

목적에 맞는 방법 하나를 고르세요.

| 하고 싶은 것 | 방법 | 주소 | 걸리는 시간 |
|---|---|---|---|
| 내 PC 에서 코드 고치며 실행 | [1. 로컬 파이썬](#1-로컬-파이썬으로-실행) | localhost | 2분 |
| 모델을 켜고 끄거나 투표 모델 바꾸기 | [1-1. 모델 설정](#1-1-모델-켜고-끄기--투표-모델-바꾸기) | — | 1분 |
| **내 손글씨로 모델 채점** | [1-2. 내 손글씨 채점](#1-2-내-손글씨로-모델-채점하기) | `localhost:8000/#score` | 5분 |
| 내 PC 에서 운영처럼 실행 | [2. Docker](#2-docker-로-실행) | localhost | 3분 (첫 빌드) |
| 발표·공유용 **임시 공개 주소** | [3-2. Cloudflare 터널](#3-2-임시-공개-주소-cloudflare-quick-tunnel) | `https://xxxx.trycloudflare.com` | 1분 |
| 같은 와이파이의 휴대폰으로 | [3-1. 같은 네트워크](#3-1-같은-네트워크의-다른-기기에서) | `http://내PC IP:8000` | 2분 |
| **항상 켜져 있는 무료 주소** | [3-4. Hugging Face Spaces](#3-4-hugging-face-spaces-무료-상시-주소) | `https://아이디-이름.hf.space` | 10분 |
| 내 서버(클라우드 VM)에 올리기 | [3-3. 클라우드 VM](#3-3-클라우드-vm-aws--gcp--runpod-등) | `http://VM IP:8000` | 15분 |
| **대시보드를 무료 상시 주소로** | [3-5. Streamlit Community Cloud](#3-5-streamlit-community-cloud-xxxstreamlitapp) | `https://원하는이름.streamlit.app` | 5분 |
| PC 설치 없이 GPU 서버에서 | [3-6. Colab](#3-6-colab-에서-실행) | trycloudflare 주소 | 5분 |

> 기본 API 키 : 관리자 `dev-admin-key`, 데모 `demo-key`. **외부에 공개하기 전에 반드시 바꾸세요** → [4. 공개 전 점검표](#4-외부-공개-전-점검표)

---

## 1. 로컬 파이썬으로 실행

모델 6개의 파일(`models/*.onnx`)이 저장소에 들어 있어서 **학습하지 않아도 바로 서빙**됩니다.

```bash
cd Serving/Serving02
python -m venv .venv
# Windows : .venv\Scripts\activate      macOS·Linux·WSL : source .venv/bin/activate
pip install -r requirements.txt -r requirements-ui.txt
```

**터미널 1 — API + 그림판**
```bash
uvicorn app.main:app --port 8000
```
→ 로그에 `모델 로드 : logreg … vit` 6줄과 `모델 로드 완료` 가 보이면 준비 끝
- 그림판 : http://localhost:8000
- API 문서 : http://localhost:8000/docs
- 내 손글씨 채점 : http://localhost:8000/#score (아래 1-2)

**터미널 2 — Streamlit 대시보드 (선택)**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```
→ http://localhost:8501  (API 서버가 꺼져 있으면 대시보드가 API 를 직접 함께 띄웁니다 — 내장 모드)

### 설정 바꾸기
모든 설정은 환경 변수로 바꿉니다. 목록과 설명은 [.env.example](.env.example) 에 있습니다.

```bash
# 예 : TorchScript 형식으로 서빙 (torch 가 설치된 환경)
MODEL_BACKEND=torchscript uvicorn app.main:app --port 8000          # macOS·Linux·WSL
$env:MODEL_BACKEND="torchscript"; uvicorn app.main:app --port 8000  # PowerShell
```

### 1-1. 모델 켜고 끄기 · 투표 모델 바꾸기

| 환경 변수 | 기본값 | 뜻 |
|---|---|---|
| `ENABLED_MODELS` | `logreg,mlp,lenet,cnn,lstm,vit` | 서버가 로드할 모델. 빼면 그 모델은 화면에서도 사라집니다 |
| `ENSEMBLE_MODELS` | `lenet,cnn,lstm` | 최종 답 투표에 참여할 모델. 나머지 켜진 모델은 '참고' 로 결과만 보여 줍니다 |
| `REVIEW_THRESHOLD` | `0.7` | 투표 모델 확률 평균이 이보다 낮으면 '확인 필요' |
| `MIN_AGREEMENT` | `0.66` | 최종 답에 동의한 투표 모델 비율이 이보다 낮으면 '확인 필요' |

```bash
# 예 : 가벼운 서버 — 합성곱 둘만 켜고 둘이 투표
ENABLED_MODELS=lenet,cnn ENSEMBLE_MODELS=lenet,cnn uvicorn app.main:app --port 8000
```

- 요청마다 고를 수도 있습니다 : `{"image": "...", "models": ["cnn", "vit"]}`
  - 이때 투표 모델이 하나도 안 끼어 있으면, 고른 모델 전부가 투표합니다.
- 그림판의 '참여할 모델' 체크박스와 대시보드 사이드바가 이 기능을 씁니다.

### 1-2. 내 손글씨로 모델 채점하기
그림판에는 모드가 둘 있습니다. 둘 다 숫자(0~9)를 그립니다.

| | 한 장씩 보기 | 내 손글씨 채점 |
|---|---|---|
| 무엇을 그리나 | 원하는 숫자 아무거나 | 화면이 정해 준 숫자 |
| 결과 | 그 한 장에서 모델들이 어떻게 판단했는지 자세히 | 기록이 쌓여 모델별 정확도로 채점 |
| 모델 선택 | 끄고 켤 수 있음 | 공정한 비교를 위해 6개 고정 |

1. 그림판 위쪽의 **내 손글씨 채점** 을 누릅니다 (또는 `http://localhost:8000/#score`).
2. 화면에 나온 숫자를 그리고 **기록하고 다음** (Enter) 을 누릅니다. 마음에 안 들게 그렸으면 **지우기**, 넘기려면 **건너뛰기**를 누릅니다.
3. 오른쪽 채점표에 모델별 정확도와 MNIST 정확도의 차이, 자동 처리·정답·걸러 낸 비율이 쌓입니다. **30장 이상** 모으는 것을 권합니다(0~9 각 3장).
4. **결과 내려받기** 로 JSON 을 저장합니다. 기록은 이 브라우저에만 남고 서버에는 저장되지 않습니다.
   - VS Code 안의 브라우저에서는 내려받기가 막힐 수 있습니다. Chrome·Edge 에서 여세요.
   - 기록이 하나도 없으면 아래 세 버튼(취소·내려받기·모두 지우기)은 꺼져 있습니다. 모두 지우기는 실수를 막으려고 두 번 눌러야 합니다.

모델을 다시 학습하거나 바꾼 뒤에는 같은 손글씨로 전후를 비교합니다.
```bash
python scripts/score_my_digits.py my_digits_2026-11-07.json --api http://localhost:8000 --key dev-admin-key
python scripts/score_my_digits.py my_digits_2026-11-07.json --npz my_digits.npz    # 학습·분석용 배열로 저장
```

### 다시 학습하기 (선택)
```bash
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m training.train --epochs 3                 # 6개 모두. MNIST 를 data/ 에 내려받습니다 (약 55MB)
python -m training.train --models vit lstm          # 일부만 다시 학습 (나머지 결과는 그대로)
python -m training.ensemble_eval                     # 앙상블 조합 63가지만 다시 채점
```
- 6개 모두 학습하면 CPU 로 약 6분 걸립니다 (모델당 20초~2분). `models/` 의 18개 파일, `metrics.json`, `samples/` 가 새로 만들어집니다.
- 학습이 끝나면 세 형식(state_dict · TorchScript · ONNX)이 같은 답을 내는지 스스로 검사합니다. 이어서 앙상블 조합을 모두 채점해 `models/metrics.json` 에 남깁니다.

### 모델 추가하기
1. `training/models.py` 에 클래스를 만들고 `MODELS` 에 등록합니다. 입력 `(N,1,28,28)`, 출력 `(N,10)` 이면 됩니다.
2. `app/model_catalog.py` 의 `CATALOG` 에 이름·계열·설명·색을 한 줄 추가합니다.
3. `python -m training.train --models <키>` 를 실행하고 서버를 다시 켭니다. 그림판과 대시보드에 카드·차트가 자동으로 생깁니다.

---

## 2. Docker 로 실행

Docker Desktop, 또는 WSL 안의 Docker Engine 이면 됩니다.

```bash
cd Serving/Serving02                  # WSL 이면 /mnt/c/... 아래의 같은 폴더
cp .env.example .env
docker compose up -d --build
docker compose ps                     # api 가 (healthy) 가 되면 준비 끝
```

| 서비스 | 주소 | 이미지 크기 |
|---|---|---|
| api (FastAPI + 그림판) | http://localhost:8000 | 432MB — torch 없이 onnxruntime 만, 모델 6개 내장 |
| ui (Streamlit) | http://localhost:8501 | 799MB |

- 모델은 **이미지 안에 들어 있습니다**. 처음 뜰 때 아무것도 내려받지 않으므로 인터넷이 없어도 됩니다.
- `ui` 는 `api` 가 `/ready` 를 통과한 다음에 시작합니다 (`depends_on: condition: service_healthy`).
- 로그 보기 : `docker compose logs -f api` · 끄기 : `docker compose down`

---

## 3. 외부에서 접속하기

### 3-1. 같은 네트워크의 다른 기기에서
휴대폰으로 그림판을 써 보고 싶을 때 씁니다.

1. `.env` 에서 `BIND_ADDR=0.0.0.0` 으로 바꾸고 `docker compose up -d` 를 다시 실행합니다.
   - 로컬 파이썬이면 `uvicorn app.main:app --host 0.0.0.0 --port 8000` 로 실행합니다.
2. PC 의 IP 를 확인합니다 (Windows : `ipconfig` 의 IPv4 주소).
3. 휴대폰 브라우저에서 `http://<PC IP>:8000` 으로 접속합니다.

> **Windows + WSL Docker** 는 WSL 이 별도 네트워크라 이것만으로는 안 열릴 수 있습니다.
> - 이때는 Windows 쪽에서 직접 `uvicorn --host 0.0.0.0` 으로 띄우거나, 아래 3-2 의 터널을 쓰는 편이 빠릅니다.
> - Windows 방화벽이 처음 묻는 창에서는 '개인 네트워크 허용'을 고릅니다.

### 3-2. 임시 공개 주소 (Cloudflare Quick Tunnel)
**계정 가입 · 포트 개방 · 공유기 설정 없이** 내 PC 의 서버에 `https://` 주소를 붙입니다. 발표나 지인에게 잠깐 보여 줄 때 가장 간단합니다.

**Docker 로 실행 중이라면**
```bash
docker compose --profile public up -d
docker compose logs tunnel | grep trycloudflare.com
# → https://random-words-1234.trycloudflare.com   ← 이 주소를 공유
```

**로컬 파이썬으로 실행 중이라면**
```bash
# 설치 : Windows  winget install Cloudflare.cloudflared   ·   macOS  brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

- 터널(또는 컨테이너)을 끄면 주소도 사라집니다. 다시 켜면 **새 주소**가 나옵니다.
- 그림판에는 데모 키가 자동으로 채워집니다. 그래서 데모 키에는 꼭 속도 제한을 걸어 둡니다 (기본 분당 120회).
- Streamlit 대시보드까지 공개하려면 터널을 하나 더 엽니다 : `cloudflared tunnel --url http://localhost:8501`

### 3-3. 클라우드 VM (AWS · GCP · RunPod 등)
Docker 가 설치된 리눅스 VM 이라면 어디든 같습니다.

```bash
# 이 폴더만 받아 오기 (저장소 전체를 받지 않음)
git clone --depth 1 --filter=blob:none --sparse https://github.com/stella1298/AIFFEL_Quest_EPA.git
cd AIFFEL_Quest_EPA && git sparse-checkout set Serving/Serving02 && cd Serving/Serving02

cp .env.example .env
python3 scripts/make_api_key.py admin      # 새 키를 만들어 .env 의 API_KEYS 에 붙여 넣기
#   .env 에서 BIND_ADDR=0.0.0.0 으로 변경
docker compose up -d --build
```

- 클라우드 콘솔의 **보안 그룹(방화벽)** 에서 8000 포트(대시보드도 쓰면 8501)를 엽니다.
- 오래 운영할 주소라면 앞에 Nginx·Caddy 를 두고 HTTPS 를 붙이는 것을 권합니다. 게이트웨이 노트의 역할입니다.

### 3-4. Hugging Face Spaces (무료 상시 주소)
Docker 형식 Space 는 이 프로젝트의 `Dockerfile` 을 그대로 빌드합니다. 무료 CPU(2코어·16GB)로 충분합니다.

1. https://huggingface.co/new-space 에서 Space 를 만듭니다. **SDK 는 Docker, 템플릿은 Blank** 를 고릅니다.
2. Space 의 **Settings → Variables and secrets** 에 다음을 넣습니다.
   - Secret `API_KEYS` = `admin:<긴 무작위 키>,demo:<데모 키>` (`python scripts/make_api_key.py` 로 생성)
   - Variable `PUBLIC_DEMO_KEY` = 위의 데모 키
3. 내 PC 에서 올립니다. 필요한 파일만 골라 올리고, 바이너리(.onnx)는 허브가 알아서 처리합니다.
   ```bash
   pip install huggingface_hub
   hf auth login                      # 쓰기(write) 권한 토큰
   python scripts/deploy_hf_space.py <허깅페이스 아이디>/<Space 이름>
   ```
4. Space 페이지의 Logs 에서 빌드가 끝나면 `https://<아이디>-<Space이름>.hf.space` 로 접속합니다.

> - 무료 Space 는 한동안 접속이 없으면 잠들고, 다음 접속 때 다시 깨어나느라 30초쯤 걸립니다.
> - Space 에는 API 서버(그림판 포함)만 올라갑니다. Streamlit 대시보드는 Space 를 하나 더 만들어 따로 올립니다.

### 3-5. Streamlit Community Cloud (xxx.streamlit.app)
`https://<이름>.streamlit.app` 형태의 주소는 Streamlit 이 운영하는 **무료 호스팅**이 주는 주소입니다. GitHub 저장소를 연결하면 Streamlit 이 코드를 받아 실행하고, `<원하는 이름>.streamlit.app` 주소를 붙여 줍니다.

**이 프로젝트에서 바로 되는 이유 — 내장 모드**
- Community Cloud 는 Streamlit 앱 **하나만** 실행합니다. FastAPI 서버를 따로 띄울 수 없습니다.
- 그래서 대시보드는 API 서버를 찾지 못하면 **같은 프로세스 안의 스레드로 FastAPI 를 직접 띄웁니다**. 주소는 `127.0.0.1` 의 빈 포트입니다.
- API 코드는 바꾸지 않았습니다. 화면도 여전히 HTTP 로 API 를 부르므로 인증·속도 제한·검증을 똑같이 지납니다.
- 내장 API 는 컨테이너 밖에서 닿지 않습니다. 그래서 외부에 공개되는 것은 대시보드 화면뿐입니다.
- API 주소를 고르는 순서는 다음과 같습니다.
  1. `API_BASE_URL` 환경 변수
  2. `localhost:8000` 에 떠 있는 서버
  3. 둘 다 없으면 내장 모드
- 필요한 패키지는 [frontend/requirements.txt](frontend/requirements.txt) 에 따로 모았습니다. 화면과 서버 패키지를 합친 목록이고, torch 는 없습니다. Cloud 는 실행 파일과 같은 폴더의 `requirements.txt` 를 먼저 찾습니다.

**배포 순서**
1. https://share.streamlit.io 에 **GitHub 계정으로 로그인**합니다. 저장소 접근 권한을 허용합니다.
2. **Create app → Deploy a public app from GitHub** 를 누르고 다음과 같이 채웁니다.
   | 항목 | 값 |
   |---|---|
   | Repository | `stella1298/AIFFEL_Quest_EPA` |
   | Branch | `master` |
   | Main file path | `Serving/Serving02/frontend/streamlit_app.py` |
   | **App URL** | 원하는 이름. 예 : `mnist-6models` → `https://mnist-6models.streamlit.app` |
3. **Advanced settings** 에서 Python 버전을 **3.12** 로 고릅니다. 위 패키지 버전이 3.12 기준입니다.
4. **Deploy** 를 누릅니다. 패키지를 설치하느라 첫 실행은 2~3분 걸립니다. 그 뒤 사이드바에 `내장 모드` 가 보이면 성공입니다.

> - App URL 은 전 세계에서 하나만 쓸 수 있습니다. 이미 쓰인 이름이면 다른 이름을 고릅니다. 배포 후에도 앱의 **Settings → General** 에서 바꿀 수 있습니다.
> - 저장소에 `git push` 하면 앱이 자동으로 다시 배포됩니다.
> - 며칠 동안 접속이 없으면 앱이 잠듭니다. 다음 방문자가 깨우기 버튼을 누르면 1분쯤 뒤 다시 켜집니다.
> - 그림판(`/`)과 Swagger(`/docs`)는 내장 API 에 있어 외부에서 열 수 없습니다. 그림판까지 공개하려면 3-4 Hugging Face Spaces 를 함께 씁니다.
> - 설정을 바꾸려면 앱의 **Settings → Secrets** 에 TOML 로 적습니다. 예 : `RATE_LIMIT_PER_MIN = "300"`. 최상위 값은 환경 변수로도 전달됩니다.
>   - 방문자 모두가 같은 키를 쓰기 때문에, 사람이 많으면 분당 120회 제한에 걸릴 수 있습니다.

### 3-6. Colab 에서 실행
[colab_quickstart.ipynb](colab_quickstart.ipynb) 를 Colab 에서 열고 위에서부터 실행합니다.
- 이 폴더를 받아 의존성을 설치합니다.
- (선택) GPU 로 다시 학습합니다.
- 서버를 띄우고 Cloudflare 터널로 공개 주소를 만듭니다.

---

## 4. 외부 공개 전 점검표

| 확인 | 이유 |
|---|---|
| ☐ `API_KEYS` 를 `make_api_key.py` 로 만든 새 키로 바꿨다 | `dev-admin-key` 는 이 문서에 적혀 있어 누구나 압니다 |
| ☐ 데모 키(`PUBLIC_DEMO_KEY`)는 관리자 키와 **다른 키**다 | 데모 키는 그림판을 여는 모든 사람의 브라우저에 보입니다 |
| ☐ 데모 키를 공개하기 싫으면 `PUBLIC_DEMO_KEY=` 로 비웠다 | 비우면 그림판에서 키를 직접 입력해야 합니다 |
| ☐ `RATE_LIMIT_PER_MIN` · `RATE_LIMIT_BURST` 를 정했다 | 키마다 분당 요청 수를 제한합니다 (초과하면 429) |
| ☐ 다른 도메인의 화면에서 부를 거라면 `CORS_ORIGINS` 를 그 주소로 좁혔다 | 기본값 `*` 는 어느 사이트에서나 호출할 수 있습니다 |
| ☐ `.env` 가 git 에 올라가지 않는다 | `.gitignore` 에 들어 있습니다 |

---

## 5. 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest                      # 34개 (torch 없으면 28개). 서버를 따로 띄우지 않아도 됩니다 (TestClient)
```

- **API 테스트** (`tests/test_api.py`)
  - 정상(200), 인증(401), 검증(422), 이미지 검사(400·413·415), 속도 제한(429)
  - 앙상블 규칙(확률 평균·투표 모델·확인 필요), 모델 일부 선택
  - 일괄 분류, 스트리밍, 흰 배경 사진 반전, 전처리 회귀
- **내 손글씨 테스트** (`tests/test_my_digits.py`) : 채점 모드 화면, 내려받은 파일 읽기, 서버로 다시 채점
- **형식 테스트** (`tests/test_formats.py`) : 6개 모델 각각 세 형식이 같은 답을 내는지 확인합니다. torch 가 없는 환경에서는 건너뜁니다.

**부하 테스트** (서버가 떠 있어야 합니다)
```bash
# 속도 제한에 걸리지 않게 한도를 크게 준 서버를 따로 띄웁니다
RATE_LIMIT_PER_MIN=1000000 RATE_LIMIT_BURST=1000000 uvicorn app.main:app --port 8000
python scripts/load_test.py --concurrency 16 --requests 3000
```

---

## 6. 문제 해결

| 증상 | 원인 · 대처 |
|---|---|
| 그림판에 `🔑 API 키를 확인해 주세요` | 키가 틀림. `.env` 의 `API_KEYS` 와 화면의 키를 맞춥니다. 브라우저에 예전 키가 저장돼 있으면 입력란을 고쳐 씁니다 |
| `⏳ 요청이 너무 많습니다` (429) | 속도 제한. 응답 헤더 `Retry-After` 초만큼 기다리거나 `.env` 의 한도를 올립니다 |
| `/ready` 가 503 | 모델 파일이 없거나 로드 실패. 로그의 `모델 로드 실패` 아래 원인을 봅니다. `models/*.onnx` 가 있는지 확인합니다 |
| `address already in use` | 같은 포트에 이미 서버가 떠 있습니다. 끄거나 `--port` 를 바꿉니다 (Docker 는 `.env` 의 `API_PORT`) |
| Streamlit 사이드바 🔴 | `API_BASE_URL` 로 준 주소에 서버가 없습니다. 주소를 고치거나, 변수를 지우면 내장 모드로 API 를 직접 띄웁니다 |
| streamlit.app 배포가 `Error installing requirements` | Advanced settings 의 Python 버전이 3.12 인지 봅니다. 바꾸려면 앱을 지우고 다시 만듭니다 |
| `MODEL_BACKEND=pytorch` 에서 `No module named 'training'` | state_dict 는 모델 클래스 코드가 필요합니다. 프로젝트 폴더에서 실행합니다. Docker 이미지에는 ONNX 만 들어 있습니다 |
| 직접 그린 숫자를 자꾸 틀림 | 칸을 꽉 채우게 크게, 선을 끊지 말고 그려 봅니다. 미리보기(28×28)가 MNIST 처럼 보이는지 확인합니다 |
| 화면에 모델이 6개보다 적게 나옴 | `ENABLED_MODELS` 에서 뺐거나 `models/mnist_<키>.onnx` 가 없습니다. 서버 로그의 `모델 파일이 없어 건너뜁니다` 를 확인합니다 |
| 로지스틱 회귀·ViT 가 자주 틀림 | 정상입니다. 같은 조건에서 가장 약한 두 모델이라 '참고' 로만 보여 주고 투표에서는 뺐습니다 |
