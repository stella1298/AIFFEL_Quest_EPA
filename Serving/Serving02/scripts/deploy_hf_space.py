"""Hugging Face Spaces(Docker) 로 API 서버를 올립니다.

준비 : pip install huggingface_hub  →  hf auth login (write 토큰)
실행 : python scripts/deploy_hf_space.py <아이디>/<Space 이름>

- Space 가 없으면 Docker SDK 로 새로 만듭니다.
- 서빙에 필요한 파일만 올립니다 (학습 코드·원본 체크포인트·스크린샷 제외).
- Space 의 README 는 deploy/huggingface/README.md 를 씁니다 (app_port 등 Space 설정이 들어 있음).
- API 키는 코드로 올리지 않습니다. Space 의 Settings → Secrets 에 API_KEYS 를 넣으세요.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    sys.exit("사용법 : python scripts/deploy_hf_space.py <아이디>/<Space 이름>")
repo_id = sys.argv[1]

api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    repo_id=repo_id, repo_type="space", folder_path=ROOT,
    allow_patterns=["Dockerfile", "requirements.txt", "app/**", "models/*.onnx", "models/metrics.json"],
    ignore_patterns=["**/__pycache__/**"],
    commit_message="Deploy MNIST serving API",
)
api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=ROOT / "deploy" / "huggingface" / "README.md",
                path_in_repo="README.md", commit_message="Space README")
user, name = repo_id.split("/")
print(f"올렸습니다. 빌드 로그 : https://huggingface.co/spaces/{repo_id}")
print(f"완료 후 주소 : https://{user}-{name}.hf.space".lower().replace("_", "-"))
print("잊지 마세요 : Settings → Variables and secrets 에 API_KEYS(Secret) 와 PUBLIC_DEMO_KEY 를 넣어야 합니다.")
