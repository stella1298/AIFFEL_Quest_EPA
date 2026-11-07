"""외부에 공개할 때 쓸 무작위 API 키를 만듭니다.

실행 : python scripts/make_api_key.py admin
출력된 'API_KEYS=...' 줄을 .env 에 붙여 넣습니다. (dev-admin-key 같은 기본 키로 외부에 띄우지 마세요)
"""
import secrets
import sys

owner = sys.argv[1] if len(sys.argv) > 1 else "admin"
key = "sk-" + secrets.token_urlsafe(24)
print(f"새 키 ({owner}) : {key}")
print(f"\n.env 예시\nAPI_KEYS={owner}:{key},demo:demo-{secrets.token_urlsafe(8)}")
