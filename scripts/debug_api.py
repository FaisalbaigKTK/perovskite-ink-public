"""
debug_api.py  —  Run this FIRST to find the exact error and correct model name.
Place this file in the same scripts/ folder and run:
    python debug_api.py
"""
import os, requests, json

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"

if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    raise SystemExit(1)

print(f"API key found: {API_KEY[:15]}...")

MODELS_TO_TRY = [
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

SIMPLE_PAYLOAD = {
    "max_tokens": 20,
    "messages": [{"role": "user", "content": "Say OK"}]
}

HEADERS = {
    "Content-Type":      "application/json",
    "x-api-key":         API_KEY,
    "anthropic-version": "2023-06-01",
}

print("\nTesting models...\n")
working = None
for model in MODELS_TO_TRY:
    payload = {**SIMPLE_PAYLOAD, "model": model}
    try:
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            print(f"  OK  {model}")
            if not working:
                working = model
        else:
            body = resp.json()
            err  = body.get("error", {}).get("message", resp.text[:120])
            print(f"  {resp.status_code} {model}  ->  {err}")
    except Exception as e:
        print(f"  ERR {model}  ->  {e}")

print()
if working:
    print(f"USE THIS MODEL:  {working}")
    print()
    print(f"In step4L_llm_enhance.py change line 44 to:")
    print(f'  MODEL = "{working}"')
else:
    print("No model worked. Check your API key and account balance.")
