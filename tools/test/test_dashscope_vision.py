"""
Generate a valid 16x16 JPEG for vision testing and run the test.
"""
import base64
import json
import os
import sys
import io
import urllib.request
import urllib.error

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_BASE = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    print("ERROR: Set DASHSCOPE_API_KEY env var first")
    sys.exit(1)
MODEL = "qwen3.7-plus"
ENDPOINT = f"{API_BASE}/chat/completions"


def make_tiny_jpeg_b64():
    """Create a 16x16 solid-color JPEG and return base64."""
    try:
        from PIL import Image
        img = Image.new('RGB', (16, 16), color='blue')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Minimal valid 16x16 JPEG (solid blue) — hand-crafted bytes
        # This is a valid JPEG that Pillow or any JPEG decoder accepts.
        # 16x16 pixels, all same color.
        import struct
        # Use a known-good base64 of a tiny JPEG
        return (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCA"
            "AQABABAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAA"
            "AAAAAAD/2gAIAQEAAD8AVN//2Q=="
        )


def test_vision():
    print("Testing VISION route (image_url with valid 16x16 JPEG)")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model: {MODEL}")
    print("-" * 60)

    b64 = make_tiny_jpeg_b64()
    print(f"Image base64 length: {len(b64)} chars")

    prompt = "请描述你在这张图片中看到了什么颜色。如果图片太小无法识别细节，请描述大致颜色。"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err_body[:500]}")
        return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Response: {content}")

    usage = result.get("usage", {})
    print(f"Tokens: input={usage.get('prompt_tokens', '?')}, output={usage.get('completion_tokens', '?')}")

    if content:
        print("✅ Vision route works with image_url!")
        return True
    else:
        print("❌ Vision route returned empty content")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen 3.7-plus Vision Route Test (valid image)")
    print("=" * 60)
    test_vision()
