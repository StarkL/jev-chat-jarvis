"""
Standalone test for the Qwen 3.7-plus Judge route via DashScope.
Simulates what JudgeClient.kt does: build prompt → POST /chat/completions → parse JSON.

Usage: python test_dashscope_judge.py
Requires: DASHSCOPE_API_KEY env var or set below.
"""
import json
import os
import sys
import urllib.request
import urllib.error

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config (from CC Switch / settings.json) ──
API_BASE = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    print("ERROR: Set DASHSCOPE_API_KEY env var first")
    sys.exit(1)
MODEL = "qwen3.7-plus"
ENDPOINT = f"{API_BASE}/chat/completions"

# ── Simulated chat snapshot ──
RELATIONSHIP = "对方是我的伴侣；from=me 的是我发的，from=other 的是对方发的"
MESSAGES = [
    {"from": "other", "text": "你最近是不是很忙？"},
    {"from": "me", "text": "还行吧，就是项目有点多"},
    {"from": "other", "text": "那你上次说周末陪我去看展，还算数吗？"},
    {"from": "me", "text": "算啊，周六下午吧"},
    {"from": "other", "text": "呵，上次也是这么说的"},
]

# ── System prompt (same as JudgeClient.JUDGE_SYSTEM) ──
JUDGE_SYSTEM = (
    "你是一个专业的聊天对话分析助手。根据给定的聊天记录和关系背景，"
    "严格按照要求的 JSON 格式输出分析结果。不要添加任何额外解释。"
)

# ── Judge prompt (same structure as JudgeClient.JUDGE_QUESTIONS_PROMPT) ─
def build_judge_prompt(messages, relationship):
    lines = [f"## 关系\n{relationship}\n\n", "## 最近对话（越靠下越新）"]
    for m in messages:
        side = "我" if m["from"] == "me" else "对方"
        lines.append(f"{side}：{m['text']}")

    lines.append("\n## 判断任务")
    lines.append(
        "请对以上聊天内容做出以下 7 项判断，严格按 JSON 格式输出，不要加任何解释。\n"
    )
    lines.append(
        '{"literal_question": {"value": true或false}, '
        '"true_intent": {"choice": "选项", "confidence": 0.0-1.0}, '
        '"danger_level": {"level": 1-10, "confidence": 0.0-1.0}, '
        '"should_reply_now": {"value": true或false}, '
        '"best_action": {"choice": "选项", "confidence": 0.0-1.0}, '
        '"she_needs": {"choice": "选项", "confidence": 0.0-1.0}, '
        '"tension_resolved": {"value": true或false}}\n\n'
        "各字段说明：\n"
        "1. **literal_question** (value: true/false)\n"
        "   对方最新消息是否完全是字面意思、没有潜台词？true=纯字面，false=有潜台词。\n"
        "2. **true_intent** (choice + confidence)\n"
        "   选项: confirm_you_care / vent_anger / request_action / seek_explanation / casual_chat / close_topic\n"
        "3. **danger_level** (level: 1-10, confidence)\n"
        "   1=轻松闲聊，5=冷淡试探，8=最后通牒，10=关系破裂。\n"
        "4. **should_reply_now** (value: true/false)\n"
        "   你的下一条消息是否应包含实质性内容？\n"
        "5. **best_action** (choice + confidence)\n"
        "   选项: check_history / apologize / give_commitment / explain / acknowledge / say_less / make_plan\n"
        "6. **she_needs** (choice + confidence)\n"
        "   选项: apology / action / explanation / care / nothing\n"
        "7. **tension_resolved** (value: true/false)\n"
        "   人际紧张是否已解除？"
    )
    return "\n".join(lines)


def call_qwen_judge():
    """POST to DashScope /chat/completions and parse the response."""
    prompt = build_judge_prompt(MESSAGES, RELATIONSHIP)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
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

    print(f"POST {ENDPOINT}")
    print(f"Model: {MODEL}")
    print(f"Prompt length: {len(prompt)} chars")
    print("-" * 60)

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {err_body[:500]}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    # Extract assistant message
    choices = result.get("choices", [])
    if not choices:
        print("No choices in response")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:500])
        return None

    content = choices[0]["message"]["content"]
    print("Raw response:")
    print(content)
    print("-" * 60)

    # Parse JSON (strip markdown fences if present)
    cleaned = content.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
    else:
        print("WARNING: No JSON object found in response")
        return None

    # Validate all 7 fields
    print("\n✅ Parsed judgment result:")
    fields = {
        "literal_question": lambda v: f"value={v.get('value')}",
        "true_intent": lambda v: f"choice={v.get('choice')} conf={v.get('confidence')}",
        "danger_level": lambda v: f"level={v.get('level')} conf={v.get('confidence')}",
        "should_reply_now": lambda v: f"value={v.get('value')}",
        "best_action": lambda v: f"choice={v.get('choice')} conf={v.get('confidence')}",
        "she_needs": lambda v: f"choice={v.get('choice')} conf={v.get('confidence')}",
        "tension_resolved": lambda v: f"value={v.get('value')}",
    }

    all_ok = True
    for field, fmt in fields.items():
        val = parsed.get(field)
        if val is None:
            print(f"  ❌ {field}: MISSING")
            all_ok = False
        else:
            print(f"  ✅ {field}: {fmt(val)}")

    # Usage stats
    usage = result.get("usage", {})
    print(f"\n📊 Tokens: input={usage.get('prompt_tokens', '?')}, output={usage.get('completion_tokens', '?')}")

    return parsed if all_ok else None


def call_qwen_reply():
    """Test the Reply route (standard chat completions)."""
    print("\n" + "=" * 60)
    print("Testing REPLY route (candidate generation)")
    print("=" * 60)

    sys_prompt = (
        "你是中文即时通讯回复助手。只输出一个 JSON 数组，含且仅含 3 条候选回复文本，"
        "三条策略要有区别（例如：一条稳妥承接、一条给具体行动或承诺、一条简短低姿态）。"
        "每条不超过 40 字，口语、自然、像真人在聊天软件里发消息。不要解释，直接输出 JSON 数组。"
    )

    convo = "\n".join(
        f"{'我' if m['from'] == 'me' else '对方'}：{m['text']}" for m in MESSAGES
    )
    user_prompt = f"关系：{RELATIONSHIP}\n\n最近对话：\n{convo}\n\n请给出 3 条候选回复。"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
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
    except Exception as e:
        print(f"Reply request failed: {e}")
        return None

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("Raw response:")
    print(content)

    # Parse JSON array
    start = content.find("[")
    end = content.rfind("]")
    if start >= 0 and end > start:
        replies = json.loads(content[start : end + 1])
        print(f"\n✅ Parsed {len(replies)} candidate replies:")
        for i, r in enumerate(replies):
            print(f"  {i+1}. {r}")
        return replies
    else:
        print("WARNING: No JSON array found")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen 3.7-plus Judge Route Test")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model: {MODEL}")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print("=" * 60)

    judge_result = call_qwen_judge()
    reply_result = call_qwen_reply()

    print("\n" + "=" * 60)
    if judge_result and reply_result:
        print("🎉 ALL ROUTES PASSED — Judge + Reply both working!")
    elif judge_result:
        print("⚠️  Judge passed, Reply failed")
    elif reply_result:
        print("⚠️  Reply passed, Judge failed")
    else:
        print("❌ Both routes failed")
    print("=" * 60)
