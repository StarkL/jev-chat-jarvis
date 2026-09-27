import json, urllib.request, time, sys

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = "sk-sp-ff335702001748248c70ba75168b1c92"
ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"

# Test 1: Simple prompt, should be fast
payload = {
    'model': 'qwen3.7-plus',
    'messages': [
        {'role': 'system', 'content': '只输出JSON，不要其他文字。'},
        {'role': 'user', 'content': '{"answer": "yes"} 仿照这个格式回答：今天星期几？'}
    ],
    'temperature': 0.0,
    'response_format': {'type': 'json_object'}
}

data = json.dumps(payload).encode()
req = urllib.request.Request(
    ENDPOINT, data=data,
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
)
t0 = time.time()
resp = json.loads(urllib.request.urlopen(req, timeout=40).read())
t1 = time.time()
content = resp['choices'][0]['message']['content']
usage = resp['usage']
print(f'=== 简单测试 ===')
print(f'耗时: {t1-t0:.1f}s')
print(f'input={usage["prompt_tokens"]}, output={usage["completion_tokens"]}')
print(f'内容: {repr(content[:300])}')
print()

# Test 2: Full judge prompt
JUDGE_PROMPT = """{"literal_question": {"value": true或false}, "true_intent": {"choice": "选项", "confidence": 0.0-1.0}, "danger_level": {"level": 1-10, "confidence": 0.0-1.0}, "should_reply_now": {"value": true或false}, "best_action": {"choice": "选项", "confidence": 0.0-1.0}, "she_needs": {"choice": "选项", "confidence": 0.0-1.0}, "tension_resolved": {"value": true或false}}

各字段说明：
1. literal_question (true/false): 对方消息是否纯字面？
2. true_intent: confirm_you_care|vent_anger|request_action|seek_explanation|casual_chat|close_topic
3. danger_level (1-10): 1=轻松, 10=破裂
4. should_reply_now (true/false): 是否应含实质内容？
5. best_action: check_history|apologize|give_commitment|explain|acknowledge|say_less|make_plan
6. she_needs: apology|action|explanation|care|nothing
7. tension_resolved (true/false): 紧张是否解除？

**只输出JSON对象，以{开头以}结尾，不要任何其他文字。**"""

CONVO = """关系：对方是我的伴侣

最近对话：
对方：你最近是不是很忙？
我：还行吧，就是项目有点多
对方：那你上次说周末陪我去看展，还算数吗？
我：算啊，周六下午吧
对方：呵，上次也是这么说的

请判断。"""

payload2 = {
    'model': 'qwen3.7-plus',
    'messages': [
        {'role': 'system', 'content': '你是聊天分析助手。严格按JSON格式输出。'},
        {'role': 'user', 'content': JUDGE_PROMPT + '\n\n' + CONVO}
    ],
    'temperature': 0.0,
    'response_format': {'type': 'json_object'}
}

data2 = json.dumps(payload2).encode()
req2 = urllib.request.Request(
    ENDPOINT, data=data2,
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
)
t0 = time.time()
resp2 = json.loads(urllib.request.urlopen(req2, timeout=40).read())
t1 = time.time()
content2 = resp2['choices'][0]['message']['content']
usage2 = resp2['usage']
print(f'=== Judge 完整测试 ===')
print(f'耗时: {t1-t0:.1f}s')
print(f'input={usage2["prompt_tokens"]}, output={usage2["completion_tokens"]}')
print(f'内容: {repr(content2[:500])}')
