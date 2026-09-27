package com.jev.probe.jev

import android.util.Log
import com.jev.probe.core.Analysis
import com.jev.probe.core.ChatSnapshot
import com.jev.probe.core.Choice
import com.jev.probe.core.Prefs
import com.jev.probe.core.RankedReply
import com.jev.probe.core.Score
import com.jev.probe.core.kb.ChatContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Judgment client that works with any OpenAI-compatible `/chat/completions`
 * endpoint (Qwen 3.7-plus, etc.). Sends the 7 judgment questions as a single
 * structured prompt and parses the JSON response.
 *
 * For the legacy TypeSafe `/alpha/decisions` protocol, see the original
 * implementation in git history. This version uses standard chat completions
 * so the same model handles judge, reply and vision.
 */
class JudgeClient(private val prefs: Prefs) {

    /**
     * The 7 judgment questions in one call (~3-5s on Qwen). Errors returned
     * inside [Analysis.error] rather than thrown.
     */
    fun judge(snapshot: ChatSnapshot, relationship: String, ctx: ChatContext? = null): Analysis {
        val start = System.currentTimeMillis()
        return try {
            val state = JevQuestions.buildState(snapshot, relationship)
            val background = ctx?.background(relationship) ?: ""
            val history = ctx?.history ?: emptyList()
            val prompt = buildJudgePrompt(state, relationship, background, history)
            val json = chatCompletion(prompt)
            Analysis(
                trueIntent = parseChoiceField(json, "true_intent"),
                dangerLevel = parseScoreField(json, "danger_level"),
                sheNeeds = parseChoiceField(json, "she_needs"),
                shouldReplyNow = parseBoolField(json, "should_reply_now"),
                bestAction = parseChoiceField(json, "best_action"),
                tensionResolved = parseBoolField(json, "tension_resolved"),
                literalQuestion = parseBoolField(json, "literal_question"),
                rankedReplies = emptyList(),
                latencyMs = System.currentTimeMillis() - start
            )
        } catch (e: Exception) {
            Log.w(TAG, "judge failed: ${e.message}")
            Analysis(null, null, null, null, null, null, null, emptyList(),
                System.currentTimeMillis() - start, error = e.message ?: "判断接口请求失败")
        }
    }

    /**
     * Rank 3 candidate replies by asking the model which is best.
     * Returns them sorted by assigned probability (confidence).
     */
    fun rank(
        snapshot: ChatSnapshot,
        relationship: String,
        candidates: List<String>,
        ctx: ChatContext? = null
    ): List<RankedReply> {
        val state = JevQuestions.buildState(snapshot, relationship)
        val background = ctx?.background(relationship) ?: ""
        val history = ctx?.history ?: emptyList()
        val prompt = buildRankPrompt(state, relationship, candidates, background, history)
        val json = chatCompletion(prompt)
        return parseRanking(json, candidates)
    }

    // --------------------------------------------------------- internals

    /** One standard /chat/completions call; returns the parsed JSON body. */
    private fun chatCompletion(prompt: String): JSONObject {
        val url = prefs.judgeEndpoint()
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", JUDGE_SYSTEM))
            .put(JSONObject().put("role", "user").put("content", prompt))
        val body = JSONObject()
            .put("model", prefs.judgeModel)
            .put("messages", messages)
            .put("temperature", 0.0)
        // response_format=json_object only works on providers that support it
        // (Qwen, OpenRouter). Custom endpoints may reject unknown fields.
        if (isJsonModeProvider(url)) {
            body.put("response_format", JSONObject().put("type", "json_object"))
        }
        val resp = HttpJson.post(url, prefs.judgeKey, body, Route.JUDGE, HttpJson.headersFor(url))
        val content = resp.optJSONArray("choices")?.optJSONObject(0)
            ?.optJSONObject("message")?.optString("content") ?: ""
        // Strip markdown code fences if the model wraps JSON in them.
        val cleaned = content
            .replace(Regex("^```(?:json)?\\s*"), "")
            .replace(Regex("\\s*```$"), "")
            .trim()
        val start = cleaned.indexOf('{')
        val end = cleaned.lastIndexOf('}')
        if (start >= 0 && end > start) {
            return JSONObject(cleaned.substring(start, end + 1))
        }
        throw ApiException(Route.JUDGE, null, "响应不是合法 JSON：${content.take(120)}")
    }

    /** Build the judge prompt: conversation + 7 questions in one block. */
    private fun buildJudgePrompt(
        state: JSONObject,
        relationship: String,
        background: String,
        history: List<com.jev.probe.core.kb.LogEntry>
    ): String {
        val sb = StringBuilder()
        sb.append("## 关系\n$relationship\n\n")
        sb.append("## 最近对话（越靠下越新）\n")
        val chat = state.optJSONObject("chat") ?: state
        val msgs = chat.optJSONArray("messages") ?: JSONArray()
        for (i in 0 until msgs.length()) {
            val m = msgs.getJSONObject(i)
            val side = if (m.optString("from") == "me") "我" else "对方"
            sb.append("$side：${m.optString("text")}\n")
        }
        if (background.isNotBlank()) {
            sb.append("\n## 背景信息\n$background\n")
            sb.append("以上背景为已知事实，回复必须与之保持一致。\n")
        }
        if (history.isNotEmpty()) {
            sb.append("\n## 更早的聊天记录（越靠下越新）\n")
            history.takeLast(prefs.contextHistoryCount.coerceIn(0, 100)).forEach {
                sb.append(if (it.side == "me") "我：" else "对方：").append(it.text).append('\n')
            }
        }
        sb.append("\n## 判断任务\n")
        sb.append("请对以上聊天内容做出以下 7 项判断，严格按 JSON 格式输出，不要加任何解释。\n\n")
        sb.append(JUDGE_QUESTIONS_PROMPT)
        return sb.toString()
    }

    /** Build the ranking prompt: 3 candidates + which is best. */
    private fun buildRankPrompt(
        state: JSONObject,
        relationship: String,
        candidates: List<String>,
        background: String,
        history: List<com.jev.probe.core.kb.LogEntry>
    ): String {
        val sb = StringBuilder()
        sb.append("## 关系\n$relationship\n\n")
        val chat = state.optJSONObject("chat") ?: state
        val msgs = chat.optJSONArray("messages") ?: JSONArray()
        sb.append("## 最近对话\n")
        for (i in 0 until msgs.length()) {
            val m = msgs.getJSONObject(i)
            val side = if (m.optString("from") == "me") "我" else "对方"
            sb.append("$side：${m.optString("text")}\n")
        }
        if (background.isNotBlank()) {
            sb.append("\n## 背景\n$background\n")
        }
        if (history.isNotEmpty()) {
            sb.append("\n## 更早的聊天记录\n")
            history.takeLast(prefs.contextHistoryCount.coerceIn(0, 100)).forEach {
                sb.append(if (it.side == "me") "我：" else "对方：").append(it.text).append('\n')
            }
        }
        sb.append("\n## 候选回复\n")
        val labels = listOf("A", "B", "C")
        candidates.forEachIndexed { i, c -> sb.append("${labels[i]}：$c\n") }
        sb.append("\n请判断哪条候选回复最合适，输出 JSON：\n")
        sb.append("""{"best_reply": "A或B或C", "confidence": 0.0-1.0}""")
        return sb.toString()
    }

    // --------------------------------------------------------- parsing

    /** Parse a choice field: {"choice": "xxx", "confidence": 0.85}. */
    private fun parseChoiceField(json: JSONObject, key: String): Choice? {
        val obj = json.optJSONObject(key) ?: return null
        val choice = obj.optString("choice").ifBlank { null } ?: return null
        val confidence = obj.optDouble("confidence", 0.0)
        val probs = HashMap<String, Double>()
        obj.optJSONObject("probabilities")?.let { p ->
            p.keys().forEach { k -> probs[k] = p.optDouble(k) }
        }
        return Choice(choice, confidence, probs)
    }

    /** Parse a score field: {"level": 3, "confidence": 0.9}. */
    private fun parseScoreField(json: JSONObject, key: String): Score? {
        val obj = json.optJSONObject(key) ?: return null
        val level = obj.optInt("level", -1)
        if (level < 0) return null
        val confidence = obj.optDouble("confidence", 0.0)
        return Score(level.toDouble(), confidence, 9)
    }

    /** Parse a boolean field, return as 0.0 (false) or 1.0 (true) like noul. */
    private fun parseBoolField(json: JSONObject, key: String): Double? {
        val obj = json.optJSONObject(key)
        if (obj != null) {
            return if (obj.optBoolean("value", false)) 1.0 else 0.0
        }
        // Also accept raw boolean
        if (json.has(key)) {
            val raw = json.opt(key)
            if (raw is Boolean) return if (raw) 1.0 else 0.0
            if (raw is Number) return raw.toDouble()
        }
        return null
    }

    /** Parse ranking result: {"best_reply": "A", "confidence": 0.85}. */
    private fun parseRanking(json: JSONObject, candidates: List<String>): List<RankedReply> {
        val best = json.optString("best_reply", "").uppercase().trim()
        val confidence = json.optDouble("confidence", 0.5)
        val labels = listOf("A", "B", "C")
        return candidates.mapIndexed { i, text ->
            val prob = if (labels[i] == best) confidence else (1.0 - confidence) / 2.0
            RankedReply(text, prob.coerceIn(0.0, 1.0))
        }.sortedByDescending { it.prob }
    }

    companion object {
        private const val TAG = "JEVASSIST"

        /**
         * Providers known to support `response_format: json_object`.
         * Custom endpoints may not — omit the field to avoid 400 errors.
         */
        private val JSON_MODE_HOSTS = listOf(
            "dashscope.aliyuncs.com",
            "openrouter.ai",
            "api.deepseek.com",
        )

        fun isJsonModeProvider(url: String): Boolean =
            JSON_MODE_HOSTS.any { url.contains(it, ignoreCase = true) }

        /** System prompt for judgment calls. */
        private const val JUDGE_SYSTEM = "你是一个专业的聊天对话分析助手。根据给定的聊天记录和关系背景，" +
            "严格按照要求的 JSON 格式输出分析结果。不要添加任何额外解释。"

        /** The 7 judgment questions rendered as a prompt block. */
        private val JUDGE_QUESTIONS_PROMPT = """
{"literal_question": {"value": true或false}, "true_intent": {"choice": "选项", "confidence": 0.0-1.0}, "danger_level": {"level": 1-10, "confidence": 0.0-1.0}, "should_reply_now": {"value": true或false}, "best_action": {"choice": "选项", "confidence": 0.0-1.0}, "she_needs": {"choice": "选项", "confidence": 0.0-1.0}, "tension_resolved": {"value": true或false}}

各字段说明：

1. **literal_question** (value: true/false)
   对方最新消息是否完全是字面意思、没有潜台词？true=纯字面，false=有潜台词（试探、讽刺、暗示、反话等）。

2. **true_intent** (choice + confidence)
   对方的真实意图：
   - confirm_you_care: 测试你是否记得、是否在乎
   - vent_anger: 生气/受伤，需要情绪被看见
   - request_action: 要具体行动/承诺/时间
   - seek_explanation: 要事实解释
   - casual_chat: 轻松闲聊无试探
   - close_topic: 和平收尾，话题已关闭

3. **danger_level** (level: 1-10, confidence)
   对话距离吵架/伤害关系的程度：1=轻松闲聊，5=冷淡试探，8=最后通牒，10=关系破裂。

4. **should_reply_now** (value: true/false)
   你的下一条消息是否应包含实质性内容（承认过错、给出具体时间/计划、解释你知道的事实）？
   false=事实不在当前片段中/他们在试探/他们已关闭话题。

5. **best_action** (choice + confidence)
   最佳行动类型：
   - check_history: 先查之前的聊天记录
   - apologize: 真诚道歉
   - give_commitment: 给出具体承诺/deadline
   - explain: 解释发生了什么
   - acknowledge: 表示你在听、在乎
   - say_less: 少说或不说
   - make_plan: 提出/确认具体安排

6. **she_needs** (choice + confidence)
   对方现在需要你做什么：
   - apology: 真诚道歉
   - action: 具体行动/承诺/事实回顾
   - explanation: 清楚解释
   - care: 证明你记得/在乎（忠诚度测试）
   - nothing: 不需要更多（真正接受/和平收尾）

7. **tension_resolved** (value: true/false)
   人际紧张是否已解除？true=从未紧张或已接受/和好，false=仍在试探/生气/最后通牒未撤回。
""".trimIndent()
    }
}
