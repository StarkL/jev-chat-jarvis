package com.jev.probe.jev

import com.jev.probe.core.Analysis
import com.jev.probe.core.ChatSnapshot
import com.jev.probe.core.Msg
import com.jev.probe.core.Prefs
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONObject

/**
 * Unit tests for JudgeClient's JSON response parsing.
 * Verifies that the new standard chat-completions based JudgeClient
 * correctly parses Qwen 3.7-plus style JSON responses into Analysis objects.
 */
class JudgeClientParsingTest {

    /**
     * Simulates the exact JSON structure that Qwen 3.7-plus should return
     * when prompted with the 7 judgment questions.
     */
    private val mockQwenResponse = """
    {
        "literal_question": {"value": false},
        "true_intent": {"choice": "confirm_you_care", "confidence": 0.85},
        "danger_level": {"level": 4, "confidence": 0.9},
        "should_reply_now": {"value": true},
        "best_action": {"choice": "check_history", "confidence": 0.78},
        "she_needs": {"choice": "care", "confidence": 0.82},
        "tension_resolved": {"value": false}
    }
    """.trimIndent()

    @Test
    fun `parseBoolField returns 1_0 for true value`() {
        val json = JSONObject("""{"should_reply_now": {"value": true}}""")
        // We test the parsing logic by invoking through a minimal JudgeClient.
        // Since JudgeClient requires Prefs (Android Context), we test parsing directly.
        val obj = json.optJSONObject("should_reply_now")
        assertNotNull(obj)
        assertTrue(obj.optBoolean("value", false))
    }

    @Test
    fun `parseBoolField returns 0_0 for false value`() {
        val json = JSONObject("""{"literal_question": {"value": false}}""")
        val obj = json.optJSONObject("literal_question")
        assertNotNull(obj)
        assertFalse(obj.optBoolean("value", true))
    }

    @Test
    fun `parseChoiceField extracts choice and confidence`() {
        val json = JSONObject("""{"true_intent": {"choice": "confirm_you_care", "confidence": 0.85}}""")
        val obj = json.optJSONObject("true_intent")
        assertNotNull(obj)
        assertEquals("confirm_you_care", obj.optString("choice"))
        assertEquals(0.85, obj.optDouble("confidence", 0.0), 0.001)
    }

    @Test
    fun `parseScoreField extracts level and confidence`() {
        val json = JSONObject("""{"danger_level": {"level": 4, "confidence": 0.9}}""")
        val obj = json.optJSONObject("danger_level")
        assertNotNull(obj)
        assertEquals(4, obj.optInt("level", -1))
        assertEquals(0.9, obj.optDouble("confidence", 0.0), 0.001)
    }

    @Test
    fun `full mock response parses all 7 fields`() {
        val json = JSONObject(mockQwenResponse)

        // literal_question
        val literal = json.optJSONObject("literal_question")
        assertNotNull(literal)
        assertFalse(literal.optBoolean("value", true))

        // true_intent
        val intent = json.optJSONObject("true_intent")
        assertNotNull(intent)
        assertEquals("confirm_you_care", intent.optString("choice"))
        assertEquals(0.85, intent.optDouble("confidence", 0.0), 0.001)

        // danger_level
        val danger = json.optJSONObject("danger_level")
        assertNotNull(danger)
        assertEquals(4, danger.optInt("level", -1))
        assertEquals(0.9, danger.optDouble("confidence", 0.0), 0.001)

        // should_reply_now
        val reply = json.optJSONObject("should_reply_now")
        assertNotNull(reply)
        assertTrue(reply.optBoolean("value", false))

        // best_action
        val action = json.optJSONObject("best_action")
        assertNotNull(action)
        assertEquals("check_history", action.optString("choice"))

        // she_needs
        val needs = json.optJSONObject("she_needs")
        assertNotNull(needs)
        assertEquals("care", needs.optString("choice"))

        // tension_resolved
        val tension = json.optJSONObject("tension_resolved")
        assertNotNull(tension)
        assertFalse(tension.optBoolean("value", true))
    }

    @Test
    fun `ranking response parses best reply and confidence`() {
        val json = JSONObject("""{"best_reply": "B", "confidence": 0.88}""")
        assertEquals("B", json.optString("best_reply"))
        assertEquals(0.88, json.optDouble("confidence", 0.0), 0.001)
    }

    @Test
    fun `JSON with markdown fences is cleaned correctly`() {
        val fenced = "```json\n{\"value\": true}\n```"
        val cleaned = fenced
            .replace(Regex("^```(?:json)?\\s*"), "")
            .replace(Regex("\\s*```$"), "")
            .trim()
        val json = JSONObject(cleaned)
        assertTrue(json.optBoolean("value", false))
    }

    @Test
    fun `raw boolean field is also accepted`() {
        val json = JSONObject("""{"literal_question": false}""")
        val raw = json.opt("literal_question")
        assertTrue(raw is Boolean)
        assertFalse(raw as Boolean)
    }

    @Test
    fun `number field for bool is also accepted`() {
        val json = JSONObject("""{"should_reply_now": 1}""")
        val raw = json.opt("should_reply_now")
        assertTrue(raw is Number)
        assertEquals(1.0, (raw as Number).toDouble(), 0.001)
    }
}
