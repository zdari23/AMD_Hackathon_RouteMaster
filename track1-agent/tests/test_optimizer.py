import unittest

from src.optimizer import build_optimization
from src.optimizer.execution import execute_task


class OptimizerIntegrationTests(unittest.TestCase):
    def test_public_optimization_shape(self):
        result = build_optimization("Fix this JavaScript function.")
        self.assertEqual(result["task_type"], "bug_fixing")
        self.assertIn("system_prompt", result)
        self.assertIn("max_tokens", result)
        self.assertEqual(result["metadata"]["constraints"]["programming_language"], "javascript")

    def test_local_sentiment_repair_keeps_single_call(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "The sentiment is Positive.", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}

        result = execute_task("Classify the sentiment as Positive, Negative, or Neutral. Return the label only.", "model", fake_chat)
        self.assertEqual(result["text"], "Positive")
        self.assertEqual(len(calls), 1)

    def test_unrepairable_format_stays_single_call(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "not json", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "finish_reason": "stop"}

        prompt = "Extract persons as JSON.\n\nText:\nAlice arrived."
        result = execute_task(prompt, "model", fake_chat)
        self.assertFalse(result["validation"].valid)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["total_tokens"], 12)

    def test_math_always_uses_api(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "Answer: 30", "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12, "finish_reason": "stop"}

        result = execute_task("Calculate 20% of 150.", "model", fake_chat)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["text"], "Answer: 30")
        self.assertEqual(result["solver_type"], "api")
        self.assertTrue(result["validation"].valid)
