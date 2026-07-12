import unittest

from src.optimizer import build_optimization


class DynamicTokenBudgetTests(unittest.TestCase):
    def cap(self, category, prompt):
        return build_optimization(prompt, category)["max_tokens"]

    def plan(self, category, prompt):
        return build_optimization(prompt, category)

    def test_sentiment_label_only_is_tiny(self):
        self.assertLessEqual(self.cap("sentiment_analysis", "Classify sentiment. Return the label only."), 10)

    def test_sentiment_reason_is_larger(self):
        label_cap = self.cap("sentiment_analysis", "Classify sentiment. Return the label only.")
        reason_cap = self.cap("sentiment_analysis", "Classify sentiment and provide a brief justification.")
        self.assertGreater(reason_cap, label_cap)

    def test_large_explicit_summary_uses_requested_length_budget(self):
        plan = self.plan("summarization", "Summarize the article in no more than 150 words.\n\nArticle:\n" + "text " * 300)
        self.assertEqual(plan["max_tokens"], 253)
        self.assertLessEqual(plan["max_tokens"], plan["metadata"]["category_max_tokens"])

    def test_long_ner_budget_stops_at_active_category_ceiling(self):
        prompt = "Extract persons and organizations as JSON.\n\nText:\n" + "Alice joined Example Corp in Paris. " * 120
        plan = self.plan("entity_extraction", prompt)
        self.assertLessEqual(plan["max_tokens"], plan["metadata"]["category_max_tokens"])
        self.assertGreater(plan["max_tokens"], 200)

    def test_rich_ner_schema_gets_output_space(self):
        prompt = (
            "Extract entities as JSON objects with fields canonical, aliases, context, and normalized.\n\n"
            "Text:\n" + "Alice joined Example Corp in Paris on 5 May 2024. " * 20
        )
        plan = self.plan("entity_extraction", prompt)
        self.assertLessEqual(plan["max_tokens"], plan["metadata"]["category_max_tokens"])
        self.assertGreater(plan["max_tokens"], 200)

    def test_debug_budget_tracks_code_size(self):
        short = "Fix this Python code.\n```python\ndef f():\n return 1\n```"
        long = "Fix this Python code.\n```python\n" + "def f%d():\n    return %d\n" * 0 + "\n".join(
            f"def f{i}():\n    return {i}" for i in range(100)
        ) + "\n```"
        short_cap = self.cap("bug_fixing", short)
        long_cap = self.cap("bug_fixing", long)
        self.assertGreaterEqual(short_cap, 192)
        self.assertGreater(long_cap, short_cap)
        self.assertLessEqual(long_cap, 1024)

    def test_show_steps_math_has_more_room(self):
        final_cap = self.cap("math_solving", "Calculate 20% of 150. Return only the answer.")
        steps_cap = self.cap("math_solving", "Calculate 20% of 150 and show your work step-by-step.")
        self.assertGreater(steps_cap, final_cap)
        self.assertLessEqual(steps_cap, 384)

    def test_multi_step_percentage_is_not_treated_as_simple(self):
        prompt = "A value starts at 100, rises 20%, falls 10%, adds 35, loses 8, and is taxed 5%. Calculate the result."
        self.assertEqual(self.cap("math_solving", prompt), 160)

    def test_every_category_respects_active_category_ceiling(self):
        generic_prompt = "Explain the task completely with all requested details. " * 200
        categories = (
            "knowledge_qa", "math_solving", "sentiment_analysis", "summarization", "entity_extraction",
            "bug_fixing", "logical_puzzles", "code_authoring", "fallback",
        )
        for category in categories:
            with self.subTest(category=category):
                plan = self.plan(category, generic_prompt)
                self.assertLessEqual(plan["max_tokens"], plan["metadata"]["category_max_tokens"])
