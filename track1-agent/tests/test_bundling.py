import json
import unittest

from src.optimizer import (
    BundleExecutionError,
    assess_bundle_eligibility,
    create_bundles,
    execute_bundle,
)
from src.optimizer.bundling import BUNDLE_BASE_OVERHEAD, BUNDLE_PER_ITEM_OVERHEAD, build_bundle_prompt, parse_bundle_response
from src.optimizer.token_budget import PREVIOUS_VERSION_MAX_TOKENS
from scripts.benchmark_bundling import _dataset_category, _normalize_categories


class BundleBenchmarkCliTests(unittest.TestCase):
    def test_dataset_category_aliases_map_to_router_categories(self):
        self.assertEqual(_normalize_categories("factual_knowledge"), {"knowledge_qa"})
        self.assertEqual(
            _normalize_categories("math, ner, codegen"),
            {"math_solving", "entity_extraction", "code_authoring"},
        )

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_categories("not_a_category")

    def test_dataset_prefix_keeps_factual_fallback_in_factual_experiment(self):
        self.assertEqual(_dataset_category("factual_v6_04", "fallback"), "knowledge_qa")


class BundleEligibilityTests(unittest.TestCase):
    def test_math_is_always_individual(self):
        decision = assess_bundle_eligibility("arbitrary-id", "Calculate 20% of 150. Return only the answer.")
        self.assertFalse(decision.eligible)
        self.assertIn("individual-only", decision.reason)

    def test_task_id_does_not_affect_decision(self):
        prompt = "Calculate 20% of 150. Return only the answer."
        first = assess_bundle_eligibility("first", prompt)
        second = assess_bundle_eligibility("completely-different", prompt)
        self.assertEqual(first.eligible, second.eligible)
        self.assertEqual(first.reason, second.reason)

    def test_math_with_explanation_is_still_individual(self):
        decision = assess_bundle_eligibility("x", "Calculate 20% of 150 and explain every step.")
        self.assertFalse(decision.eligible)

    def test_why_question_is_cautious_knowledge_candidate(self):
        decision = assess_bundle_eligibility("x", "Why does insulation reduce heat transfer?")
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.candidate.task_type, "knowledge_qa")
        self.assertEqual(decision.candidate.safety_tier, "cautious")

    def test_code_is_experimental_candidate(self):
        decision = assess_bundle_eligibility("x", "Write a Python function that adds two numbers.")
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.candidate.task_type, "code_authoring")
        self.assertEqual(decision.candidate.safety_tier, "experimental")

    def test_bug_fixing_is_always_individual(self):
        decision = assess_bundle_eligibility("x", "Fix the bug in this Python function: def f(): return missing")
        self.assertFalse(decision.eligible)
        self.assertIn("individual-only", decision.reason)

    def test_complex_labeled_summary_is_individual(self):
        decision = assess_bundle_eligibility(
            "x",
            "Write a headline followed by exactly two bullet points labeled Gain and Challenge:\n\nUsage improved but costs rose.",
        )
        self.assertFalse(decision.eligible)
        self.assertIn("complex output format", decision.reason)

    def test_long_code_input_is_individual(self):
        code = "\n".join(f"value_{index} = {index}" for index in range(250))
        decision = assess_bundle_eligibility(
            "x",
            f"Write a Python function that wraps this implementation.\n```python\n{code}\n```",
        )
        self.assertFalse(decision.eligible)
        self.assertIn("long code", decision.reason)

    def test_only_easy_high_confidence_logic_is_bundled(self):
        easy = assess_bundle_eligibility(
            "easy",
            "Alice and Bob each choose a different color.\n- Alice is not blue.\n- Bob is blue.\nWho has each color?",
        )
        low_confidence = assess_bundle_eligibility(
            "low",
            "Four reports are ordered.\n- X is immediately before Y.\n- Z is first.\nWhat is the order?",
        )
        self.assertTrue(easy.eligible)
        self.assertEqual(easy.candidate.task_type, "logical_puzzles")
        self.assertFalse(low_confidence.eligible)

    def test_numeric_payload_keeps_summary_category(self):
        decision = assess_bundle_eligibility(
            "x",
            "Summarize in no more than 20 words:\n\nUsage rose 25% over 3 months and delays fell by 10 minutes.",
        )
        self.assertEqual(decision.candidate.task_type, "summarization")
        self.assertEqual(decision.candidate.safety_tier, "cautious")


class BundleExecutionTests(unittest.TestCase):
    def make_knowledge_bundle(self, count=3):
        candidates = []
        for index in range(count):
            decision = assess_bundle_eligibility(
                f"task-{index}",
                f"What is concept number {index + 1}?",
            )
            self.assertTrue(decision.eligible)
            candidates.append(decision.candidate)
        bundles, singles = create_bundles(candidates, max_bundle_size=5, allowed_tiers=("cautious",))
        self.assertFalse(singles)
        self.assertEqual(len(bundles), 1)
        return bundles[0]

    def test_bundle_cap_sums_individual_caps_plus_overhead(self):
        bundle = self.make_knowledge_bundle(5)
        individual_total = sum(item.optimization["max_tokens"] for item in bundle.items)
        self.assertEqual(
            bundle.max_tokens,
            individual_total + BUNDLE_BASE_OVERHEAD + BUNDLE_PER_ITEM_OVERHEAD["knowledge_qa"] * len(bundle.items),
        )
        self.assertGreater(bundle.max_tokens, PREVIOUS_VERSION_MAX_TOKENS["knowledge_qa"])

    def test_bundle_prompt_uses_category_policy_and_exact_json_contract(self):
        bundle = self.make_knowledge_bundle(2)
        system, user = build_bundle_prompt(bundle)
        self.assertIn("Answer every factual question accurately", system)
        self.assertIn("exactly one valid JSON object", system)
        self.assertIn("when a task explicitly requests JSON", system)
        self.assertEqual(len(json.loads(user.split("\n", 1)[1])), 2)

    def test_execute_bundle_uses_one_call_and_validates_outputs(self):
        bundle = self.make_knowledge_bundle(3)
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {
                "text": json.dumps({"0": "First", "1": "Second", "2": "Third"}),
                "prompt_tokens": 40,
                "completion_tokens": 12,
                "total_tokens": 52,
                "finish_reason": "stop",
            }

        result = execute_bundle(bundle, "model", fake_chat)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["total_tokens"], 52)
        self.assertTrue(all(item["validation"].valid for item in result["outputs"]))

    def test_malformed_response_fails_after_one_call(self):
        bundle = self.make_knowledge_bundle(2)
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "not json", "total_tokens": 5, "finish_reason": "stop"}

        with self.assertRaises(BundleExecutionError) as caught:
            execute_bundle(bundle, "model", fake_chat)
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.response["total_tokens"], 5)
        self.assertEqual(caught.exception.response["text"], "not json")

    def test_parser_requires_exact_keys(self):
        with self.assertRaises(ValueError):
            parse_bundle_response('{"0":"a","extra":"b"}', 1)
