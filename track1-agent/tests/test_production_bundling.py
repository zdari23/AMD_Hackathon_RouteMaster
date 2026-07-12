import json
import unittest

from src.optimizer.execution import solve_tasks
from scripts.evaluate_production import evaluate, evaluate_ab


class ProductionBundlingTests(unittest.TestCase):
    def test_policy_bundles_safe_categories_and_individualizes_math_and_bugs(self):
        prompts_and_answers = [
            ("What is photosynthesis?", "Plants convert light into chemical energy."),
            ("Calculate 2 + 3. Return only the answer.", "5"),
            ("Classify sentiment as Positive, Negative, or Neutral. Return only the label. Text: Great work!", "Positive"),
            ("Summarize in one sentence: Cats sleep for much of the day.", "Cats sleep for much of the day."),
            (
                'Extract all named entities as JSON with text and type fields. Text: Alice visited Paris.',
                [{"text": "Alice", "type": "PERSON"}, {"text": "Paris", "type": "LOCATION"}],
            ),
            (
                "Fix the bug in this Python function. Return only corrected code.\n```python\ndef broken():\n    return 1\n```",
                "def broken():\n    return 1",
            ),
            (
                "Alice and Bob each choose a different color.\n- Alice is not blue.\n- Bob is blue.\nWho has each color?",
                "Alice: Red, Bob: Blue",
            ),
            (
                "Write a Python function `identity(value)` that returns value. Return only code.",
                "def identity(value):\n    return value",
            ),
        ]
        tasks = []
        answer_by_prompt = {}
        for category_index, (prompt, answer) in enumerate(prompts_and_answers):
            for copy in range(2):
                task_id = f"category-{category_index}-{copy}"
                tasks.append({"task_id": task_id, "prompt": prompt})
                answer_by_prompt[prompt] = answer

        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            if kwargs["prompt"].startswith("Independent tasks:"):
                payload = json.loads(kwargs["prompt"].split("\n", 1)[1])
                response = {item["key"]: answer_by_prompt[item["task"]] for item in payload}
                text = json.dumps(response)
            else:
                text = answer_by_prompt[kwargs["prompt"]]
            return {
                "text": text,
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120,
                "finish_reason": "stop",
            }

        solved = solve_tasks(tasks, "model", fake_chat)

        self.assertEqual(len(calls), 10)
        self.assertEqual(solved["bundle_calls"], 6)
        self.assertEqual(solved["individual_calls"], 4)
        self.assertEqual(solved["bundle_fallbacks"], 0)
        self.assertEqual([item["task_id"] for item in solved["results"]], [task["task_id"] for task in tasks])
        solver_types = [item["solver_type"] for item in solved["results"]]
        self.assertEqual(solver_types.count("api_bundled"), 12)
        self.assertEqual(solver_types.count("api"), 4)
        individual_reasons = {
            record["reason"] for record in solved["api_call_records"] if record["call_type"] == "individual"
        }
        self.assertIn("math_solving is individual-only", individual_reasons)
        self.assertIn("bug_fixing is individual-only", individual_reasons)

    def test_category_sizes_and_caps_scale_with_item_count(self):
        tasks = [
            {
                "task_id": f"sentiment-{index}",
                "prompt": "Classify sentiment as Positive, Negative, or Neutral. Return only the label. Text: Great!",
            }
            for index in range(9)
        ]
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["prompt"].split("\n", 1)[1])
            return {
                "text": json.dumps({item["key"]: "Positive" for item in payload}),
                "total_tokens": 20,
                "finish_reason": "stop",
            }

        solved = solve_tasks(tasks, "model", fake_chat)

        # Sentiment's category size is eight; 9 is balanced as 7+2 rather than 8+1.
        self.assertEqual([len(json.loads(call["prompt"].split("\n", 1)[1])) for call in calls], [7, 2])
        self.assertEqual(solved["bundle_calls"], 2)
        self.assertEqual(solved["individual_calls"], 0)
        bundle_call = calls[0]
        # The dynamic label-only cap is 10, not the category hard ceiling of 40.
        self.assertEqual(bundle_call["max_tokens"], 24 + 7 * (10 + 12))
        self.assertLessEqual(bundle_call["max_tokens"], 4096)

    def test_malformed_bundle_falls_back_without_losing_order(self):
        tasks = [
            {"task_id": "first", "prompt": "What is HTTP?"},
            {"task_id": "second", "prompt": "What is TLS?"},
        ]
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            if kwargs["prompt"].startswith("Independent tasks:"):
                return {"text": "not json", "total_tokens": 7, "finish_reason": "stop"}
            return {"text": "A factual answer.", "total_tokens": 3, "finish_reason": "stop"}

        solved = solve_tasks(tasks, "model", fake_chat)

        self.assertEqual(len(calls), 3)
        self.assertEqual(solved["bundle_calls"], 1)
        self.assertEqual(solved["individual_calls"], 2)
        self.assertEqual(solved["bundle_fallbacks"], 2)
        self.assertEqual(solved["total_tokens"], 13)
        self.assertEqual([item["task_id"] for item in solved["results"]], ["first", "second"])

    def test_router_fallback_tasks_are_always_sent_individually(self):
        tasks = [
            {"task_id": "fallback-1", "prompt": "Tell me something useful."},
            {"task_id": "fallback-2", "prompt": "Please respond helpfully."},
        ]
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "A direct response.", "total_tokens": 4, "finish_reason": "stop"}

        solved = solve_tasks(tasks, "model", fake_chat)

        self.assertEqual(solved["bundle_calls"], 0)
        self.assertEqual(solved["individual_calls"], 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not call["prompt"].startswith("Independent tasks:") for call in calls))
        self.assertEqual([item["solver_type"] for item in solved["results"]], ["api", "api"])

    def test_production_evaluation_reports_router_accuracy_and_category_tokens(self):
        tasks = [
            {"task_id": "factual_v7_01", "prompt": "What is HTTP?"},
            {"task_id": "factual_v7_02", "prompt": "What is TLS?"},
            {"task_id": "misc_v7_01", "prompt": "Please respond helpfully."},
        ]

        def fake_chat(**kwargs):
            if kwargs["prompt"].startswith("Independent tasks:"):
                payload = json.loads(kwargs["prompt"].split("\n", 1)[1])
                return {
                    "text": json.dumps({item["key"]: "A factual answer." for item in payload}),
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                    "finish_reason": "stop",
                }
            return {
                "text": "A fallback answer.",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "finish_reason": "stop",
            }

        def fake_judge(prompt, answer, model, chat_fn):
            return {"verdict": "correct", "reason": "CORRECT test verdict", "tokens": 2}

        report = evaluate(tasks, "model", "judge", fake_chat, fake_judge)

        self.assertEqual(report["router_distribution"], {"knowledge_qa": 2, "fallback": 1})
        self.assertEqual(report["fallback_task_ids"], ["misc_v7_01"])
        self.assertEqual(report["production"]["bundle_calls"], 1)
        self.assertEqual(report["production"]["individual_calls"], 1)
        self.assertEqual(report["production"]["total_tokens"], 35)
        self.assertEqual(report["category_metrics"]["knowledge_qa"]["total_tokens"], 30)
        self.assertEqual(report["category_metrics"]["fallback"]["total_tokens"], 5)
        self.assertEqual(report["category_metrics"]["knowledge_qa"]["accuracy_pct"], 100.0)
        self.assertEqual(report["overall_accuracy_pct"], 100.0)
        self.assertEqual(report["judge_tokens"], 6)

    def test_ab_evaluation_compares_individual_and_production_by_category(self):
        tasks = [
            {"task_id": "factual_v7_01", "prompt": "What is HTTP?"},
            {"task_id": "factual_v7_02", "prompt": "What is TLS?"},
        ]

        def fake_chat(**kwargs):
            if kwargs["prompt"].startswith("Independent tasks:"):
                payload = json.loads(kwargs["prompt"].split("\n", 1)[1])
                return {
                    "text": json.dumps({item["key"]: "A factual answer." for item in payload}),
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "finish_reason": "stop",
                }
            return {
                "text": "A factual answer.",
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
                "finish_reason": "stop",
            }

        def fake_judge(prompt, answer, model, chat_fn):
            return {"verdict": "correct", "reason": "CORRECT test verdict", "tokens": 2}

        report = evaluate_ab(tasks, "model", "judge", fake_chat, fake_judge)
        row = report["ab_test"]["category_comparison"]["knowledge_qa"]

        self.assertEqual(row["individual_tokens"], 24)
        self.assertEqual(row["production_tokens"], 20)
        self.assertEqual(row["token_savings_pct"], 16.67)
        self.assertEqual(row["individual_accuracy_pct"], 100.0)
        self.assertEqual(row["production_accuracy_pct"], 100.0)
        self.assertEqual(row["accuracy_delta_pp"], 0.0)
        self.assertEqual(report["ab_test"]["overall"]["individual_calls"], 2)


if __name__ == "__main__":
    unittest.main()
