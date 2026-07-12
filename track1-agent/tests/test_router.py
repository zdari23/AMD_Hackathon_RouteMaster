import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.output_optimizer import detect_task_type, detect_task_type_detailed
from router.infer_router import predict_with_backend

# Map dataset prefixes/categories to the internal categories used by the router
CATEGORY_MAP = {
    "factual": "knowledge_qa",
    "math": "math_solving",
    "sentiment": "sentiment_analysis",
    "summary": "summarization",
    "ner": "entity_extraction",
    "debug": "bug_fixing",
    "logic": "logical_puzzles",
    "codegen": "code_authoring"
}


class WeightedRouterMinimalPairTests(unittest.TestCase):
    CASES = {
        "Explain what NER is.": "knowledge_qa",
        "Extract named entities from this text.": "entity_extraction",
        "Explain how sentiment analysis works.": "knowledge_qa",
        "Classify the sentiment of this review.": "sentiment_analysis",
        "Write a Python function for sentiment analysis.": "code_authoring",
        "Explain what a logic puzzle is.": "knowledge_qa",
        "Solve this logic puzzle.": "logical_puzzles",
        "Write a function that solves a logic puzzle.": "code_authoring",
        "Explain why this code fails.": "bug_fixing",
        "Fix this code.": "bug_fixing",
        "Write a function that summarizes text.": "code_authoring",
        "Summarize this article about a 20% increase.": "summarization",
        "Calculate the 20% increase described in this article.": "math_solving",
        "Explain what 20% inflation means.": "knowledge_qa",
        "Return the entities as JSON.": "entity_extraction",
        "Summarize in exactly 3 bullets.": "summarization",
        "Summarize in no more than 120 words.": "summarization",
        "Fix this JavaScript function.": "bug_fixing",
        "Implement this C++ method.": "code_authoring",
    }

    def test_minimal_pairs(self):
        for prompt, expected in self.CASES.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_task_type(prompt), expected)

    def test_detailed_decision_is_observable(self):
        decision = detect_task_type_detailed("Extract named entities from this text.")
        self.assertEqual(decision.task_type, "entity_extraction")
        self.assertGreater(decision.confidence, 0.5)
        self.assertIn("entity_extraction", decision.matched_signals)

    def test_payload_does_not_override_instruction(self):
        prompt = "Summarize the source in one sentence.\n\nText:\nThe tool can classify sentiment and extract named entities."
        self.assertEqual(detect_task_type(prompt), "summarization")

    def test_numeric_payload_does_not_override_summary_action(self):
        prompt = "Summarize in one sentence.\n\nText:\nUsage rose 20% over 3 months and wait time fell by 5 minutes."
        self.assertEqual(detect_task_type(prompt), "summarization")

    def test_quantity_question_can_follow_numeric_context(self):
        prompt = "A store starts with 500 items, sells 20%, and ships 30 more. How many remain?"
        self.assertEqual(detect_task_type(prompt), "math_solving")

    def test_weighted_and_growth_questions_route_to_math(self):
        prompts = (
            "Scores worth 25%, 35%, and 40% are 88, 76, and 92. What is the weighted final grade?",
            "A population of 8,000 increases by 15% and then decreases by 10%. What is the final population?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(detect_task_type(prompt), "math_solving")

    def test_stacked_constraint_structure_routes_to_logic(self):
        prompt = (
            "Five packages are stacked from bottom to top.\n"
            "- C is in the middle.\n- A is immediately below B.\n- E is on top.\n- D is immediately above C.\n"
            "What is the order from bottom to top?"
        )
        self.assertEqual(detect_task_type(prompt), "logical_puzzles")

    def test_demo_router_works_without_optional_checkpoint(self):
        label, backend = predict_with_backend("Fix this Python function.")
        self.assertIn(label, {"easy", "hard"})
        self.assertIn(backend, {"fine-tuned DistilBERT", "deterministic local fallback"})

def main():
    parser = argparse.ArgumentParser(description="Test router accuracy on a dataset.")
    parser.add_argument("--dataset", type=str, default="data/track1_balanced_40_tasks.json", help="Path to the dataset JSON")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} not found.")
        return
        
    tasks = json.loads(dataset_path.read_text())
    
    total = 0
    correct = 0
    
    # Track performance per category
    category_stats = {}
    
    for task in tasks:
        prompt = task["prompt"]
        
        # Determine the true category of the task
        true_category = task.get("category")
        if not true_category:
            task_id = task.get("task_id", task.get("id", ""))
            if "_" in task_id:
                true_category = task_id.split("_")[0]
            else:
                true_category = "unknown"
                
        expected_router_category = CATEGORY_MAP.get(true_category, true_category)
        
        # Run the router
        detected = detect_task_type(prompt)
        
        if expected_router_category not in category_stats:
            category_stats[expected_router_category] = {"total": 0, "correct": 0}
            
        category_stats[expected_router_category]["total"] += 1
        total += 1
        
        if detected == expected_router_category:
            correct += 1
            category_stats[expected_router_category]["correct"] += 1
        else:
            print(f"❌ Mismatch in {task.get('task_id', 'unknown')}:")
            print(f"   Expected: {expected_router_category}")
            print(f"   Detected: {detected}")
            print(f"   Prompt snippet: {prompt[:100]}...\n")

    print(f"\n--- Router Accuracy ---")
    print(f"Dataset: {args.dataset}")
    print(f"Overall Accuracy: {correct}/{total} ({(correct/total*100) if total else 0:.1f}%)")
    
    print("\nBreakdown by expected category:")
    for cat, stats in sorted(category_stats.items()):
        cat_correct = stats["correct"]
        cat_total = stats["total"]
        acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        print(f"  • {cat:<20} | {cat_correct}/{cat_total} ({acc:.1f}%)")

if __name__ == "__main__":
    main()
