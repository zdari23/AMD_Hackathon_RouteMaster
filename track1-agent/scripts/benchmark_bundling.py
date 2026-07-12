"""A/B benchmark individual versus prompt-structure-eligible bundled calls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimizer import (  # noqa: E402
    BundleCandidate,
    BundleExecutionError,
    assess_bundle_eligibility,
    build_optimization,
    create_bundles,
    execute_bundle,
    execute_task,
)


DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/glm-5p2"

CATEGORY_ALIASES = {
    "factual": "knowledge_qa",
    "factual_knowledge": "knowledge_qa",
    "knowledge": "knowledge_qa",
    "knowledge_qa": "knowledge_qa",
    "math": "math_solving",
    "math_solving": "math_solving",
    "sentiment": "sentiment_analysis",
    "sentiment_analysis": "sentiment_analysis",
    "summary": "summarization",
    "summarization": "summarization",
    "ner": "entity_extraction",
    "entity_extraction": "entity_extraction",
    "debug": "bug_fixing",
    "bug_fixing": "bug_fixing",
    "code_debugging": "bug_fixing",
    "logic": "logical_puzzles",
    "logical_puzzles": "logical_puzzles",
    "codegen": "code_authoring",
    "code_authoring": "code_authoring",
}

DATASET_PREFIX_CATEGORIES = {
    "factual": "knowledge_qa",
    "math": "math_solving",
    "sentiment": "sentiment_analysis",
    "summary": "summarization",
    "ner": "entity_extraction",
    "debug": "bug_fixing",
    "logic": "logical_puzzles",
    "codegen": "code_authoring",
}


def _normalize_categories(raw: str) -> set[str]:
    categories: set[str] = set()
    for value in raw.split(","):
        name = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not name:
            continue
        if name not in CATEGORY_ALIASES:
            choices = ", ".join(sorted(CATEGORY_ALIASES))
            raise ValueError(f"unknown category {value!r}; choose one of: {choices}")
        categories.add(CATEGORY_ALIASES[name])
    return categories


def _dataset_category(task_id: str, detected_task_type: str) -> str:
    """Use the benchmark's explicit ID prefix for selection, never for production routing."""
    prefix = task_id.split("_", 1)[0].lower()
    return DATASET_PREFIX_CATEGORIES.get(prefix, detected_task_type)


def _verdict_label(judge: dict[str, Any]) -> str:
    verdict = str(judge.get("verdict", "unknown")).upper()
    if verdict == "CORRECT":
        return "PASS"
    if verdict == "INCORRECT":
        return "FAIL"
    return verdict


def _print_full_results(
    task_records: list[dict[str, Any]],
    bundle_records: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 88)
    print("INDIVIDUAL RESULTS (production tokens exclude judge tokens)")
    print("=" * 88)
    for record in task_records:
        result = record["individual"]
        judge = result["judge"]
        print(f"\n[{_verdict_label(judge)}] {record['task_id']} ({record['task_type']})")
        print(f"PROMPT:\n{record['prompt']}")
        if result["ok"]:
            execution = result["result"]
            print(f"ANSWER:\n{execution['text']}")
            print(
                "TOKENS: "
                f"prompt={execution['prompt_tokens']} "
                f"completion={execution['completion_tokens']} "
                f"total={execution['total_tokens']}"
            )
            print(f"LOCAL VALIDATION: {'PASS' if execution['validation'].valid else 'FAIL'}")
        else:
            print(f"EXECUTION ERROR: {result['error']}")
        print(f"JUDGE: {judge['reason']} (judge_tokens={judge['tokens']})")

    print("\n" + "=" * 88)
    print("BUNDLE CALLS (one token total per API call)")
    print("=" * 88)
    for bundle in bundle_records:
        print(
            f"\nBUNDLE #{bundle['bundle_index'] + 1} category={bundle['task_type']} "
            f"tasks={bundle['task_ids']}"
        )
        print(
            "TOKENS: "
            f"prompt={bundle['prompt_tokens']} "
            f"completion={bundle['completion_tokens']} "
            f"total={bundle['total_tokens']}"
        )
        if bundle["error"]:
            print(f"EXECUTION ERROR: {bundle['error']}")

    print("\n" + "=" * 88)
    print("BUNDLED TASK RESULTS")
    print("=" * 88)
    for record in task_records:
        result = record["bundled"]
        judge = result["judge"]
        print(f"\n[{_verdict_label(judge)}] {record['task_id']} ({record['task_type']})")
        print(f"PROMPT:\n{record['prompt']}")
        if result["ok"]:
            output = result["result"]
            print(f"ANSWER:\n{output['text']}")
            print(f"LOCAL VALIDATION: {'PASS' if output['validation'].valid else 'FAIL'}")
        else:
            print(f"EXECUTION ERROR: {result['error']}")
        print(f"JUDGE: {judge['reason']} (judge_tokens={judge['tokens']})")


def _chat(**kwargs):
    """Import the network client only when an actual API call is requested."""
    from src.fireworks_client import chat

    return chat(**kwargs)


def _select_model() -> str:
    allowed = [entry.strip() for entry in os.environ.get("ALLOWED_MODELS", "").split(",") if entry.strip()]
    if allowed:
        return os.environ.get("MODEL", allowed[0])
    return os.environ.get("MODEL", "accounts/fireworks/models/kimi-k2p6")


def _judge(prompt: str, answer: str, model: str) -> dict[str, Any]:
    judge_prompt = (
        "Decide whether the candidate answer is correct, complete, and follows the requested format.\n\n"
        f"Task:\n{prompt}\n\nCandidate answer:\n{answer}\n\n"
        "Return exactly one line beginning with CORRECT or INCORRECT, followed by a short reason."
    )
    try:
        response = _chat(
            model=model,
            prompt=judge_prompt,
            max_tokens=60,
            system_prompt="Act as a strict answer verifier. Follow the requested verdict format exactly.",
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc), "tokens": 0}
    text = response["text"].strip()
    verdict = "correct" if text.upper().startswith("CORRECT") else "incorrect"
    return {"verdict": verdict, "reason": text, "tokens": response["total_tokens"]}


def _safe_execute_individual(task: dict[str, Any], model: str) -> dict[str, Any]:
    try:
        result = execute_task(
            task["prompt"],
            model,
            _chat,
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _metric_row() -> dict[str, Any]:
    return {
        "tasks": 0,
        "individual_calls": 0,
        "bundled_calls": 0,
        "individual_tokens": 0,
        "bundled_tokens": 0,
        "individual_correct": 0,
        "individual_judged": 0,
        "individual_judge_tokens": 0,
        "bundled_correct": 0,
        "bundled_judged": 0,
        "bundled_judge_tokens": 0,
        "individual_validation_passed": 0,
        "bundled_validation_passed": 0,
        "bundle_failures": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tiers", default="safe,cautious", help="Comma-separated: safe,cautious,experimental")
    parser.add_argument(
        "--categories",
        default="",
        help="Optional comma-separated category names or aliases, e.g. factual_knowledge, math, ner",
    )
    parser.add_argument("--max-bundle-size", type=int, default=5)
    parser.add_argument("--min-accuracy", type=float, default=80.0)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--show-results", action="store_true", help="Print every prompt, answer, verdict, and token count")
    parser.add_argument(
        "--force-ineligible",
        action="store_true",
        help="Benchmark every selected task as experimental without changing production eligibility rules",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned bundles without API calls")
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    tiers = tuple(value.strip() for value in args.tiers.split(",") if value.strip())
    try:
        category_filter = _normalize_categories(args.categories)
    except ValueError as exc:
        parser.error(str(exc))
    if args.force_ineligible and not category_filter:
        parser.error("--force-ineligible requires --categories to limit the experiment")
    model = _select_model()
    judge_model = os.environ.get("MODEL_JUDGE", DEFAULT_JUDGE_MODEL)

    task_by_id: dict[str, dict[str, Any]] = {}
    candidates = []
    excluded = Counter()
    decisions: list[dict[str, Any]] = []
    selected_tasks = 0
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id", task.get("id", f"task_{index}")))
        optimization = build_optimization(task["prompt"])
        detected_task_type = optimization["task_type"]
        dataset_category = _dataset_category(task_id, detected_task_type)
        if category_filter and dataset_category not in category_filter:
            excluded["excluded by category filter"] += 1
            continue
        selected_tasks += 1
        task_by_id[task_id] = task
        decision = assess_bundle_eligibility(task_id, task["prompt"])
        decision_record = {
            "task_id": task_id,
            "dataset_category": dataset_category,
            "task_type": detected_task_type,
            "eligible": decision.eligible,
            "forced": False,
            "reason": decision.reason,
        }
        decisions.append(decision_record)
        if not decision.eligible or decision.candidate is None:
            if args.force_ineligible:
                decision_record["forced"] = True
                candidates.append(
                    BundleCandidate(
                        task_id=task_id,
                        prompt=task["prompt"],
                        task_type=dataset_category,
                        mode=optimization["metadata"]["mode"],
                        safety_tier="experimental",
                        expected_output_tokens=int(optimization["max_tokens"]),
                        optimization=optimization,
                    )
                )
                continue
            excluded[decision.reason] += 1
            continue
        candidates.append(decision.candidate)

    bundles, singles = create_bundles(candidates, args.max_bundle_size, tiers)
    bundled_ids = {item.task_id for bundle in bundles for item in bundle.items}

    print(f"Dataset tasks: {len(tasks)}")
    print(f"Selected tasks: {selected_tasks}")
    print(f"Bundle candidates: {len(candidates)}")
    print(f"Bundled tasks: {len(bundled_ids)} in {len(bundles)} calls")
    print(f"Eligible but unbundled: {len(singles)}")
    print(f"Individual baseline calls: {len(bundled_ids)}")
    if not bundles:
        print("No bundle with at least two compatible tasks was formed.")
        return
    if args.dry_run:
        print("\nPlanned bundles:")
        for index, bundle in enumerate(bundles):
            print(
                f"  {index}: category={bundle.task_type} tier={bundle.safety_tier} "
                f"tasks={len(bundle.items)} max_tokens={bundle.max_tokens} "
                f"ids={[item.task_id for item in bundle.items]}"
            )
        if excluded:
            print(f"\nExcluded reasons: {dict(excluded)}")
        return

    individual: dict[str, dict[str, Any]] = {}
    for task_id in sorted(bundled_ids):
        task = task_by_id[task_id]
        individual[task_id] = _safe_execute_individual(task, model)

    bundled: dict[str, dict[str, Any]] = {}
    bundle_records: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles):
        try:
            result = execute_bundle(
                bundle,
                model,
                _chat,
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
            )
            bundle_records.append({
                "bundle_index": bundle_index,
                "task_type": bundle.task_type,
                "task_ids": [item.task_id for item in bundle.items],
                "max_tokens": bundle.max_tokens,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "finish_reason": result["finish_reason"],
                "raw_response": result["raw_response"],
                "error": None,
            })
            for output in result["outputs"]:
                bundled[output["task_id"]] = {"ok": True, "result": output, "bundle": result}
        except Exception as exc:
            response = exc.response if isinstance(exc, BundleExecutionError) else {}
            bundle_records.append({
                "bundle_index": bundle_index,
                "task_type": bundle.task_type,
                "task_ids": [item.task_id for item in bundle.items],
                "max_tokens": bundle.max_tokens,
                "prompt_tokens": int(response.get("prompt_tokens", 0)),
                "completion_tokens": int(response.get("completion_tokens", 0)),
                "total_tokens": int(response.get("total_tokens", 0)),
                "finish_reason": response.get("finish_reason"),
                "raw_response": str(response.get("text", "")),
                "error": str(exc),
            })
            for item in bundle.items:
                bundled[item.task_id] = {"ok": False, "error": str(exc)}

    task_records: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = defaultdict(_metric_row)
    for bundle in bundles:
        row = metrics[bundle.task_type]
        row["bundled_calls"] += 1
    for record in bundle_records:
        metrics[record["task_type"]]["bundled_tokens"] += record["total_tokens"]
        metrics[record["task_type"]]["bundle_failures"] += int(record["error"] is not None)

    for task_id in sorted(bundled_ids):
        task = task_by_id[task_id]
        candidate = next(item for item in candidates if item.task_id == task_id)
        row = metrics[candidate.task_type]
        row["tasks"] += 1
        row["individual_calls"] += 1
        individual_result = individual[task_id]
        bundled_result = bundled[task_id]
        if individual_result["ok"]:
            result = individual_result["result"]
            row["individual_tokens"] += result["total_tokens"]
            row["individual_validation_passed"] += int(result["validation"].valid)
            individual_text = result["text"]
        else:
            individual_text = ""
        if bundled_result["ok"]:
            output = bundled_result["result"]
            row["bundled_validation_passed"] += int(output["validation"].valid)
            bundled_text = output["text"]
        else:
            bundled_text = ""

        individual_judge = {"verdict": "skipped", "reason": "", "tokens": 0}
        bundled_judge = {"verdict": "skipped", "reason": "", "tokens": 0}
        if not args.skip_judge:
            if individual_result["ok"]:
                individual_judge = _judge(task["prompt"], individual_text, judge_model)
            if bundled_result["ok"]:
                bundled_judge = _judge(task["prompt"], bundled_text, judge_model)
        if individual_judge["verdict"] in {"correct", "incorrect"}:
            row["individual_judged"] += 1
            row["individual_correct"] += int(individual_judge["verdict"] == "correct")
        row["individual_judge_tokens"] += individual_judge["tokens"]
        if bundled_judge["verdict"] in {"correct", "incorrect"}:
            row["bundled_judged"] += 1
            row["bundled_correct"] += int(bundled_judge["verdict"] == "correct")
        row["bundled_judge_tokens"] += bundled_judge["tokens"]

        task_records.append({
            "task_id": task_id,
            "prompt": task["prompt"],
            "task_type": candidate.task_type,
            "safety_tier": candidate.safety_tier,
            "individual": {**individual_result, "judge": individual_judge},
            "bundled": {**bundled_result, "judge": bundled_judge},
        })

    for row in metrics.values():
        individual_tokens = row["individual_tokens"]
        row["token_savings_pct"] = (
            round((individual_tokens - row["bundled_tokens"]) / individual_tokens * 100, 2)
            if individual_tokens and not row["bundle_failures"]
            else None
        )
        row["individual_accuracy_pct"] = round(
            row["individual_correct"] / row["individual_judged"] * 100,
            2,
        ) if row["individual_judged"] else None
        row["bundled_accuracy_pct"] = round(
            row["bundled_correct"] / row["bundled_judged"] * 100,
            2,
        ) if row["bundled_judged"] else None
        if row["individual_accuracy_pct"] is not None and row["bundled_accuracy_pct"] is not None:
            row["accuracy_delta_pct_points"] = round(
                row["bundled_accuracy_pct"] - row["individual_accuracy_pct"],
                2,
            )
        else:
            row["accuracy_delta_pct_points"] = None
        if row["bundle_failures"]:
            row["recommendation"] = "reject_bundle_execution_failure"
        elif args.skip_judge or row["bundled_accuracy_pct"] is None:
            row["recommendation"] = "insufficient_accuracy_data"
        elif row["bundled_validation_passed"] < row["tasks"]:
            row["recommendation"] = "reject_validation_regression"
        elif row["bundled_accuracy_pct"] < args.min_accuracy:
            row["recommendation"] = "reject_below_accuracy_gate"
        elif row["accuracy_delta_pct_points"] < -args.max_accuracy_drop:
            row["recommendation"] = "reject_accuracy_drop"
        elif row["token_savings_pct"] is None or row["token_savings_pct"] <= 0:
            row["recommendation"] = "reject_no_token_savings"
        else:
            row["recommendation"] = "candidate_for_larger_sample"

    report = {
        "dataset": str(dataset_path),
        "model": model,
        "judge_model": None if args.skip_judge else judge_model,
        "tiers": tiers,
        "max_bundle_size": args.max_bundle_size,
        "min_accuracy": args.min_accuracy,
        "max_accuracy_drop": args.max_accuracy_drop,
        "force_ineligible": args.force_ineligible,
        "summary": dict(metrics),
        "eligibility": decisions,
        "excluded_reasons": dict(excluded),
        "bundles": bundle_records,
        "tasks": task_records,
    }
    output_path = Path(args.output) if args.output else ROOT / "results" / f"bundle_benchmark_{datetime.now():%d%m%Y%H%M%S}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if args.show_results:
        _print_full_results(task_records, bundle_records)

    print("\nCategory comparison:")
    for task_type, row in sorted(metrics.items()):
        print(
            f"  {task_type:<22} tasks={row['tasks']:<2} calls={row['individual_calls']}->{row['bundled_calls']} "
            f"tokens={row['individual_tokens']}->{row['bundled_tokens']} savings={row['token_savings_pct']}% "
            f"accuracy={row['individual_accuracy_pct']}->{row['bundled_accuracy_pct']} "
            f"delta={row['accuracy_delta_pct_points']}pp recommendation={row['recommendation']}"
        )
        print(
            " " * 24
            + f"judge_tokens={row['individual_judge_tokens']}->{row['bundled_judge_tokens']} "
            + "(reported separately; not included in production token comparison)"
        )
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
