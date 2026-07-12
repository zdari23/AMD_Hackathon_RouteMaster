"""Run the real router + production bundling path and report accuracy/tokens."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer import build_optimization, execute_task, solve_tasks


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


def _chat(**kwargs):
    from src.fireworks_client import chat

    return chat(**kwargs)


def _select_model() -> str:
    allowed = [entry.strip() for entry in os.environ.get("ALLOWED_MODELS", "").split(",") if entry.strip()]
    if allowed:
        return os.environ.get("MODEL", allowed[0])
    return os.environ.get("MODEL", "accounts/fireworks/models/kimi-k2p6")


def _dataset_category(task_id: str) -> str:
    return DATASET_PREFIX_CATEGORIES.get(task_id.split("_", 1)[0].lower(), "unknown")


def _judge(prompt: str, answer: str, model: str, chat_fn: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    judge_prompt = (
        "Decide whether the candidate answer is correct, complete, and follows every requested format constraint.\n\n"
        f"Task:\n{prompt}\n\nCandidate answer:\n{answer}\n\n"
        "Return exactly one line beginning with CORRECT or INCORRECT, followed by a short reason."
    )
    try:
        response = chat_fn(
            model=model,
            prompt=judge_prompt,
            max_tokens=60,
            system_prompt="Act as a strict answer verifier. Follow the verdict format exactly.",
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc), "tokens": 0}
    text = str(response.get("text", "")).strip()
    verdict = "correct" if text.upper().startswith("CORRECT") else "incorrect"
    return {"verdict": verdict, "reason": text, "tokens": int(response.get("total_tokens", 0))}


def evaluate(
    tasks: list[dict[str, Any]],
    model: str,
    judge_model: str,
    chat_fn: Callable[..., dict[str, Any]],
    judge_fn: Callable[[str, str, str, Callable[..., dict[str, Any]]], dict[str, Any]] = _judge,
    skip_judge: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    plans = [build_optimization(str(task["prompt"])) for task in tasks]
    if progress:
        distribution = Counter(plan["task_type"] for plan in plans)
        print(f"Router completed: {dict(distribution)}", flush=True)
        fallback_ids = [task["task_id"] for task, plan in zip(tasks, plans) if plan["task_type"] == "fallback"]
        print(f"Fallback tasks: {fallback_ids or 'none'}", flush=True)
        print("Running production solve_tasks()...", flush=True)
    solved = solve_tasks(
        tasks,
        model,
        chat_fn,
        extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
    )
    if progress:
        print(
            f"Production completed: {solved['bundle_calls']} bundled + "
            f"{solved['individual_calls']} individual calls, {solved['total_tokens']} tokens",
            flush=True,
        )

    call_metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"bundle_calls": 0, "individual_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    for call in solved["api_call_records"]:
        row = call_metrics[call["task_type"]]
        row[f"{call['call_type']}_calls"] += int(call.get("api_calls", 1))
        row["prompt_tokens"] += int(call["prompt_tokens"])
        row["completion_tokens"] += int(call["completion_tokens"])
        row["total_tokens"] += int(call["total_tokens"])

    records = []
    category_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tasks": 0, "correct": 0, "incorrect": 0, "judge_errors": 0, "judge_tokens": 0}
    )
    for position, (task, plan, answer) in enumerate(zip(tasks, plans, solved["results"]), 1):
        if skip_judge:
            judge = {"verdict": "skipped", "reason": "judge disabled", "tokens": 0}
        elif answer.get("error"):
            judge = {"verdict": "error", "reason": answer["error"], "tokens": 0}
        else:
            judge = judge_fn(str(task["prompt"]), str(answer["text"]), judge_model, chat_fn)
        if progress:
            print(f"Judge {position}/{len(tasks)}: {task['task_id']} -> {judge['verdict'].upper()}", flush=True)
        detected = plan["task_type"]
        metrics = category_metrics[detected]
        metrics["tasks"] += 1
        metrics["judge_tokens"] += int(judge["tokens"])
        if judge["verdict"] == "correct":
            metrics["correct"] += 1
        elif judge["verdict"] == "incorrect":
            metrics["incorrect"] += 1
        elif judge["verdict"] == "error":
            metrics["judge_errors"] += 1
        records.append({
            "task_id": task["task_id"],
            "dataset_category": _dataset_category(str(task["task_id"])),
            "detected_category": detected,
            "router_confidence": plan["confidence"],
            "router_scores": plan["metadata"]["scores"],
            "matched_signals": plan["metadata"]["matched_signals"],
            "mode": plan["metadata"]["mode"],
            "solver_type": answer.get("solver_type", "error"),
            "validation_valid": bool(answer.get("validation") and answer["validation"].valid),
            "prompt": task["prompt"],
            "answer": answer["text"],
            "execution_error": answer.get("error"),
            "judge": judge,
        })

    all_categories = set(category_metrics) | set(call_metrics)
    category_report = {}
    for category in sorted(all_categories):
        judged = category_metrics[category]["correct"] + category_metrics[category]["incorrect"]
        category_report[category] = {
            **category_metrics[category],
            **call_metrics[category],
            "accuracy_pct": round(category_metrics[category]["correct"] / judged * 100, 2) if judged else None,
        }

    fallback_records = [record for record in records if record["detected_category"] == "fallback"]
    mismatches = [record for record in records if record["dataset_category"] != record["detected_category"]]
    total_judged = sum(row["correct"] + row["incorrect"] for row in category_metrics.values())
    total_correct = sum(row["correct"] for row in category_metrics.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "judge_model": judge_model,
        "tasks": len(tasks),
        "router_distribution": dict(Counter(plan["task_type"] for plan in plans)),
        "fallback_count": len(fallback_records),
        "fallback_task_ids": [record["task_id"] for record in fallback_records],
        "dataset_router_mismatch_count": len(mismatches),
        "dataset_router_mismatch_task_ids": [record["task_id"] for record in mismatches],
        "production": {
            "bundle_calls": solved["bundle_calls"],
            "individual_calls": solved["individual_calls"],
            "bundle_fallbacks": solved["bundle_fallbacks"],
            "prompt_tokens": solved["prompt_tokens"],
            "completion_tokens": solved["completion_tokens"],
            "total_tokens": solved["total_tokens"],
        },
        "overall_accuracy_pct": round(total_correct / total_judged * 100, 2) if total_judged else None,
        "judge_tokens": sum(row["judge_tokens"] for row in category_metrics.values()),
        "category_metrics": category_report,
        "api_call_records": solved["api_call_records"],
        "results": records,
    }


def evaluate_ab(
    tasks: list[dict[str, Any]],
    model: str,
    judge_model: str,
    chat_fn: Callable[..., dict[str, Any]],
    judge_fn: Callable[[str, str, str, Callable[..., dict[str, Any]]], dict[str, Any]] = _judge,
    skip_judge: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Compare the real production path against an all-individual baseline."""
    report = evaluate(tasks, model, judge_model, chat_fn, judge_fn, skip_judge, progress)
    baseline_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tasks": 0,
            "correct": 0,
            "incorrect": 0,
            "judge_errors": 0,
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "judge_tokens": 0,
        }
    )
    baseline_results = []
    if progress:
        print("Running all-individual A/B baseline...", flush=True)
    for position, (task, production_record) in enumerate(zip(tasks, report["results"]), 1):
        category = production_record["detected_category"]
        row = baseline_metrics[category]
        row["tasks"] += 1
        try:
            answer = execute_task(
                str(task["prompt"]),
                model,
                chat_fn,
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
            )
            row["api_calls"] += int(answer.get("api_calls", 1))
            row["prompt_tokens"] += int(answer["prompt_tokens"])
            row["completion_tokens"] += int(answer["completion_tokens"])
            row["total_tokens"] += int(answer["total_tokens"])
            if skip_judge:
                judge = {"verdict": "skipped", "reason": "judge disabled", "tokens": 0}
            else:
                judge = judge_fn(str(task["prompt"]), str(answer["text"]), judge_model, chat_fn)
            baseline_results.append({
                "task_id": task["task_id"],
                "detected_category": category,
                "answer": answer["text"],
                "validation_valid": answer["validation"].valid,
                "prompt_tokens": answer["prompt_tokens"],
                "completion_tokens": answer["completion_tokens"],
                "total_tokens": answer["total_tokens"],
                "judge": judge,
                "error": None,
            })
        except Exception as exc:
            row["api_calls"] += 1
            judge = {"verdict": "error", "reason": str(exc), "tokens": 0}
            baseline_results.append({
                "task_id": task["task_id"],
                "detected_category": category,
                "answer": "",
                "validation_valid": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "judge": judge,
                "error": str(exc),
            })
        row["judge_tokens"] += int(judge["tokens"])
        if judge["verdict"] == "correct":
            row["correct"] += 1
        elif judge["verdict"] == "incorrect":
            row["incorrect"] += 1
        elif judge["verdict"] == "error":
            row["judge_errors"] += 1
        if progress:
            print(
                f"Individual {position}/{len(tasks)}: {task['task_id']} -> {judge['verdict'].upper()} "
                f"({baseline_results[-1]['total_tokens']} tokens)",
                flush=True,
            )

    comparisons = {}
    categories = set(report["category_metrics"]) | set(baseline_metrics)
    for category in sorted(categories):
        individual = baseline_metrics[category]
        production = report["category_metrics"][category]
        individual_judged = individual["correct"] + individual["incorrect"]
        individual_accuracy = round(individual["correct"] / individual_judged * 100, 2) if individual_judged else None
        production_accuracy = production["accuracy_pct"]
        individual_tokens = individual["total_tokens"]
        production_tokens = production["total_tokens"]
        comparisons[category] = {
            "tasks": production["tasks"],
            "individual_calls": individual["api_calls"],
            "production_bundle_calls": production["bundle_calls"],
            "production_individual_calls": production["individual_calls"],
            "individual_tokens": individual_tokens,
            "production_tokens": production_tokens,
            "token_savings_pct": (
                round((individual_tokens - production_tokens) / individual_tokens * 100, 2)
                if individual_tokens else None
            ),
            "individual_accuracy_pct": individual_accuracy,
            "production_accuracy_pct": production_accuracy,
            "accuracy_delta_pp": (
                round(production_accuracy - individual_accuracy, 2)
                if production_accuracy is not None and individual_accuracy is not None else None
            ),
            "individual_judge_tokens": individual["judge_tokens"],
            "production_judge_tokens": production["judge_tokens"],
        }

    baseline_judged = sum(row["correct"] + row["incorrect"] for row in baseline_metrics.values())
    baseline_correct = sum(row["correct"] for row in baseline_metrics.values())
    baseline_total_tokens = sum(row["total_tokens"] for row in baseline_metrics.values())
    production_total_tokens = report["production"]["total_tokens"]
    report["ab_test"] = {
        "category_comparison": comparisons,
        "overall": {
            "individual_calls": sum(row["api_calls"] for row in baseline_metrics.values()),
            "production_bundle_calls": report["production"]["bundle_calls"],
            "production_individual_calls": report["production"]["individual_calls"],
            "individual_tokens": baseline_total_tokens,
            "production_tokens": production_total_tokens,
            "token_savings_pct": (
                round((baseline_total_tokens - production_total_tokens) / baseline_total_tokens * 100, 2)
                if baseline_total_tokens else None
            ),
            "individual_accuracy_pct": (
                round(baseline_correct / baseline_judged * 100, 2) if baseline_judged else None
            ),
            "production_accuracy_pct": report["overall_accuracy_pct"],
            "accuracy_delta_pp": (
                round(report["overall_accuracy_pct"] - baseline_correct / baseline_judged * 100, 2)
                if report["overall_accuracy_pct"] is not None and baseline_judged else None
            ),
            "individual_judge_tokens": sum(row["judge_tokens"] for row in baseline_metrics.values()),
            "production_judge_tokens": report["judge_tokens"],
        },
        "individual_results": baseline_results,
    }
    return report


def _print_report(report: dict[str, Any], show_results: bool) -> None:
    print("\nROUTER DISTRIBUTION")
    for category, count in sorted(report["router_distribution"].items()):
        print(f"  {category:<22} {count}")
    print(f"  fallback task ids: {report['fallback_task_ids'] or 'none'}")
    print(f"  dataset/router mismatches: {report['dataset_router_mismatch_task_ids'] or 'none'}")

    print("\nCATEGORY RESULTS (judge tokens are separate from production tokens)")
    print("category               tasks accuracy  calls(B/I) production_tokens judge_tokens")
    for category, row in report["category_metrics"].items():
        accuracy = "n/a" if row["accuracy_pct"] is None else f"{row['accuracy_pct']:.2f}%"
        print(
            f"{category:<22} {row['tasks']:<5} {accuracy:<9} "
            f"{row['bundle_calls']}/{row['individual_calls']:<9} {row['total_tokens']:<17} {row['judge_tokens']}"
        )

    production = report["production"]
    print("\nOVERALL")
    print(f"  accuracy: {report['overall_accuracy_pct']}%")
    print(f"  API calls: {production['bundle_calls']} bundled + {production['individual_calls']} individual")
    print(
        f"  production tokens: prompt={production['prompt_tokens']} "
        f"completion={production['completion_tokens']} total={production['total_tokens']}"
    )
    print(f"  judge tokens (separate): {report['judge_tokens']}")

    if "ab_test" in report:
        print("\nA/B COMPARISON (production tokens exclude judge tokens)")
        print("category               tasks tokens(I/P) savings   accuracy(I/P) delta")
        for category, row in report["ab_test"]["category_comparison"].items():
            savings = "n/a" if row["token_savings_pct"] is None else f"{row['token_savings_pct']:.2f}%"
            individual_accuracy = "n/a" if row["individual_accuracy_pct"] is None else f"{row['individual_accuracy_pct']:.2f}%"
            production_accuracy = "n/a" if row["production_accuracy_pct"] is None else f"{row['production_accuracy_pct']:.2f}%"
            delta = "n/a" if row["accuracy_delta_pp"] is None else f"{row['accuracy_delta_pp']:+.2f}pp"
            print(
                f"{category:<22} {row['tasks']:<5} {row['individual_tokens']}/{row['production_tokens']:<11} "
                f"{savings:<9} {individual_accuracy}/{production_accuracy:<15} {delta}"
            )
        overall = report["ab_test"]["overall"]
        overall_savings = "n/a" if overall["token_savings_pct"] is None else f"{overall['token_savings_pct']:.2f}%"
        overall_individual_accuracy = (
            "n/a" if overall["individual_accuracy_pct"] is None else f"{overall['individual_accuracy_pct']:.2f}%"
        )
        overall_production_accuracy = (
            "n/a" if overall["production_accuracy_pct"] is None else f"{overall['production_accuracy_pct']:.2f}%"
        )
        overall_delta = "n/a" if overall["accuracy_delta_pp"] is None else f"{overall['accuracy_delta_pp']:+.2f}pp"
        print(
            f"OVERALL                {report['tasks']:<5} "
            f"{overall['individual_tokens']}/{overall['production_tokens']:<11} "
            f"{overall_savings:<9} "
            f"{overall_individual_accuracy}/{overall_production_accuracy:<15} {overall_delta}"
        )

    if show_results:
        print("\nTASK RESULTS")
        for record in report["results"]:
            verdict = record["judge"]["verdict"].upper()
            print(
                f"\n[{verdict}] {record['task_id']} dataset={record['dataset_category']} "
                f"detected={record['detected_category']} confidence={record['router_confidence']} "
                f"solver={record['solver_type']}"
            )
            print(f"ANSWER:\n{record['answer']}")
            print(f"JUDGE: {record['judge']['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default=_select_model())
    parser.add_argument("--judge-model", default="accounts/fireworks/models/glm-5p2")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--show-results", action="store_true")
    parser.add_argument("--ab-test", action="store_true", help="Also run every task individually for A/B comparison")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    evaluator = evaluate_ab if args.ab_test else evaluate
    report = evaluator(tasks, args.model, args.judge_model, _chat, skip_judge=args.skip_judge, progress=True)
    report["dataset"] = str(dataset_path)
    _print_report(report, args.show_results)

    output = Path(args.output) if args.output else ROOT / "results" / f"production_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
