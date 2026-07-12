"""Single-call production execution pipeline with local output repair."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from .bundling import (
    CATEGORY_MAX_BUNDLE_SIZE,
    BundleExecutionError,
    assess_bundle_eligibility,
    create_bundles,
    execute_bundle,
)
from .optimizer import build_optimization
from .parser import parse_constraints
from .validators import validate_output


ChatFunction = Callable[..., dict[str, Any]]


def execute_task(
    prompt: str,
    model: str,
    chat_fn: ChatFunction,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one task with exactly one API call, then repair and validate locally."""
    optimization = build_optimization(prompt)
    task_type = optimization["task_type"]
    constraints = parse_constraints(prompt)
    metadata = optimization["metadata"]
    metadata["postprocess_applied"] = False
    calls: list[dict[str, Any]] = []
    solver_type = "api"

    answer = chat_fn(
        model=model,
        prompt=prompt,
        max_tokens=optimization["max_tokens"],
        system_prompt=optimization["system_prompt"],
        extra_params=extra_params or {},
    )
    calls.append(answer)

    if answer.get("finish_reason") == "length":
        optimization["max_tokens"] = min(2048, optimization["max_tokens"] * 2 + 128)
        answer = chat_fn(
            model=model,
            prompt=prompt,
            max_tokens=optimization["max_tokens"],
            system_prompt=optimization["system_prompt"],
            extra_params=extra_params or {},
        )
        calls.append(answer)

    validation = validate_output(
        task_type,
        prompt,
        answer.get("text", ""),
        answer.get("finish_reason"),
        constraints,
    )
    original_text = str(answer.get("text", ""))
    if validation.repaired_output is not None:
        answer["text"] = validation.repaired_output
        metadata["postprocess_applied"] = validation.repaired_output != original_text

    total_tokens = sum(int(call.get("total_tokens", 0)) for call in calls)
    prompt_tokens = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0)) for call in calls)

    return {
        "text": str(answer.get("text", "")),
        "finish_reason": answer.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "api_calls": len(calls),
        "solver_type": solver_type,
        "validation": validation,
        "optimization": optimization,
    }


def solve_tasks(
    tasks: Iterable[dict[str, Any]],
    model: str,
    chat_fn: ChatFunction,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle same-category tasks, preserving input order and per-task validation."""
    task_list = list(tasks)
    candidates = []
    individual_indexes: set[int] = set()
    individual_reasons: dict[int, str] = {}
    for index, task in enumerate(task_list):
        decision = assess_bundle_eligibility(str(index), str(task["prompt"]))
        if decision.eligible and decision.candidate is not None:
            candidates.append(decision.candidate)
        else:
            individual_indexes.add(index)
            individual_reasons[index] = decision.reason

    bundles, unbundled = create_bundles(
        candidates,
        max_bundle_size=CATEGORY_MAX_BUNDLE_SIZE,
        allowed_tiers=("safe", "cautious", "experimental"),
    )
    for item in unbundled:
        index = int(item.task_id)
        individual_indexes.add(index)
        individual_reasons[index] = "eligible category singleton"

    results: dict[int, dict[str, Any]] = {}
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    bundle_calls = 0
    individual_calls = 0
    bundle_fallbacks = 0
    api_call_records: list[dict[str, Any]] = []

    def run_individual(index: int, reason: str = "router_fallback_or_singleton") -> None:
        nonlocal total_tokens, prompt_tokens, completion_tokens, individual_calls
        task = task_list[index]
        try:
            result = execute_task(
                prompt=str(task["prompt"]),
                model=model,
                chat_fn=chat_fn,
                extra_params=extra_params,
            )
            total_tokens += result["total_tokens"]
            prompt_tokens += result["prompt_tokens"]
            completion_tokens += result["completion_tokens"]
            individual_calls += int(result.get("api_calls", 1))
            results[index] = {"task_id": task["task_id"], **result}
            api_call_records.append({
                "call_type": "individual",
                "task_type": result["optimization"]["task_type"],
                "task_ids": [task["task_id"]],
                "reason": reason,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "api_calls": int(result.get("api_calls", 1)),
                "error": None,
            })
        except Exception as exc:  # Preserve the container's per-task fault isolation.
            individual_calls += 1
            results[index] = {
                "task_id": task["task_id"],
                "text": "Unable to process task.",
                "solver_type": "error",
                "error": str(exc),
            }
            api_call_records.append({
                "call_type": "individual",
                "task_type": "fallback",
                "task_ids": [task["task_id"]],
                "reason": reason,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 1,
                "error": str(exc),
            })

    for bundle in bundles:
        bundle_indexes = [int(item.task_id) for item in bundle.items]
        try:
            bundled = execute_bundle(bundle, model, chat_fn, extra_params)
            bundle_calls += 1
            total_tokens += bundled["total_tokens"]
            prompt_tokens += bundled["prompt_tokens"]
            completion_tokens += bundled["completion_tokens"]
            api_call_records.append({
                "call_type": "bundle",
                "task_type": bundle.task_type,
                "task_ids": [task_list[index]["task_id"] for index in bundle_indexes],
                "reason": "same_category_bundle",
                "prompt_tokens": bundled["prompt_tokens"],
                "completion_tokens": bundled["completion_tokens"],
                "total_tokens": bundled["total_tokens"],
                "api_calls": 1,
                "error": None,
            })
            for item, output in zip(bundle.items, bundled["outputs"]):
                index = int(item.task_id)
                if not output["validation"].valid:
                    bundle_fallbacks += 1
                    run_individual(index, "bundle_validation_fallback")
                    continue
                results[index] = {
                    "task_id": task_list[index]["task_id"],
                    "text": output["text"],
                    "finish_reason": bundled["finish_reason"],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "solver_type": "api_bundled",
                    "validation": output["validation"],
                    "optimization": item.optimization,
                }
        except BundleExecutionError as exc:
            bundle_calls += 1
            bundle_fallbacks += len(bundle_indexes)
            response = exc.response
            total_tokens += int(response.get("total_tokens", 0))
            prompt_tokens += int(response.get("prompt_tokens", 0))
            completion_tokens += int(response.get("completion_tokens", 0))
            api_call_records.append({
                "call_type": "bundle",
                "task_type": bundle.task_type,
                "task_ids": [task_list[index]["task_id"] for index in bundle_indexes],
                "reason": "same_category_bundle",
                "prompt_tokens": int(response.get("prompt_tokens", 0)),
                "completion_tokens": int(response.get("completion_tokens", 0)),
                "total_tokens": int(response.get("total_tokens", 0)),
                "api_calls": 1,
                "error": str(exc),
            })
            for index in bundle_indexes:
                run_individual(index, "bundle_parse_fallback")
        except Exception as exc:
            bundle_calls += 1
            bundle_fallbacks += len(bundle_indexes)
            api_call_records.append({
                "call_type": "bundle",
                "task_type": bundle.task_type,
                "task_ids": [task_list[index]["task_id"] for index in bundle_indexes],
                "reason": "same_category_bundle",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 1,
                "error": str(exc),
            })
            for index in bundle_indexes:
                run_individual(index, "bundle_execution_fallback")

    for index in sorted(individual_indexes):
        if index not in results:
            run_individual(index, individual_reasons.get(index, "router_fallback_or_singleton"))

    return {
        "results": [results[index] for index in range(len(task_list))],
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "bundle_calls": bundle_calls,
        "individual_calls": individual_calls,
        "bundle_fallbacks": bundle_fallbacks,
        "api_call_records": api_call_records,
    }


def structured_log(result: dict[str, Any]) -> str:
    """Serialize non-sensitive optimizer telemetry for optional logging."""
    optimization = result["optimization"]
    metadata = optimization["metadata"]
    validation = result["validation"]
    payload = {
        "task_type": optimization["task_type"],
        "confidence": optimization["confidence"],
        "scores": metadata["scores"],
        "matched_signals": metadata["matched_signals"],
        "max_tokens": optimization["max_tokens"],
        "token_policy_reason": metadata["token_policy_reason"],
        "validator": optimization["validator_name"],
        "postprocess_applied": metadata["postprocess_applied"],
        "validation_errors": validation.errors,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
