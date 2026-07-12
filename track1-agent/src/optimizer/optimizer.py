"""Public orchestration API for routing, constraints, policies, and budgets."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .parser import parse_constraints, split_prompt
from .policies import build_system_policy
from .router import CATEGORIES, FALLBACK, route_task
from .token_budget import build_token_policy, estimate_tokens
from .types import Optimization, RouteDecision
from .validators import validator_name


def _forced_decision(task_type: str) -> RouteDecision:
    known = task_type if task_type in CATEGORIES else FALLBACK
    scores = {category: (16 if category == known else 0) for category in CATEGORIES}
    return RouteDecision(known, 1.0 if known != FALLBACK else 0.0, scores, {known: ["caller_override"]})


def build_optimization(
    prompt: str,
    decision: RouteDecision | str | None = None,
) -> dict[str, Any]:
    """Return a serializable optimization plan without logging prompt content."""
    if isinstance(decision, str):
        route = _forced_decision(decision)
    else:
        route = decision or route_task(prompt)
    constraints = parse_constraints(prompt)
    parts = split_prompt(prompt)
    system_prompt, mode = build_system_policy(route.task_type, prompt, constraints)
    token_policy = build_token_policy(route.task_type, prompt, constraints, mode, route)

    metadata: dict[str, Any] = {
        "scores": route.scores,
        "matched_signals": route.matched_signals,
        "mode": mode,
        "token_policy_reason": token_policy.reason,
        "min_tokens": token_policy.min_tokens,
        "category_max_tokens": token_policy.category_max,
        "aggressive": token_policy.aggressive,
        "constraints": asdict(constraints),
        "instruction_tokens_estimate": estimate_tokens(parts.instruction),
        "payload_tokens_estimate": estimate_tokens(parts.payload),
        "postprocess_applied": False,
    }
    optimization = Optimization(
        task_type=route.task_type,
        confidence=route.confidence,
        system_prompt=system_prompt,
        max_tokens=token_policy.max_tokens,
        validator_name=validator_name(route.task_type),
        metadata=metadata,
    )
    return optimization.to_dict()
