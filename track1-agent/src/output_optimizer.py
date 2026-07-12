"""Backward-compatible adapter for the modular dynamic output optimizer.

New callers should use :func:`route_task` when they need confidence and signal
details, then pass that decision to :func:`build_optimization`. Legacy callers
can continue using ``detect_task_type`` and ``get_dynamic_limits``.
"""

from __future__ import annotations

from typing import Any

from .optimizer import RouteDecision, build_optimization, route_task
from .optimizer.token_budget import PREVIOUS_VERSION_MAX_TOKENS


# Kept for import compatibility. Values represent safe category ceilings, not
# the dynamic cap returned for an individual prompt.
TOKEN_LIMITS = {
    category: {"system": "", "cap": cap}
    for category, cap in PREVIOUS_VERSION_MAX_TOKENS.items()
}


def detect_task_type(user_prompt: str) -> str:
    """Return the legacy string category name."""
    return route_task(user_prompt).task_type


def detect_task_type_detailed(user_prompt: str) -> RouteDecision:
    """Return the weighted route decision with confidence and matched signals."""
    return route_task(user_prompt)


def get_dynamic_limits(task_type: str, prompt: str) -> dict[str, Any]:
    """Return legacy ``system``/``cap`` keys plus safe optimizer metadata."""
    optimization = build_optimization(prompt, task_type)
    return {
        "system": optimization["system_prompt"],
        "cap": optimization["max_tokens"],
        "confidence": optimization["confidence"],
        "validator_name": optimization["validator_name"],
        "metadata": optimization["metadata"],
    }


__all__ = [
    "TOKEN_LIMITS",
    "build_optimization",
    "detect_task_type",
    "detect_task_type_detailed",
    "get_dynamic_limits",
    "route_task",
]
