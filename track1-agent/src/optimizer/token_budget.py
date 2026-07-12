"""Dynamic completion-token budgets derived from explicit output requirements."""

from __future__ import annotations

import math
import re

from .parser import extract_code_block, split_prompt
from .types import OutputConstraints, RouteDecision, TokenPolicy


LOW_CONFIDENCE_MULTIPLIER = 1.25
LOW_CONFIDENCE_THRESHOLD = 0.62

# Legacy ceilings retained for benchmark comparisons. Active dynamic ceilings
# are returned by each category builder and exposed as category_max_tokens.
PREVIOUS_VERSION_MAX_TOKENS = {
    "knowledge_qa": 96,
    "math_solving": 128,
    "sentiment_analysis": 40,
    "summarization": 128,
    "entity_extraction": 200,
    "bug_fixing": 160,
    "logical_puzzles": 128,
    "code_authoring": 320,
    "fallback": 256,
}


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free token estimate suitable for budgeting."""
    if not text:
        return 0
    words = len(re.findall(r"\S+", text))
    char_estimate = math.ceil(len(text) / 4)
    return max(words, char_estimate)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _explicit_length_budget(constraints: OutputConstraints, overhead: int = 16) -> int | None:
    if constraints.word_limit:
        return math.ceil(constraints.word_limit * 1.55) + overhead
    if constraints.sentence_limit:
        return constraints.sentence_limit * 40 + overhead
    if constraints.bullet_count:
        return constraints.bullet_count * 48 + overhead
    return None


def _knowledge_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    explicit = _explicit_length_budget(constraints)
    if explicit:
        return explicit, 48, 512, "explicit knowledge length constraint", False
    view = split_prompt(prompt).instruction.lower()
    if re.search(r"\b(?:who|where|when) (?:is|are|was|were|did)\b", view):
        return 64, 32, 512, "short fact lookup", True
    if mode == "explanation":
        detail_signals = len(re.findall(r"\b(?:explain|define|describe|discuss|compare|include|address)\b", view))
        return 160 + min(64, detail_signals * 16), 96, 512, "how/why/explanation request", False
    return 96, 48, 512, "direct factual answer", True


def _math_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    if constraints.steps_requested:
        return 256, 192, 384, "explicit show-steps request", False
    number_count = len(re.findall(r"\d+(?:\.\d+)?", prompt))
    if number_count <= 2 and len(prompt) < 240:
        return 64, 32, 384, "simple arithmetic final result", True
    if number_count >= 5 or len(prompt) >= 500:
        return 160, 96, 384, "multi-step mathematical problem", False
    if "%" in prompt or re.search(r"\bpercent", prompt, re.I):
        return 80, 48, 384, "percentage calculation", True
    return 160, 80, 384, "multi-step mathematical problem", False


def _sentiment_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    if mode == "label_only":
        return 10, 4, 128, "single sentiment label", True
    if mode == "label_reason":
        return 48, 24, 128, "label with concise reason", True
    if "reason" in mode:
        return 96, 48, 128, "multi-target sentiment with reasons", False
    return 64, 24, 128, "multi-target sentiment labels", True


def _summary_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    explicit = _explicit_length_budget(constraints, overhead=20)
    if explicit:
        return explicit, 32, 768, f"explicit {mode} constraint", True
    parts = split_prompt(prompt)
    source_tokens = estimate_tokens(parts.payload or prompt)
    ratio = 0.24 if source_tokens > 600 else 0.32
    return math.ceil(source_tokens * ratio) + 48, 96, 768, "source-length summary ratio", True


def _entity_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    parts = split_prompt(prompt)
    source_tokens = estimate_tokens(parts.payload or prompt)
    rich_schema = bool(re.search(r"\b(?:aliases|canonical|context|normalized|objects? with fields?)\b", prompt[:2400], re.I))
    normalization = bool(re.search(r"\b(?:normalize|normalise|coreference|disambiguate)\b", prompt[:2400], re.I))
    if rich_schema:
        density, overhead = 1.20, 96
    elif normalization or len(constraints.requested_entity_types) >= 4:
        density, overhead = 0.75, 64
    else:
        density, overhead = 0.38, 48
    return math.ceil(source_tokens * density) + overhead, 80, 512, "source length, schema, and entity density", not rich_schema


def _bug_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    code = extract_code_block(prompt) or split_prompt(prompt).payload
    code_tokens = estimate_tokens(code or prompt)
    overhead = 160 if constraints.explanation_requested else 96
    return math.ceil(code_tokens * 1.35) + overhead, 192, 1024, "original code size and response mode", False


def _logic_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    if constraints.steps_requested:
        return 320, 256, 384, "explicit full reasoning", False
    if constraints.explanation_requested:
        return 224, 160, 384, "conclusion with short support", False
    if mode == "full_assignment":
        return 192, 128, 384, "full structured assignment", True
    return 96, 64, 384, "single puzzle conclusion", True


def _authoring_budget(prompt: str, constraints: OutputConstraints, mode: str) -> tuple[int, int, int, str, bool]:
    prompt_tokens = estimate_tokens(prompt)
    ranges = {
        "single_function": (256, 192),
        "algorithmic_function": (448, 320),
        "class_or_multiple_methods": (672, 512),
        "script": (768, 640),
        "code_plus_explanation": (768, 384),
    }
    baseline, minimum = ranges.get(mode, (384, 192))
    dynamic = math.ceil(prompt_tokens * 0.70) + 128
    return max(baseline, dynamic), minimum, 1024, f"{mode} code-generation budget", False


def build_token_policy(
    task_type: str,
    prompt: str,
    constraints: OutputConstraints,
    mode: str,
    decision: RouteDecision,
) -> TokenPolicy:
    """Build and confidence-adjust a category-aware completion budget."""
    builders = {
        "knowledge_qa": _knowledge_budget,
        "math_solving": _math_budget,
        "sentiment_analysis": _sentiment_budget,
        "summarization": _summary_budget,
        "entity_extraction": _entity_budget,
        "bug_fixing": _bug_budget,
        "logical_puzzles": _logic_budget,
        "code_authoring": _authoring_budget,
    }
    builder = builders.get(task_type)
    if builder:
        raw, minimum, maximum, reason, aggressive = builder(prompt, constraints, mode)
    else:
        raw, minimum, maximum, reason, aggressive = 384, 128, 768, "safe low-confidence fallback", False

    if decision.confidence < LOW_CONFIDENCE_THRESHOLD:
        raw = math.ceil(raw * LOW_CONFIDENCE_MULTIPLIER)
        reason += "; low-confidence safety margin"
        aggressive = False
    
    cap = _clamp(raw, minimum, maximum)
    return TokenPolicy(cap, minimum, reason, aggressive, maximum)
