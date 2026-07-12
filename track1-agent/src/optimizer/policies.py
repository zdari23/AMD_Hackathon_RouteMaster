"""Category-specific system prompt policies."""

from __future__ import annotations

import re

from .types import OutputConstraints


BASE_POLICY = "Answer in English. Follow the user's requested format exactly. No preamble or restatement."


def _knowledge_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    view = prompt[:1200].lower()
    mode = "explanation" if re.search(r"\b(?:how|why|explain|explanation|describe)\b", view) else "direct"
    return (
        "Answer the factual question directly and accurately. Use only the detail necessary to fully answer "
        "the request. Do not add unrelated facts. Follow any requested length or format exactly.",
        mode,
    )


def _math_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    mode = "show_steps" if constraints.steps_requested or constraints.explanation_requested else "final_only"
    ending = "" if constraints.output_format != "text" else " If no format is specified, end with 'Answer: <value>'."
    
    if mode == "final_only":
        policy = (
            "Solve accurately. Preserve units, percentages, rounding instructions, and constraints. "
            "DO NOT output any reasoning, explanations, or steps. Return strictly the concise final result only." + ending
        )
    else:
        policy = (
            "Solve accurately. Preserve units, percentages, rounding instructions, and constraints. "
            "Show your step-by-step reasoning." + ending
        )
    return policy, mode


def _sentiment_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    multi_target = bool(re.search(r"\b(?:aspect|multi-target|toward|towards|each)\b", prompt[:1600], re.I))
    if constraints.explanation_requested:
        mode = "multi_label_reason" if multi_target else "label_reason"
        policy = "Use only labels requested by the user and give exactly the requested concise justification."
    else:
        mode = "multi_label" if multi_target else "label_only"
        policy = "Classify using only labels requested by the user. Return only the requested label output, with no extra prose."
    return policy, mode


def _summary_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    view = prompt[:1600]
    if constraints.bullet_count:
        mode = "bullet_summary"
    elif constraints.sentence_limit == 1:
        mode = "one_sentence"
    elif constraints.word_limit:
        mode = "word_limited"
    elif constraints.sentence_limit:
        mode = "sentence_limited"
    elif re.search(r"\b(?:action items|decisions|deadlines|next steps)\b", view, re.I):
        mode = "action_items"
    else:
        mode = "general_summary"
    return (
        "Summarize the source faithfully and concisely. Preserve the main point, key facts, names, numbers, "
        "dates, decisions, deadlines, and constraints. Do not add unsupported information. Follow the "
        "requested length and format exactly.",
        mode,
    )


def _entity_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    mode = "json" if constraints.output_format == "json" else "requested_format"
    return (
        "Extract only the requested entity types. Copy entity spans exactly as they appear in the source unless "
        "the user explicitly requests normalization or canonicalization. Do not infer, expand, or explain. "
        "Follow the requested output format exactly and include no extra prose.",
        mode,
    )


def _bug_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    if constraints.explanation_requested:
        mode = "code_plus_short_explanation"
    elif re.search(r"\b(?:patch|diff|minimal change)\b", prompt[:1600], re.I):
        mode = "minimal_patch"
    else:
        mode = "code_only"
    return (
        "Fix the code while preserving the intended behavior, programming language, function signatures, "
        "inputs, and outputs. Return only the complete corrected code unless an explanation is explicitly "
        "requested. Preserve necessary validation and error handling.",
        mode,
    )


def _logic_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    view = prompt[:1600]
    if constraints.explanation_requested or constraints.steps_requested:
        mode = "explanation_requested"
    elif re.search(r"\b(?:complete assignment|complete schedule|for each|all assignments)\b", view, re.I):
        mode = "full_assignment"
    else:
        mode = "final_only"
    return (
        "Solve the puzzle while satisfying every stated constraint. Verify the final result against all "
        "constraints. Return only the requested conclusion unless an explanation is explicitly requested.",
        mode,
    )


def _authoring_policy(prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    view = prompt[:1600]
    if constraints.explanation_requested:
        mode = "code_plus_explanation"
    elif re.search(r"\bclass\b|\bmultiple (?:functions|methods)\b", view, re.I):
        mode = "class_or_multiple_methods"
    elif re.search(r"\b(?:script|program|cli|command line)\b", view, re.I):
        mode = "script"
    elif re.search(r"\b(?:efficient|complexity|dynamic programming|graph|algorithm)\b", view, re.I):
        mode = "algorithmic_function"
    else:
        mode = "single_function"
    return (
        "Write complete, correct, well-structured code in the requested language. Follow the requested function "
        "signature, inputs, outputs, constraints, and edge cases exactly. Return only the code unless an "
        "explanation is explicitly requested. Avoid unnecessary dependencies and abstractions.",
        mode,
    )


def build_system_policy(task_type: str, prompt: str, constraints: OutputConstraints) -> tuple[str, str]:
    """Return the concise system prompt and selected category mode."""
    builders = {
        "knowledge_qa": _knowledge_policy,
        "math_solving": _math_policy,
        "sentiment_analysis": _sentiment_policy,
        "summarization": _summary_policy,
        "entity_extraction": _entity_policy,
        "bug_fixing": _bug_policy,
        "logical_puzzles": _logic_policy,
        "code_authoring": _authoring_policy,
    }
    builder = builders.get(task_type)
    if not builder:
        return f"{BASE_POLICY} Answer accurately and completely, using only necessary detail.", "safe_fallback"
    category_policy, mode = builder(prompt, constraints)
    return f"{BASE_POLICY} {category_policy}", mode
