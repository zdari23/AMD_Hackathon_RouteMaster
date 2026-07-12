"""Generalizable, opt-in batching for A/B token and accuracy experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterable

from .optimizer import build_optimization
from .parser import extract_code_block, parse_constraints
from .token_budget import estimate_tokens
from .validators import validate_output


ChatFunction = Callable[..., dict[str, Any]]
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(?P<body>[\s\S]*?)\n```$", re.IGNORECASE)
BUNDLE_BASE_OVERHEAD = 24
BUNDLE_PER_ITEM_OVERHEAD = {
    "knowledge_qa": 16,
    "math_solving": 14,
    "sentiment_analysis": 12,
    "summarization": 16,
    "entity_extraction": 20,
    "bug_fixing": 32,
    "logical_puzzles": 18,
    "code_authoring": 40,
}
MAX_BUNDLE_TOKENS = 4096
MIN_BUNDLE_CONFIDENCE = 0.75
MIN_LOGIC_BUNDLE_CONFIDENCE = 0.85
MAX_BUNDLED_CODE_INPUT_TOKENS = 180

# Smaller bundles protect categories whose answers are long or whose tasks need
# more independent reasoning. Short classification answers can safely amortize
# the prompt across more items.
CATEGORY_MAX_BUNDLE_SIZE = {
    "knowledge_qa": 5,
    "math_solving": 5,
    "sentiment_analysis": 8,
    "summarization": 4,
    "entity_extraction": 5,
    "bug_fixing": 3,
    "logical_puzzles": 4,
    "code_authoring": 3,
}

CATEGORY_BUNDLE_POLICIES = {
    "knowledge_qa": (
        "Answer every factual question accurately and independently. Use only the detail needed to fully answer "
        "that task; preserve requested length and format and do not add unrelated facts."
    ),
    "math_solving": (
        "Solve every math task independently and verify arithmetic, units, rounding, and requested output format. "
        "Do not expose reasoning unless that task explicitly requests it."
    ),
    "sentiment_analysis": (
        "Classify every text independently using only that task's allowed labels. Preserve label casing. For every "
        "task that requests a justification, its JSON value MUST contain both the label and the complete requested "
        "justification; never return a label alone in that case."
    ),
    "summarization": (
        "Summarize every source independently and faithfully. Preserve key facts, names, numbers, and dates while "
        "obeying each task's exact word, sentence, bullet, and required-term constraints."
    ),
    "entity_extraction": (
        "Extract entities independently for every task. Use only requested entity types, copy spans exactly unless "
        "normalization is requested, and preserve each task's exact schema."
    ),
    "bug_fixing": (
        "Diagnose and fix every code task independently. Preserve language, signatures, intended behavior, edge "
        "cases, and requested explanation or code-only format."
    ),
    "logical_puzzles": (
        "Solve every puzzle independently, check the conclusion against all of that puzzle's constraints, and "
        "return exactly the requested conclusion or assignment format."
    ),
    "code_authoring": (
        "Implement every coding task independently. Preserve language, signatures, inputs, outputs, edge cases, "
        "and code-only requirements; do not mix code or assumptions between tasks."
    ),
}


class BundleExecutionError(ValueError):
    """Preserve single-call telemetry when a bundle response cannot be parsed."""

    def __init__(self, message: str, response: dict[str, Any]):
        super().__init__(message)
        self.response = response


@dataclass(frozen=True)
class BundleCandidate:
    """A task approved for an experimental category bundle."""

    task_id: str
    prompt: str
    task_type: str
    mode: str
    safety_tier: str
    expected_output_tokens: int
    optimization: dict[str, Any]


@dataclass(frozen=True)
class BundleDecision:
    """Eligibility decision independent from dataset IDs or categories."""

    eligible: bool
    reason: str
    candidate: BundleCandidate | None = None


@dataclass(frozen=True)
class TaskBundle:
    """A group whose cap is the sum of per-task caps plus envelope overhead."""

    task_type: str
    items: tuple[BundleCandidate, ...]
    max_tokens: int
    safety_tier: str


_TIER_ORDER = {"safe": 0, "cautious": 1, "experimental": 2}


def _complex_output_requested(task_type: str, prompt: str, constraints: dict[str, Any]) -> bool:
    """Detect output envelopes that are safer to execute independently."""
    view = prompt[:2400].lower()
    if constraints["output_format"] == "table":
        return True
    if constraints.get("bullet_count") and int(constraints["bullet_count"]) > 3:
        return True
    if re.search(r"\b(?:headline followed by|labeled .{0,80}\band\b|multiple sections?|nested json)\b", view):
        return True
    if task_type == "entity_extraction" and any(
        token in view for token in ("canonical", "aliases", "coreference", "normalize", "disambiguate", "confidence score")
    ):
        return True
    return False


def _contains_long_code(prompt: str) -> bool:
    code = extract_code_block(prompt)
    return bool(code and estimate_tokens(code) > MAX_BUNDLED_CODE_INPUT_TOKENS)


def _easy_logic_prompt(prompt: str, mode: str) -> bool:
    view = prompt[:2400]
    constraint_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", view))
    advanced = re.search(
        r"\b(?:if and only if|exactly one of|at least|at most|unless|either .{0,50} or|cannot both|could be true)\b",
        view,
        re.I,
    )
    return mode == "final_only" and 2 <= constraint_lines <= 6 and not advanced and estimate_tokens(view) <= 300


def _expected_output_tokens(task_type: str, optimization: dict[str, Any]) -> int:
    constraints = optimization["metadata"]["constraints"]
    if task_type == "sentiment_analysis":
        return 6
    if task_type == "math_solving":
        return 16
    if task_type == "knowledge_qa":
        return 32
    if task_type == "logical_puzzles":
        return 24
    if task_type == "entity_extraction":
        return min(64, max(32, optimization["max_tokens"] // 2))
    if task_type == "summarization":
        word_limit = constraints.get("word_limit")
        if word_limit:
            return min(64, max(16, int(word_limit * 1.4)))
        sentence_limit = constraints.get("sentence_limit")
        if sentence_limit:
            return min(64, max(20, int(sentence_limit) * 28))
    return optimization["max_tokens"]


def assess_bundle_eligibility(task_id: str, prompt: str) -> BundleDecision:
    """Use only prompt structure and optimizer metadata to assess eligibility."""
    optimization = build_optimization(prompt)
    task_type = optimization["task_type"]
    metadata = optimization["metadata"]
    mode = metadata["mode"]
    constraints = metadata["constraints"]

    if task_type in {"math_solving", "bug_fixing", "fallback"}:
        return BundleDecision(False, f"{task_type} is individual-only")
    if optimization["confidence"] < MIN_BUNDLE_CONFIDENCE:
        return BundleDecision(False, "router confidence below bundle threshold")
    if _complex_output_requested(task_type, prompt, constraints):
        return BundleDecision(False, "complex output format requires individual execution")
    if _contains_long_code(prompt):
        return BundleDecision(False, "long code input requires individual execution")

    tier: str | None = None
    if task_type == "sentiment_analysis" and mode == "label_only":
        tier = "safe"
    elif task_type == "sentiment_analysis":
        tier = "cautious"
    elif task_type == "knowledge_qa":
        tier = "cautious"
    elif task_type == "entity_extraction":
        tier = "cautious"
    elif task_type == "logical_puzzles":
        if optimization["confidence"] < MIN_LOGIC_BUNDLE_CONFIDENCE or not _easy_logic_prompt(prompt, mode):
            return BundleDecision(False, "logic task is not easy and high-confidence")
        tier = "cautious"
    elif task_type == "summarization":
        tier = "cautious"
    elif task_type == "code_authoring":
        tier = "experimental"

    if tier is None:
        return BundleDecision(False, f"{task_type}/{mode} is not bundle-safe")

    expected = _expected_output_tokens(task_type, optimization)
    return BundleDecision(
        True,
        f"eligible at {tier} tier",
        BundleCandidate(task_id, prompt, task_type, mode, tier, expected, optimization),
    )


def create_bundles(
    candidates: Iterable[BundleCandidate],
    max_bundle_size: int | dict[str, int] = 5,
    allowed_tiers: Iterable[str] = ("safe", "cautious"),
) -> tuple[list[TaskBundle], list[BundleCandidate]]:
    """Create same-category bundles while preserving each task's cap allocation."""
    allowed = set(allowed_tiers)
    grouped: dict[str, list[BundleCandidate]] = {}
    singles: list[BundleCandidate] = []
    for candidate in candidates:
        if candidate.safety_tier not in allowed:
            singles.append(candidate)
            continue
        grouped.setdefault(candidate.task_type, []).append(candidate)

    bundles: list[TaskBundle] = []
    for task_type, items in grouped.items():
        category_size = (
            int(max_bundle_size.get(task_type, CATEGORY_MAX_BUNDLE_SIZE.get(task_type, 5)))
            if isinstance(max_bundle_size, dict)
            else int(max_bundle_size)
        )
        category_size = max(2, category_size)
        per_item_overhead = BUNDLE_PER_ITEM_OVERHEAD.get(task_type, 24)
        category_bundles: list[TaskBundle] = []
        current: list[BundleCandidate] = []
        aggregate_cap = BUNDLE_BASE_OVERHEAD
        for item in items:
            item_cap = int(item.optimization["max_tokens"])
            would_fit = (
                aggregate_cap + item_cap + per_item_overhead
                <= MAX_BUNDLE_TOKENS
            )
            if current and (len(current) >= category_size or not would_fit):
                if len(current) >= 2:
                    tier = max((entry.safety_tier for entry in current), key=_TIER_ORDER.__getitem__)
                    category_bundles.append(TaskBundle(task_type, tuple(current), aggregate_cap, tier))
                else:
                    singles.extend(current)
                current = []
                aggregate_cap = BUNDLE_BASE_OVERHEAD
            current.append(item)
            aggregate_cap += item_cap + per_item_overhead
        if len(current) >= 2:
            tier = max((entry.safety_tier for entry in current), key=_TIER_ORDER.__getitem__)
            category_bundles.append(TaskBundle(task_type, tuple(current), aggregate_cap, tier))
        elif current and category_bundles and len(category_bundles[-1].items) >= 3:
            # Avoid a final singleton (for example 6 -> 5+1) by moving one
            # task from the previous bundle (4+2). This keeps all API calls
            # bundled without exceeding either category size or token caps.
            previous = category_bundles.pop()
            previous_items = previous.items[:-1]
            final_items = (previous.items[-1], current[0])
            previous_tier = max((entry.safety_tier for entry in previous_items), key=_TIER_ORDER.__getitem__)
            final_tier = max((entry.safety_tier for entry in final_items), key=_TIER_ORDER.__getitem__)
            previous_cap = BUNDLE_BASE_OVERHEAD + sum(
                int(entry.optimization["max_tokens"]) + per_item_overhead for entry in previous_items
            )
            final_cap = BUNDLE_BASE_OVERHEAD + sum(
                int(entry.optimization["max_tokens"]) + per_item_overhead for entry in final_items
            )
            category_bundles.extend(
                (
                    TaskBundle(task_type, previous_items, previous_cap, previous_tier),
                    TaskBundle(task_type, final_items, final_cap, final_tier),
                )
            )
        else:
            singles.extend(current)
        bundles.extend(category_bundles)
    return bundles, singles


def build_bundle_prompt(bundle: TaskBundle) -> tuple[str, str]:
    """Build a category-aware, index-keyed JSON envelope."""
    items = [{"key": str(index), "task": item.prompt} for index, item in enumerate(bundle.items)]
    category_policy = CATEGORY_BUNDLE_POLICIES[bundle.task_type]
    system = (
        "Process each task independently. " + category_policy + " Return exactly one valid JSON object mapping "
        "every provided key to its final answer. Do not omit, add, merge, or reorder keys. Use a JSON string value "
        "for text or code; when a task explicitly requests JSON, place its requested JSON value directly under the "
        "key. Each value must contain only that task's answer. No markdown fence or preamble."
    )
    user = "Independent tasks:\n" + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return system, user


def parse_bundle_response(text: str, expected_count: int) -> dict[str, Any]:
    """Parse an exact index-keyed JSON response without another call."""
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group("body").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"bundle response is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("bundle response must be a JSON object")
    expected = {str(index) for index in range(expected_count)}
    actual = set(parsed)
    if actual != expected:
        raise ValueError(f"bundle keys mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    return parsed


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def execute_bundle(
    bundle: TaskBundle,
    model: str,
    chat_fn: ChatFunction,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one bundle in one API call and validate each answer locally."""
    system_prompt, prompt = build_bundle_prompt(bundle)
    response = chat_fn(
        model=model,
        prompt=prompt,
        max_tokens=bundle.max_tokens,
        system_prompt=system_prompt,
        extra_params=extra_params or {},
    )
    try:
        parsed = parse_bundle_response(str(response.get("text", "")), len(bundle.items))
    except ValueError as exc:
        raise BundleExecutionError(str(exc), response) from exc
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(bundle.items):
        text = _answer_text(parsed[str(index)])
        constraints = parse_constraints(item.prompt)
        validation = validate_output(item.task_type, item.prompt, text, response.get("finish_reason"), constraints)
        repaired = validation.repaired_output if validation.repaired_output is not None else text
        outputs.append({"task_id": item.task_id, "text": repaired, "validation": validation})
    return {
        "task_type": bundle.task_type,
        "safety_tier": bundle.safety_tier,
        "max_tokens": bundle.max_tokens,
        "prompt_tokens": int(response.get("prompt_tokens", 0)),
        "completion_tokens": int(response.get("completion_tokens", 0)),
        "total_tokens": int(response.get("total_tokens", 0)),
        "finish_reason": response.get("finish_reason"),
        "raw_response": str(response.get("text", "")),
        "outputs": outputs,
    }
