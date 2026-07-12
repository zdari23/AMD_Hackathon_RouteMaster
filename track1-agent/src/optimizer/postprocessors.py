"""Safe local output repairs performed after the single API call."""

from __future__ import annotations

import json
import re
from typing import Any

from .types import OutputConstraints


_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+#.-]*)\s*\n(?P<body>[\s\S]*?)```")
_PREAMBLE_RE = re.compile(r"^\s*(?:here(?:'s| is) (?:the )?(?:answer|summary)|summary|answer)\s*:\s*", re.I)


def _extract_first_json(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index:index + end]
    return None


def _dedupe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _dedupe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        result: list[Any] = []
        fingerprints: set[str] = set()
        for item in value:
            repaired = _dedupe_json(item)
            fingerprint = json.dumps(repaired, sort_keys=True, ensure_ascii=False)
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                result.append(repaired)
        return result
    return value


def _repair_json(text: str) -> str | None:
    candidate = _extract_first_json(text)
    if candidate is None:
        fence = _FENCE_RE.search(text)
        candidate = _extract_first_json(fence.group("body")) if fence else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return json.dumps(_dedupe_json(parsed), ensure_ascii=False, separators=(",", ":"))


def _repair_sentiment(text: str, constraints: OutputConstraints) -> str | None:
    allowed = constraints.allowed_sentiment_labels
    matches: list[tuple[int, str]] = []
    for label in allowed:
        match = re.search(rf"\b{re.escape(label)}\b", text, re.I)
        if match:
            matches.append((match.start(), label))
    if not matches:
        return None
    _, label = min(matches)
    if not constraints.explanation_requested:
        return label
    stripped = text.strip()
    reason = re.sub(rf"^\s*(?:sentiment\s*:\s*)?{re.escape(label)}\s*[-—:,.]*\s*", "", stripped, flags=re.I)
    return f"{label} — {reason}" if reason else label


def _repair_code(text: str, constraints: OutputConstraints) -> str:
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text.strip()
    if constraints.explanation_requested and not constraints.code_only:
        return text.strip()
    return matches[-1].group("body").strip()


def postprocess_output(task_type: str, prompt: str, output: str, constraints: OutputConstraints) -> tuple[str, bool]:
    """Apply only deterministic repairs that cannot change answer semantics."""
    original = "" if output is None else str(output)
    repaired = original.strip()

    if task_type == "sentiment_analysis":
        multi_target = bool(re.search(r"\b(?:aspect|multi-target|toward|towards|each)\b", prompt[:1600], re.I))
        if not multi_target:
            local = _repair_sentiment(repaired, constraints)
            if local is not None:
                repaired = local
    elif task_type == "entity_extraction" and constraints.output_format == "json":
        local = _repair_json(repaired)
        if local is not None:
            repaired = local
    elif task_type in ("bug_fixing", "code_authoring"):
        repaired = _repair_code(repaired, constraints)
    elif task_type == "summarization":
        repaired = _PREAMBLE_RE.sub("", repaired).strip()
        if len(repaired) >= 2 and repaired[0] == repaired[-1] and repaired[0] in "'\"":
            repaired = repaired[1:-1].strip()

    return repaired, repaired != original
