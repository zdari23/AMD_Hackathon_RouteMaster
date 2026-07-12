"""Category-aware deterministic validators for single-call execution."""

from __future__ import annotations

import ast
import json
import re

from .parser import extract_code_block, parse_constraints, split_prompt
from .postprocessors import postprocess_output
from .types import OutputConstraints, ValidationResult


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")
_NUMBER_DATE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%?|(?:19|20)\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2})\b",
    re.I,
)
_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+#.-]*)\s*\n(?P<body>[\s\S]*?)```")
_CODE_NAME_RE = re.compile(r"\b(?:def|class|function)\s+([A-Za-z_]\w*)|`([A-Za-z_]\w*)\s*\(")
_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|FIXME|not implemented|placeholder)\b|^\s*pass\s*$", re.I | re.M)
_PREAMBLE_RE = re.compile(r"^\s*(?:here(?:'s| is) (?:the )?(?:answer|summary)|summary|answer)\s*:", re.I)


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len([part for part in _SENTENCE_RE.split(stripped) if part.strip()])


def _validate_sentiment(output: str, constraints: OutputConstraints) -> list[str]:
    errors: list[str] = []
    labels = [label for label in constraints.allowed_sentiment_labels if re.search(rf"\b{re.escape(label)}\b", output, re.I)]
    if not labels:
        errors.append("required sentiment label missing")
    if not constraints.explanation_requested and len(labels) == 1 and output.strip().lower() != labels[0].lower():
        errors.append("extra prose in label-only response")
    if constraints.explanation_requested and len(output.split()) <= 1:
        errors.append("requested sentiment reason missing")
    return errors


def _flat_entity_spans(value: object) -> list[str]:
    spans: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"text", "span", "entity", "mention"} and isinstance(item, str):
                spans.append(item)
            elif isinstance(item, list) and all(isinstance(entry, str) for entry in item):
                spans.extend(item)
            else:
                spans.extend(_flat_entity_spans(item))
    elif isinstance(value, list):
        for item in value:
            spans.extend(_flat_entity_spans(item))
    return spans


def _validate_entities(prompt: str, output: str, constraints: OutputConstraints) -> list[str]:
    if constraints.output_format != "json":
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return ["malformed JSON entity output"]
    if not isinstance(parsed, (dict, list)):
        return ["entity JSON must be an object or array"]

    errors: list[str] = []
    if isinstance(parsed, dict) and constraints.requested_entity_types:
        key_aliases = {
            "person": {"person", "people", "per"},
            "organization": {"organization", "organisation", "org"},
            "location": {"location", "loc", "gpe"},
            "date": {"date"},
            "time": {"time"},
            "money": {"money"},
            "product": {"product"},
            "event": {"event"},
        }
        allowed_keys = set().union(*(key_aliases.get(kind, {kind}) for kind in constraints.requested_entity_types))
        unexpected = [key for key in parsed if key.lower() not in allowed_keys]
        if unexpected:
            errors.append("unexpected entity keys: " + ", ".join(unexpected))

    source = split_prompt(prompt).payload or prompt
    for span in _flat_entity_spans(parsed):
        if span and span.lower() not in source.lower():
            errors.append(f"entity span not found in source: {span}")
    return errors


def _validate_summary(prompt: str, output: str, constraints: OutputConstraints) -> list[str]:
    errors: list[str] = []
    
    if constraints.word_limit:
        bullets = _BULLET_RE.split(output)
        if constraints.word_limit_per_bullet and len(bullets) > 1:
            # First element is preamble (if any), the rest are bullet texts
            for i, bullet_text in enumerate(bullets[1:]):
                words = re.findall(r"\b\S+\b", bullet_text)
                if constraints.exact_word_count and len(words) != constraints.word_limit:
                    errors.append(f"expected exactly {constraints.word_limit} words per bullet, got {len(words)} in bullet {i+1}")
                elif not constraints.exact_word_count and len(words) > constraints.word_limit:
                    errors.append(f"word limit exceeded per bullet: {len(words)} > {constraints.word_limit} in bullet {i+1}")
        else:
            words = re.findall(r"\b\S+\b", output)
            if constraints.exact_word_count and len(words) != constraints.word_limit:
                errors.append(f"expected exactly {constraints.word_limit} words, got {len(words)}")
            elif not constraints.exact_word_count and len(words) > constraints.word_limit:
                errors.append(f"word limit exceeded: {len(words)} > {constraints.word_limit}")
    if constraints.sentence_limit:
        count = _sentence_count(output)
        if constraints.exact_sentence_count and count != constraints.sentence_limit:
            errors.append(f"expected exactly {constraints.sentence_limit} sentences, got {count}")
        elif not constraints.exact_sentence_count and count > constraints.sentence_limit:
            errors.append(f"sentence limit exceeded: {count} > {constraints.sentence_limit}")
    if constraints.bullet_count:
        count = len(_BULLET_RE.findall(output))
        if count != constraints.bullet_count:
            errors.append(f"expected {constraints.bullet_count} bullets, got {count}")
    if _PREAMBLE_RE.search(output):
        errors.append("summary contains preamble")

    source = split_prompt(prompt).payload
    important = _NUMBER_DATE_RE.findall(source)
    if important and not _NUMBER_DATE_RE.search(output):
        errors.append("all source numbers and dates were lost")
    return errors


def _extract_output_code(output: str) -> tuple[str, str | None]:
    matches = list(_FENCE_RE.finditer(output))
    if matches:
        match = matches[-1]
        return match.group("body").strip(), match.group("lang").lower() or None
    return output.strip(), None


def _validate_code(task_type: str, prompt: str, output: str, constraints: OutputConstraints) -> list[str]:
    code, fenced_language = _extract_output_code(output)
    errors: list[str] = []
    if not code:
        return ["empty code output"]
    if _PLACEHOLDER_RE.search(code):
        errors.append("code contains a placeholder or stub")

    requested_language = constraints.programming_language
    if requested_language and fenced_language:
        normalized = {"py": "python", "js": "javascript", "ts": "typescript", "cpp": "c++", "cs": "c#"}
        actual = normalized.get(fenced_language, fenced_language)
        if actual != requested_language:
            errors.append(f"wrong programming language: expected {requested_language}, got {actual}")

    if requested_language == "python":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"python syntax error: {exc.msg}")
    elif requested_language in {"javascript", "typescript", "java", "c++", "c#", "go", "rust"}:
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            if code.count(opening) != code.count(closing):
                errors.append(f"unbalanced {opening}{closing} delimiters")
                break

    source_code = extract_code_block(prompt)
    names_source = source_code if task_type == "bug_fixing" and source_code else prompt[:2000]
    requested_names = {a or b for a, b in _CODE_NAME_RE.findall(names_source) if a or b}
    missing = sorted(name for name in requested_names if not re.search(rf"\b{re.escape(name)}\b", code))
    if missing:
        errors.append("required code names missing: " + ", ".join(missing[:5]))
    if constraints.explanation_requested and output.strip() == code and not _FENCE_RE.search(output):
        # A code-only response cannot satisfy an explicit explanation request.
        errors.append("requested debugging explanation missing")
    return errors


def _validate_logic(output: str, constraints: OutputConstraints) -> list[str]:
    if not output.strip():
        return ["empty logical conclusion"]
    assignments = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*[-=:]", output)
    if len(assignments) != len(set(assignments)):
        return ["duplicate assignment detected"]
    return []


def _validate_math(output: str) -> list[str]:
    if not re.search(r"[-+]?\d+(?:[.,]\d+)?", output):
        return ["numeric mathematical result missing"]
    return []


def validate_output(
    task_type: str,
    prompt: str,
    output: str | None,
    finish_reason: str | None = None,
    constraints: OutputConstraints | None = None,
) -> ValidationResult:
    """Repair locally, then run the deterministic validator for the routed category."""
    constraints = constraints or parse_constraints(prompt)
    raw = "" if output is None else str(output)
    repaired, _ = postprocess_output(task_type, prompt, raw, constraints)
    errors: list[str] = []

    if finish_reason == "length":
        errors.append("truncated response")
    if not repaired.strip():
        errors.append("empty answer")
    if errors:
        return ValidationResult(False, errors, repaired or None)

    if task_type == "sentiment_analysis":
        errors.extend(_validate_sentiment(repaired, constraints))
    elif task_type == "entity_extraction":
        errors.extend(_validate_entities(prompt, repaired, constraints))
    elif task_type == "summarization":
        errors.extend(_validate_summary(prompt, repaired, constraints))
    elif task_type in {"bug_fixing", "code_authoring"}:
        code_candidate = raw if _FENCE_RE.search(raw) else repaired
        errors.extend(_validate_code(task_type, prompt, code_candidate, constraints))
    elif task_type == "logical_puzzles":
        errors.extend(_validate_logic(repaired, constraints))
    elif task_type == "math_solving":
        errors.extend(_validate_math(repaired))

    return ValidationResult(
        valid=not errors,
        errors=errors,
        repaired_output=repaired,
    )


def validator_name(task_type: str) -> str:
    return {
        "sentiment_analysis": "sentiment",
        "entity_extraction": "entity_schema",
        "summarization": "summary_constraints",
        "bug_fixing": "code",
        "code_authoring": "code",
        "logical_puzzles": "logic_structure",
    }.get(task_type, "basic")
