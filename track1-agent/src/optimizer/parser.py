"""Prompt structure and explicit output-constraint parsing."""

from __future__ import annotations

import re

from .types import OutputConstraints, PromptParts


_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+#.-]*)\s*\n(?P<code>[\s\S]*?)```")
_SECTION_RE = re.compile(
    r"(?im)^\s*(?:text|passage|article|content|source|sentence|review|code|input)\s*:\s*"
)
_WORD_LIMIT_RE = re.compile(
    r"\b(?P<kind>exactly|at most|no more than|maximum(?: of)?|under|fewer than)\s+"
    r"(?P<count>\d+)\s+words?\b",
    re.IGNORECASE,
)
_SENTENCE_LIMIT_RE = re.compile(
    r"\b(?P<kind>exactly|at most|no more than|maximum(?: of)?|in)\s+"
    r"(?P<count>\d+)\s+sentences?\b",
    re.IGNORECASE,
)
_BULLET_COUNT_RE = re.compile(r"\b(?:exactly\s+)?(?P<count>\d+)\s+bullet(?:\s+points?)?\b", re.I)
_EXPLANATION_RE = re.compile(
    r"\b(?:explain|explanation|justify|justification|root cause|why (?:it|this|the|that)|"
    r"identify (?:the |all )?(?:bugs?|issues?|flaws?|errors?))\b",
    re.I,
)
_STEPS_RE = re.compile(r"\b(?:show (?:your |the )?(?:work|steps)|step[- ]by[- ]step|show reasoning)\b", re.I)
_LABEL_ONLY_RE = re.compile(
    r"\b(?:label only|only (?:the )?label|return (?:only )?(?:one of|the sentiment label))\b",
    re.I,
)
_CODE_ONLY_RE = re.compile(r"\b(?:code only|only (?:the )?(?:complete |corrected )?code|return only code)\b", re.I)
_JSON_RE = re.compile(r"\bjson\b|\bjson object\b", re.I)
_TABLE_RE = re.compile(r"\b(?:table|tabular)\b", re.I)
_LIST_RE = re.compile(r"\b(?:numbered list|bullet list|as a list|list of)\b", re.I)

_LANGUAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("typescript", re.compile(r"\b(?:typescript|\.tsx?)\b", re.I)),
    ("javascript", re.compile(r"\b(?:javascript|node\.js|\.jsx?)\b", re.I)),
    ("python", re.compile(r"\b(?:python|\.py)\b|```python", re.I)),
    ("c++", re.compile(r"(?<!\w)c\+\+(?!\w)|\bcpp\b|```cpp", re.I)),
    ("c#", re.compile(r"(?<!\w)c#(?!\w)|\bcsharp\b|```csharp", re.I)),
    ("java", re.compile(r"\bjava\b|```java", re.I)),
    ("go", re.compile(r"\b(?:golang|go)\b|```go", re.I)),
    ("rust", re.compile(r"\brust\b|```rust", re.I)),
)

_DEFAULT_SENTIMENT_LABELS = ("Positive", "Negative", "Neutral")
_SENTIMENT_LABEL_RE = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)
_ENTITY_ALIASES = {
    "per": "person",
    "person": "person",
    "people": "person",
    "org": "organization",
    "organization": "organization",
    "organisation": "organization",
    "loc": "location",
    "location": "location",
    "gpe": "location",
    "date": "date",
    "time": "time",
    "money": "money",
    "product": "product",
    "event": "event",
}


def split_prompt(prompt: str) -> PromptParts:
    """Split instructions from long source/code payload without interpreting content."""
    text = (prompt or "").strip()
    if not text:
        return PromptParts("", "")

    code_match = _CODE_BLOCK_RE.search(text)
    section_match = _SECTION_RE.search(text)
    split_at = len(text)
    payload = ""
    code_block = None

    if code_match:
        split_at = min(split_at, code_match.start())
        code_block = code_match.group("code").strip()
        payload = code_match.group(0)
    if section_match and section_match.start() < split_at:
        split_at = section_match.start()
        payload = text[section_match.end():].strip()

    instruction = text[:split_at].strip()
    if code_match:
        trailing_instruction = text[code_match.end():].strip()
        if trailing_instruction:
            instruction = f"{instruction}\n{trailing_instruction}".strip()
    if not instruction:
        instruction = text[: min(len(text), 800)].strip()
    if not payload and split_at < len(text):
        payload = text[split_at:].strip()
    return PromptParts(instruction=instruction, payload=payload, code_block=code_block)


def get_instruction_view(prompt: str, max_chars: int = 1600) -> str:
    """Return a bounded routing view that excludes source/code payload where possible."""
    return split_prompt(prompt).instruction[:max_chars]


def extract_word_limit(prompt: str) -> tuple[int | None, bool, bool]:
    match = re.search(
        r"\b(?P<kind>exactly|at most|no more than|maximum(?: of)?|under|fewer than)\s+"
        r"(?P<count>\d+)\s+words?(?:\s+(?P<per_bullet>per\s+bullet))?\b",
        get_instruction_view(prompt, 2400),
        re.IGNORECASE,
    )
    if not match:
        return None, False, False
    kind = match.group("kind").lower()
    per_bullet = bool(match.group("per_bullet"))
    return int(match.group("count")), kind == "exactly", per_bullet


def extract_sentence_limit(prompt: str) -> tuple[int | None, bool]:
    match = _SENTENCE_LIMIT_RE.search(get_instruction_view(prompt, 2400))
    if not match:
        one = re.search(r"\b(?:one|single) sentence\b", get_instruction_view(prompt, 2400), re.I)
        return (1, False) if one else (None, False)
    kind = match.group("kind").lower()
    return int(match.group("count")), kind == "exactly"


def extract_bullet_count(prompt: str) -> int | None:
    match = _BULLET_COUNT_RE.search(get_instruction_view(prompt, 2400))
    return int(match.group("count")) if match else None


def asks_for_explanation(prompt: str) -> bool:
    view = get_instruction_view(prompt, 2400)
    if re.search(r"\b(?:no|without) explanation\b|\bdo not (?:include|provide|give)\b.{0,30}\bexplanation\b", view, re.I):
        return False
    return bool(_EXPLANATION_RE.search(view))


def asks_for_steps(prompt: str) -> bool:
    view = get_instruction_view(prompt, 2400)
    if re.search(r"\b(?:no|without) (?:steps|reasoning)\b|\bdo not (?:include|show|provide)\b.{0,30}\b(?:steps|reasoning)\b", view, re.I):
        return False
    return bool(_STEPS_RE.search(view))


def detect_output_format(prompt: str) -> str:
    view = get_instruction_view(prompt, 2400)
    if _JSON_RE.search(view):
        return "json"
    if _TABLE_RE.search(view):
        return "table"
    if _BULLET_COUNT_RE.search(view) or _LIST_RE.search(view):
        return "list"
    if _CODE_ONLY_RE.search(view):
        return "code"
    return "text"


def detect_code_language(prompt: str) -> str | None:
    view = get_instruction_view(prompt, 2400)
    code_match = _CODE_BLOCK_RE.search(prompt or "")
    if code_match and code_match.group("lang"):
        view = f"{view} {code_match.group('lang')}"
    for language, pattern in _LANGUAGE_PATTERNS:
        if pattern.search(view):
            return language
    return None


def extract_code_block(prompt: str) -> str | None:
    match = _CODE_BLOCK_RE.search(prompt or "")
    return match.group("code").strip() if match else None


def extract_allowed_sentiment_labels(prompt: str) -> tuple[str, ...]:
    view = get_instruction_view(prompt, 2400)
    labels: list[str] = []
    for match in _SENTIMENT_LABEL_RE.finditer(view):
        label = match.group(1).title()
        if label not in labels:
            labels.append(label)
    return tuple(labels) if labels else _DEFAULT_SENTIMENT_LABELS


def extract_requested_entity_types(prompt: str) -> tuple[str, ...]:
    view = get_instruction_view(prompt, 2400).lower()
    found: list[str] = []
    for token, normalized in _ENTITY_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}s?\b", view) and normalized not in found:
            found.append(normalized)
    return tuple(found)


def parse_constraints(prompt: str) -> OutputConstraints:
    word_limit, exact_words, word_limit_per_bullet = extract_word_limit(prompt)
    sentence_limit, exact_sentences = extract_sentence_limit(prompt)
    view = get_instruction_view(prompt, 2400)
    return OutputConstraints(
        word_limit=word_limit,
        exact_word_count=exact_words,
        word_limit_per_bullet=word_limit_per_bullet,
        sentence_limit=sentence_limit,
        exact_sentence_count=exact_sentences,
        bullet_count=extract_bullet_count(prompt),
        label_only=bool(_LABEL_ONLY_RE.search(view)),
        explanation_requested=asks_for_explanation(prompt),
        steps_requested=asks_for_steps(prompt),
        output_format=detect_output_format(prompt),
        code_only=bool(_CODE_ONLY_RE.search(view)),
        programming_language=detect_code_language(prompt),
        allowed_sentiment_labels=extract_allowed_sentiment_labels(prompt),
        requested_entity_types=extract_requested_entity_types(prompt),
    )
