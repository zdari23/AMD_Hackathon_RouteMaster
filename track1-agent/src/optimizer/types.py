"""Shared immutable models for the dynamic output optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    """Weighted router result with enough detail for safe observability."""

    task_type: str
    confidence: float
    scores: dict[str, int]
    matched_signals: dict[str, list[str]]


@dataclass(frozen=True)
class PromptParts:
    """Instruction-focused and payload-focused views of a user prompt."""

    instruction: str
    payload: str
    code_block: str | None = None


@dataclass(frozen=True)
class OutputConstraints:
    """Explicit output requirements parsed without changing user intent."""

    word_limit: int | None = None
    exact_word_count: bool = False
    word_limit_per_bullet: bool = False
    sentence_limit: int | None = None
    exact_sentence_count: bool = False
    bullet_count: int | None = None
    label_only: bool = False
    explanation_requested: bool = False
    steps_requested: bool = False
    output_format: str = "text"
    code_only: bool = False
    programming_language: str | None = None
    allowed_sentiment_labels: tuple[str, ...] = ()
    requested_entity_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenPolicy:
    """Completion budget selected for a prompt."""

    max_tokens: int
    min_tokens: int
    reason: str
    aggressive: bool
    category_max: int


@dataclass
class ValidationResult:
    """Deterministic validation and local repair result."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    repaired_output: str | None = None


@dataclass(frozen=True)
class Optimization:
    """Complete optimizer output consumed by API callers."""

    task_type: str
    confidence: float
    system_prompt: str
    max_tokens: int
    validator_name: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "confidence": self.confidence,
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "validator_name": self.validator_name,
            "metadata": self.metadata,
        }
