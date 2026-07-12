"""Dynamic output optimizer public API."""

from .optimizer import build_optimization
from .execution import execute_task, solve_tasks, structured_log
from .bundling import (
    BundleCandidate,
    BundleDecision,
    BundleExecutionError,
    TaskBundle,
    assess_bundle_eligibility,
    create_bundles,
    execute_bundle,
)
from .parser import (
    asks_for_explanation,
    asks_for_steps,
    detect_code_language,
    detect_output_format,
    extract_allowed_sentiment_labels,
    extract_bullet_count,
    extract_code_block,
    extract_requested_entity_types,
    extract_sentence_limit,
    extract_word_limit,
    get_instruction_view,
    parse_constraints,
    split_prompt,
)
from .postprocessors import postprocess_output
from .router import route_task
from .token_budget import estimate_tokens
from .types import Optimization, OutputConstraints, RouteDecision, TokenPolicy, ValidationResult
from .validators import validate_output

__all__ = [
    "BundleCandidate",
    "BundleDecision",
    "BundleExecutionError",
    "Optimization",
    "OutputConstraints",
    "RouteDecision",
    "TokenPolicy",
    "TaskBundle",
    "ValidationResult",
    "asks_for_explanation",
    "asks_for_steps",
    "assess_bundle_eligibility",
    "build_optimization",
    "create_bundles",
    "detect_code_language",
    "detect_output_format",
    "estimate_tokens",
    "execute_bundle",
    "execute_task",
    "extract_allowed_sentiment_labels",
    "extract_bullet_count",
    "extract_code_block",
    "extract_requested_entity_types",
    "extract_sentence_limit",
    "extract_word_limit",
    "get_instruction_view",
    "parse_constraints",
    "postprocess_output",
    "route_task",
    "solve_tasks",
    "split_prompt",
    "structured_log",
    "validate_output",
]
