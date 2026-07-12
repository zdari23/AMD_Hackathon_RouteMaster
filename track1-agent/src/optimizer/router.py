"""Fast deterministic weighted router focused on user instruction intent."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .parser import get_instruction_view, split_prompt
from .types import RouteDecision


CATEGORIES = (
    "knowledge_qa",
    "math_solving",
    "sentiment_analysis",
    "summarization",
    "entity_extraction",
    "bug_fixing",
    "logical_puzzles",
    "code_authoring",
)
FALLBACK = "fallback"
MIN_ROUTE_SCORE = 6
MIN_CONFIDENCE = 0.46


@dataclass(frozen=True)
class _Signal:
    name: str
    weight: int
    pattern: re.Pattern[str]


def _signal(name: str, weight: int, pattern: str) -> _Signal:
    return _Signal(name, weight, re.compile(pattern, re.IGNORECASE))


_SIGNALS: dict[str, tuple[_Signal, ...]] = {
    "knowledge_qa": (
        _signal("explain_intent", 9, r"\b(?:explain|define|describe)\b"),
        _signal("what_is_intent", 8, r"\bwhat (?:is|are|was|were)\b"),
        _signal("why_how_intent", 7, r"\b(?:why|how (?:does|do|did|can|is|are))\b"),
        _signal("who_where_when", 7, r"\b(?:who|where|when) (?:is|are|was|were|did)\b"),
        _signal("concept_topic", 2, r"\b(?:concept|meaning|difference between|definition)\b"),
    ),
    "math_solving": (
        _signal("calculate_intent", 12, r"\b(?:calculate|compute|evaluate)\b"),
        _signal("numeric_solve_intent", 10, r"\bsolve\b.{0,80}\b(?:equation|expression|percentage|probability|average|total)\b"),
        _signal("quantity_question", 8, r"\bhow (?:much|many|long|far)\b.{0,120}\d"),
        _signal("trailing_quantity_question", 8, r"\d.{0,160}\bhow (?:much|many|long|far)\b"),
        _signal("math_operation", 5, r"\b(?:sum|product|average|weighted|grade|ratio|percent(?:age)?|probability|revenue|distance|rate|area|perimeter|discount|tax|total cost|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|growth)\b"),
        _signal("quantitative_units", 3, r"\b(?:hours?|minutes?|kilograms?|grams?|meters?|kilometers?|servings?|parts?|items?|dollars?|liters?)\b"),
        _signal("quantified_state_change", 4, r"\b(?:begins?|starts?|initially|currently)\b.{0,80}\d"),
        _signal("numeric_structure", 2, r"\d+(?:\.\d+)?\s*(?:%|[+*/=])"),
    ),
    "sentiment_analysis": (
        _signal("classify_sentiment", 13, r"\b(?:classify|label|determine|analy[sz]e)\b.{0,80}\bsentiment\b"),
        _signal("perform_sentiment", 13, r"\bperform\b.{0,60}\bsentiment analysis\b"),
        _signal("sentiment_of", 10, r"\bsentiment (?:of|toward|towards|expressed)\b"),
        _signal("sentiment_topic", 2, r"\bsentiment analysis\b|\bpositive\b.{0,40}\bnegative\b"),
    ),
    "summarization": (
        _signal("summarize_intent", 13, r"\b(?:summari[sz]e|condense|abridge)\b"),
        _signal("summary_intent", 10, r"\b(?:provide|write|create|produce|give)\b.{0,50}\bsummary\b"),
        _signal("headline_intent", 11, r"\b(?:write|create|produce|provide)\b.{0,50}\bheadline\b"),
        _signal("action_items", 8, r"\bextract\b.{0,60}\b(?:action items|decisions|deadlines|key points)\b"),
        _signal("summary_topic", 2, r"\b(?:summary|main point|key details|tl;?dr)\b"),
    ),
    "entity_extraction": (
        _signal("extract_entities", 14, r"\b(?:extract|identify|label|list)\b.{0,90}\b(?:named )?entit(?:y|ies)\b"),
        _signal("return_entities", 12, r"\b(?:return|output|provide)\b.{0,60}\bentit(?:y|ies)\b"),
        _signal("ner_action", 12, r"\bperform\b.{0,50}\b(?:ner|named entity recognition)\b"),
        _signal("extract_entity_types", 10, r"\bextract\b.{0,100}\b(?:persons?|organizations?|organisations?|locations?|dates?)\b"),
        _signal("ner_topic", 2, r"\b(?:ner|named entity recognition|entity extraction)\b"),
    ),
    "bug_fixing": (
        _signal("fix_code", 14, r"\b(?:fix|correct|repair|debug)\b.{0,80}\b(?:code|function|method|class|implementation|snippet|bug)\b"),
        _signal("code_fails", 18, r"\b(?:explain|identify|find)\b.{0,50}\b(?:why|what)\b.{0,50}\b(?:code|function|implementation)\b.{0,40}\b(?:fails?|wrong|buggy|broken)\b"),
        _signal("bug_intent", 11, r"\b(?:identify|find|explain)\b.{0,70}\b(?:bugs?|race conditions?|flaws?|errors?)\b"),
        _signal("corrected_implementation", 9, r"\b(?:corrected|fixed)\b.{0,40}\b(?:code|implementation|function|class)\b"),
        _signal("bug_topic", 2, r"\b(?:bug|buggy|debugging|race condition|stack trace)\b"),
    ),
    "logical_puzzles": (
        _signal("solve_logic", 14, r"\b(?:solve|determine)\b.{0,80}\b(?:logic|logical|puzzle|riddle|assignment|schedule|configuration)\b"),
        _signal("puzzle_conclusion", 11, r"\b(?:determine|find)\b.{0,100}\b(?:who|which|unique|complete assignment|complete schedule)\b"),
        _signal("relational_assignment", 9, r"\beach\b.{0,60}\b(?:different|distinct)\b|\bone per\b|\b(?:sits?|scheduled|arrives?|delivered|stacked)\b.{0,80}\b(?:slot|seat|position|day|row|bottom|top)\b|\b(?:left of|right of|above|below|bottom to top|top to bottom|arrives? before|arrives? after)\b"),
        _signal("constraint_structure", 3, r"\b(?:all constraints|each (?:person|item)|exactly one|distinct)\b"),
        _signal("logic_topic", 2, r"\b(?:logic puzzle|deductive reasoning|riddle|knights? and knaves?)\b"),
    ),
    "code_authoring": (
        _signal("write_code", 15, r"\b(?:write|implement|create|code|develop)\b.{0,100}\b(?:function|method|class|script|program|algorithm|component)\b"),
        _signal("return_code", 11, r"\breturn\b.{0,50}\b(?:code|implementation|function|class)\b"),
        _signal("signature_structure", 6, r"\b(?:function|method|class)\s+[`'\"]?[A-Za-z_]\w*"),
        _signal("code_language", 2, r"\b(?:python|javascript|typescript|java|golang|rust)\b|(?<!\w)c(?:\+\+|#)(?!\w)"),
    ),
}


def _confidence(top_score: int, second_score: int) -> float:
    if top_score <= 0:
        return 0.0
    margin = max(0, top_score - second_score)
    strength = min(1.0, top_score / 16.0)
    separation = margin / max(top_score, 1)
    return round(min(0.99, 0.35 + 0.35 * strength + 0.30 * separation), 3)


def route_task(prompt: str) -> RouteDecision:
    """Score every category and return a conservative routing decision."""
    if not prompt or not prompt.strip():
        return RouteDecision(FALLBACK, 0.0, {category: 0 for category in CATEGORIES}, {})

    view = get_instruction_view(prompt)
    parts = split_prompt(prompt)
    scores = {category: 0 for category in CATEGORIES}
    matched: dict[str, list[str]] = {}

    for category, signals in _SIGNALS.items():
        names: list[str] = []
        for signal in signals:
            if signal.pattern.search(view):
                scores[category] += signal.weight
                names.append(signal.name)
        if names:
            matched[category] = names

    # Structural signals are deliberately weak; action intent remains dominant.
    if parts.code_block:
        if scores["bug_fixing"]:
            scores["bug_fixing"] += 3
            matched.setdefault("bug_fixing", []).append("code_block")
        elif scores["code_authoring"]:
            scores["code_authoring"] += 2
            matched.setdefault("code_authoring", []).append("code_block")

    number_count = len(re.findall(r"\d+(?:\.\d+)?", view))
    strong_non_math_action = any(
        scores[category] >= 10
        for category in CATEGORIES
        if category not in {"math_solving", "knowledge_qa"}
    )
    if number_count >= 2 and scores["math_solving"] >= 3 and not strong_non_math_action:
        scores["math_solving"] += 6
        matched.setdefault("math_solving", []).append("multiple_quantities")

    constraint_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", view))
    constraint_structure = re.search(
        r"\b(?:each\b.{0,60}\b(?:different|distinct)|one per|slots?|seats?|positions?|stacked|bottom to top|top to bottom|left of|right of|above|below|immediately before|immediately after)\b",
        view,
        re.I,
    )
    if constraint_lines >= 2 and constraint_structure:
        scores["logical_puzzles"] += 11
        matched.setdefault("logical_puzzles", []).append("constraint_list_structure")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], CATEGORIES.index(item[0])))
    top_type, top_score = ranked[0]
    second_score = ranked[1][1]
    confidence = _confidence(top_score, second_score)
    if top_score < MIN_ROUTE_SCORE or confidence < MIN_CONFIDENCE:
        top_type = FALLBACK

    return RouteDecision(top_type, confidence, scores, matched)
