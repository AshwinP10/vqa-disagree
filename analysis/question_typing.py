"""
Rule-based question type classification for VQA questions.
Types: counting / spatial / attribute / relational / existence / chart-reading / other
"""
from __future__ import annotations
import re

_COUNTING   = re.compile(r"\bhow\s+many\b", re.I)
_EXISTENCE  = re.compile(r"\b(is\s+there|are\s+there|does\s+the\s+image|can\s+you\s+see)\b", re.I)
_SPATIAL    = re.compile(r"\b(left|right|above|below|behind|in\s+front|next\s+to|between|beside|top|bottom|position|where)\b", re.I)
_ATTRIBUTE  = re.compile(r"\b(what\s+color|what\s+colour|what\s+shape|what\s+size|how\s+(big|large|small|tall|wide|long|old)|material|texture|what\s+is\s+the\s+(color|colour|shape|size|type|kind))\b", re.I)
_CHART      = re.compile(r"\b(chart|graph|plot|bar|axis|legend|table|row|column|figure|percent|percentage|trend)\b", re.I)
_RELATIONAL = re.compile(r"\b(taller|shorter|larger|smaller|more|fewer|heavier|lighter|compared|relation|between|than)\b", re.I)


def classify_question(question: str) -> str:
    q = question.strip()
    if _COUNTING.search(q):
        return "counting"
    if _EXISTENCE.search(q):
        return "existence"
    if _CHART.search(q):
        return "chart-reading"
    if _SPATIAL.search(q):
        return "spatial"
    if _ATTRIBUTE.search(q):
        return "attribute"
    if _RELATIONAL.search(q):
        return "relational"
    return "other"


def classify_batch(questions: list[str]) -> list[str]:
    return [classify_question(q) for q in questions]


QUESTION_TYPES = ["counting", "spatial", "attribute", "relational", "existence", "chart-reading", "other"]
