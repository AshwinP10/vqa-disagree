"""
VQA answer normalization following the official VQAv2 evaluation protocol.
"""
from __future__ import annotations
import re

_PUNCT = re.compile(r"[;/\[\]\"{}()\=+\\`!?,]")
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP  = re.compile(r"(\d)(,)(\d)")
_ARTICLES = {"a", "an", "the"}
_DIGIT_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = _COMMA_STRIP.sub(r"\1\3", s)
    s = _PUNCT.sub(" ", s)
    s = _PERIOD_STRIP.sub("", s)
    tokens = []
    for w in s.split():
        w = _DIGIT_MAP.get(w, w)
        if w not in _ARTICLES:
            tokens.append(w)
    return " ".join(tokens).strip()


def vqa_soft_match(pred: str, gt) -> float:
    """
    VQA soft accuracy.
    gt may be a single string, int/float (numeric answer), or a list of human answers.
    """
    pred_norm = normalize_answer(pred)
    if isinstance(gt, (int, float)):
        gt = str(int(gt)) if float(gt) == int(gt) else str(gt)
    if isinstance(gt, str):
        return float(normalize_answer(gt) == pred_norm)
    matches = sum(1 for g in gt if normalize_answer(str(g)) == pred_norm)
    return min(matches / 3, 1.0)


def exact_match(pred: str, gt: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gt))
