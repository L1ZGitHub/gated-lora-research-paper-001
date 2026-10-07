"""Scoring of free (greedy) generations, for tasks where teacher-forced metrics mislead.

- GSM8K: final-number accuracy. Reference = the number after "####"; prediction = the number
  after the last "####" if the model wrote one, else the last number in the text.
- XSum: ROUGE-L F1 on lower-cased word tokens (LCS, no stemming, no extra dependency; close to
  but not identical with `rouge_score`'s rougeL, so report it as such).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Max new tokens per task (GSM8K references: p99 ~ 200 tokens; XSum summaries ~ 25 words)
GEN_MAX_NEW_TOKENS: Dict[str, int] = {"gsm8k": 256, "xsum": 64}

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"\w+")


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def gsm8k_number(text: str) -> Optional[float]:
    if "####" in text:
        nums = _NUM.findall(text.rsplit("####", 1)[1])
        if nums:
            return _to_float(nums[0])
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def gsm8k_correct(prediction: str, reference: str) -> bool:
    ref, pred = gsm8k_number(reference), gsm8k_number(prediction)
    return ref is not None and pred is not None and abs(ref - pred) < 1e-6


def rouge_l_f1(prediction: str, reference: str) -> float:
    p = _WORD.findall(prediction.lower())
    r = _WORD.findall(reference.lower())
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0] * (len(r) + 1)
        for j, b in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def score(task: str, predictions: List[str], references: List[str]) -> Dict[str, object]:
    """Per-example scores + mean for one task."""
    if task == "gsm8k":
        per = [float(gsm8k_correct(p, r)) for p, r in zip(predictions, references)]
        name = "final_number_accuracy"
    elif task == "xsum":
        per = [rouge_l_f1(p, r) for p, r in zip(predictions, references)]
        name = "rouge_l_f1"
    else:
        raise ValueError(f"No generation metric for task {task!r}")
    return {"metric": name, "mean": sum(per) / len(per) if per else float("nan"),
            "num_examples": len(per), "per_example": per}
