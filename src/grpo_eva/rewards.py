"""Rule-based rewards for GSM8K.

GSM8K ships a ground-truth answer, so this is RLVR: the reward is a verifier,
not a learned reward model.  A learned RM here would only add noise and give
the policy something to hack.

Extraction is deliberately permissive.  A 0.5B model does not reliably obey a
single output format, and every correct answer we fail to parse is a false
negative that shows up as reward noise -- which is exactly the quantity the
whole project is trying to measure.  Being strict about format would bias the
experiment.
"""

from __future__ import annotations

import re

# ordered by how much the match tells us the model *meant* it as the answer
_PATTERNS = (
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"<answer>\s*(.+?)\s*</answer>", re.S),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"(?:answer|result)\s*(?:is|:|=)\s*\$?(-?[\d,]*\.?\d+)", re.I),
)
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _to_float(s: str) -> float | None:
    m = _NUMBER.findall(s.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Best-effort numeric answer from a completion."""
    for pat in _PATTERNS:
        found = pat.findall(text)
        if found:
            v = _to_float(found[-1])
            if v is not None:
                return v
    nums = _NUMBER.findall(text)          # last resort: the final number stated
    return _to_float(nums[-1]) if nums else None


def extract_gold(answer_field: str) -> float | None:
    """GSM8K puts the reference answer after '####'."""
    tail = answer_field.rsplit("####", 1)
    return _to_float(tail[-1]) if len(tail) == 2 else None


def correctness_reward(completions: list[str], golds: list[float | None],
                       tol: float = 1e-4) -> list[float]:
    """1.0 if the extracted answer matches the reference, else 0.0."""
    out = []
    for c, g in zip(completions, golds):
        p = extract_answer(c)
        out.append(1.0 if (p is not None and g is not None and abs(p - g) <= tol)
                   else 0.0)
    return out


def format_reward(completions: list[str]) -> list[float]:
    """Small bonus for stating the answer in a parseable place.

    Kept separate and off by default: any auxiliary reward changes the reward
    distribution, and therefore the group variance the shrinkage estimator is
    reacting to.  Turning it on mid-experiment would confound the A/B.
    """
    return [1.0 if any(p.search(c) for p in _PATTERNS[:3]) else 0.0
            for c in completions]
