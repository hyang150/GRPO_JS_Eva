"""GSM8K test-set evaluation.

Reports pass@1 with a binomial standard error, because at 0.5B scale the
interval is wide enough to change what you are allowed to conclude: over 60
problems at temperature 1.0 the same prompt scored 28.3% and 18.3%.
"""

from __future__ import annotations

import math

import torch

from .data import load_gsm8k
from .rewards import correctness_reward


@torch.no_grad()
def evaluate(model, tokenizer, n_problems: int = 200, batch_size: int = 32,
             max_new_tokens: int = 512, temperature: float = 0.0,
             split: str = "test", progress: bool = False) -> dict:
    """pass@1 on GSM8K. ``temperature=0`` is greedy."""
    was_training = model.training
    model.eval()
    ckpt = getattr(model, "is_gradient_checkpointing", False)
    if ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    data = load_gsm8k(tokenizer, split, limit=n_problems)
    texts, golds = [], [d["gold"] for d in data]

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        enc = tokenizer([d["prompt"] for d in chunk], return_tensors="pt",
                        padding=True, padding_side="left").to(model.device)
        out = model.generate(
            **enc, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0, top_k=0 if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
        texts += tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:],
                                        skip_special_tokens=True)
        if progress:
            print(f"  eval {min(i + batch_size, len(data))}/{len(data)}", flush=True)

    if ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train(was_training)

    rew = correctness_reward(texts, golds)
    n = len(rew)
    acc = sum(rew) / n
    lens = [len(tokenizer(t).input_ids) for t in texts]
    return {
        "accuracy": acc,
        "stderr": math.sqrt(acc * (1 - acc) / n),
        "n": n,
        "mean_len": sum(lens) / n,
        "truncated_frac": sum(l >= max_new_tokens - 2 for l in lens) / n,
        # how often the policy put the answer where we asked, rather than us
        # recovering it from the last number in the text
        "format_frac": sum(("\\boxed{" in t) or ("####" in t) for t in texts) / n,
        "temperature": temperature,
        # Every eval scores the *same* problems in the same order, so two
        # checkpoints can be compared with a paired test (McNemar) rather than
        # by overlapping binomial intervals -- much more power at this n.
        "per_problem": [int(x) for x in rew],
    }


def mcnemar(before: list[int], after: list[int]) -> dict:
    """Paired comparison of two evals over the same problems.

    Only the disagreements carry information: b = fixed, c = broken.  Uses the
    exact binomial two-sided test, which is what you want at these counts.
    """
    from math import comb

    if len(before) != len(after):
        raise ValueError("evals must cover the same problems")
    b = sum(x == 0 and y == 1 for x, y in zip(before, after))   # newly correct
    c = sum(x == 1 and y == 0 for x, y in zip(before, after))   # newly wrong
    n = b + c
    if n == 0:
        return {"fixed": 0, "broken": 0, "p_value": 1.0, "delta": 0.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"fixed": b, "broken": c, "n_discordant": n,
            "p_value": min(1.0, 2 * tail),
            "delta": (b - c) / len(before)}
