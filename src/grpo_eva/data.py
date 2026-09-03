"""GSM8K prompts for a chat-tuned policy."""

from __future__ import annotations

from datasets import load_dataset

from .rewards import extract_gold

# Chosen by experiments/select_prompt.py over 4 alternatives.  It beat a
# verbose "reason step by step" prompt and a "#### <number>" variant on both
# accuracy and truncation rate; one-shot exemplars did not help.
# Format compliance starts around 55% -- that headroom is fine, GRPO rewards
# the parseable format and the policy learns it.
SYSTEM = (
    "Solve the math problem briefly, then end your reply with the final "
    "numeric answer inside \\boxed{}. Do not write anything after it."
)


def load_gsm8k(tokenizer, split: str = "train", limit: int | None = None):
    """Returns a list of {"prompt": str, "gold": float, "question": str}."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out = []
    for ex in ds:
        gold = extract_gold(ex["answer"])
        if gold is None:                  # malformed reference: unusable
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": ex["question"]}],
            tokenize=False, add_generation_prompt=True,
        )
        out.append({"prompt": prompt, "gold": gold, "question": ex["question"]})
    return out
