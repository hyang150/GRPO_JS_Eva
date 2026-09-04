"""GSM8K prompts for a chat-tuned policy."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, config as ds_config, load_dataset

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


def _cached_arrow(split: str) -> Path | None:
    # datasets resolves HF_DATASETS_CACHE / HF_HOME itself; a hard-coded
    # ~/.cache misses machines that relocate the cache (HF_HOME=/root/autodl-tmp/hf).
    cache_root = Path(ds_config.HF_DATASETS_CACHE) / "openai___gsm8k/main/0.0.0"
    paths = sorted(cache_root.glob(f"*/gsm8k-{split}.arrow"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _load_dataset_cache_first(split: str):
    """Load GSM8K without touching the network when the Arrow cache is present."""
    cached = _cached_arrow(split)
    if cached is not None:
        print(f"GSM8K {split}: using local Arrow cache {cached}", flush=True)
        return Dataset.from_file(str(cached))

    print(f"GSM8K {split}: local Arrow cache missing; falling back to Hugging Face Hub.",
          flush=True)
    return load_dataset("openai/gsm8k", "main", split=split)


def load_gsm8k(tokenizer, split: str = "train", limit: int | None = None):
    """Returns a list of {"prompt": str, "gold": float, "question": str}."""
    ds = _load_dataset_cache_first(split)
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
