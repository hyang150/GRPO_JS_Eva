"""Pick the prompt before training anything.

The reward is only as good as the fraction of correct answers we can parse.
A prompt that makes the model ramble past the token budget turns correct
reasoning into reward 0, which is indistinguishable from a wrong answer --
noise the shrinkage estimator would then be asked to explain.
"""

import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_eva.rewards import extract_answer, extract_gold
from datasets import load_dataset

_CONCISE = ("Solve the math problem briefly, then end your reply with the final "
            "numeric answer inside \\boxed{}. Do not write anything after it.")

_SHOT_Q = ("Natalia sold clips to 48 friends in April, and half as many in May. "
           "How many clips did she sell altogether?")
_SHOT_A = ("April: 48 clips.\n"
           "May: 48 / 2 = 24 clips.\n"
           "Total: 48 + 24 = 72.\n"
           "\\boxed{72}")

# each entry: (system prompt, few-shot message pairs)
PROMPTS = {
    "concise_boxed":   (_CONCISE, []),
    "oneshot_boxed":   (_CONCISE, [(_SHOT_Q, _SHOT_A)]),
    "oneshot_capped":  (_CONCISE + " Use at most 5 short lines.",
                        [(_SHOT_Q, _SHOT_A)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).cuda().eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n_problems))
    golds = [extract_gold(a) for a in ds["answer"]]

    print(f"{args.n_problems} problems, temperature={args.temperature}, "
          f"max_new_tokens={args.max_new_tokens}\n")
    print(f"{'prompt':<16}{'acc':>7}{'parsed':>9}{'len':>7}{'truncated':>11}")

    for name, (system, shots) in PROMPTS.items():
        prefix = [{"role": "system", "content": system}]
        for q, a in shots:
            prefix += [{"role": "user", "content": q},
                       {"role": "assistant", "content": a}]
        texts = []
        for i in range(0, len(ds), 16):
            msgs = [prefix + [{"role": "user", "content": q}]
                    for q in ds["question"][i:i + 16]]
            enc = tok([tok.apply_chat_template(m, tokenize=False,
                                              add_generation_prompt=True) for m in msgs],
                      return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=args.temperature > 0,
                                     temperature=args.temperature or None,
                                     top_p=1.0, top_k=0,
                                     # explicit: the checkpoint's generation_config
                                     # says 1.1, which the README's prompt numbers
                                     # (2026-09-03) were measured with
                                     repetition_penalty=1.0,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                      skip_special_tokens=True)

        lens = [len(tok(t).input_ids) for t in texts]
        trunc = sum(l >= args.max_new_tokens - 2 for l in lens) / len(lens)
        # "parsed" = the model put the answer somewhere we can find on purpose
        parsed = sum(("\\boxed{" in t) or ("####" in t) for t in texts) / len(texts)
        acc = sum(extract_answer(t) is not None and g is not None
                  and abs(extract_answer(t) - g) < 1e-4
                  for t, g in zip(texts, golds)) / len(texts)
        print(f"{name:<16}{acc:>7.1%}{parsed:>9.1%}{sum(lens)/len(lens):>7.0f}{trunc:>11.1%}")


if __name__ == "__main__":
    main()
