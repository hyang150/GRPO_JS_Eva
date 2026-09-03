#!/usr/bin/env python3
"""grpo-eva -- shrinkage baselines for GRPO on GSM8K.

    python main.py smoke                                  # GPU gate
    python main.py synthetic --n 2 4 8 16                 # phase 1
    python main.py train --baseline js_pooled --steps 200 # phase 2
    python main.py gradvar --batches 200                  # phase 3
    python main.py eval --n-problems 200
    python main.py sweep --baselines vanilla js_fixed js_pooled global
    python main.py figures

`train` and `sweep` accept the full GRPO knob set: --beta (KL to a frozen
reference), --num-iterations (mu), --scale (advantage std normalisation),
--epsilon (PPO clip), --loss-norm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------- shared
def _step_progress(args, tag: str):
    if not getattr(args, "progress", True):
        return range(args.steps)
    try:
        from tqdm.auto import trange
    except Exception:
        print("tqdm is not installed; falling back to plain training logs.", flush=True)
        return range(args.steps)
    return trange(args.steps, desc=tag, unit="step", dynamic_ncols=True)


def _progress_write(progress, text: str):
    writer = getattr(progress, "write", None)
    if writer is not None:
        writer(text)
    else:
        print(text, flush=True)


def _stage(message: str, progress=None):
    _progress_write(progress, f"[{time.strftime('%H:%M:%S')}] {message}")


def add_model_args(p):
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)


def add_grpo_args(p):
    from grpo_eva.baselines import BASELINE_REGISTRY
    p.add_argument("--baseline", default="vanilla", choices=sorted(BASELINE_REGISTRY))
    p.add_argument("--k", type=int, default=8, help="prompts per step")
    p.add_argument("--n", type=int, default=8, help="generations per prompt")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient; >0 loads a frozen reference model")
    p.add_argument("--num-iterations", type=int, default=1, help="mu")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO clip")
    p.add_argument("--precision", default="bf16", choices=["bf16", "mixed"],
                   help="bf16 keeps params, grads and AdamW moments in bf16 "
                        "(fits 16 GB, but at lr=1e-6 only ~2%% of weights move "
                        "per step); mixed holds fp32 master weights and casts "
                        "the forward pass -- ~2x the memory")
    p.add_argument("--track-update-precision", action="store_true",
                   help="log the fraction of sampled weights each step moves")
    p.add_argument("--track-grad-var", action="store_true",
                   help="log the per-step gradient variance of arXiv:2511.03710 "
                        "eq. 17-18 (noise/signal across micro-batches); ~1.5 s/step")
    p.add_argument("--scale", default="none", choices=["none", "group", "batch"])
    p.add_argument("--loss-norm", default="constant",
                   choices=["constant", "per_seq", "per_token"])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="rollout sampler penalty; 1.0 = sample the policy itself. "
                        "Runs before 2026-09-04 inherited 1.1 from the checkpoint's "
                        "generation_config")


def build(args, need_ref: bool = False):
    """Loads tokenizer, policy and (optionally) a frozen reference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    _stage(f"seed={args.seed}: loading tokenizer {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # "mixed" keeps fp32 master weights and casts only the forward pass
    precision = getattr(args, "precision", "bf16")
    dtype = torch.float32 if precision == "mixed" else torch.bfloat16
    dtype_label = "fp32 params + bf16 autocast" if precision == "mixed" else "bf16 params"

    def load(role: str):
        _stage(f"seed={args.seed}: loading {role} model on cuda ({dtype_label})")
        loaded = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="sdpa").cuda()
        _stage(f"seed={args.seed}: {role} model ready")
        return loaded

    model = load("policy")
    model.gradient_checkpointing_enable()
    _stage(f"seed={args.seed}: policy gradient checkpointing enabled")
    ref = load("reference") if need_ref else None
    return tok, model, ref


# ---------------------------------------------------------------- subcommands
def cmd_smoke(args):
    return subprocess.call([sys.executable, str(ROOT / "experiments/smoke_gpu.py")])


def cmd_synthetic(args):
    cmd = [sys.executable, str(ROOT / "experiments/synthetic_mse.py"),
           "--k", *map(str, args.k), "--n", *map(str, args.n),
           "--trials", str(args.trials)]
    rc = subprocess.call(cmd)
    return rc or subprocess.call([sys.executable, str(ROOT / "experiments/plot_synthetic.py")])


def cmd_gradvar(args):
    return subprocess.call([sys.executable, str(ROOT / "experiments/grad_variance.py"),
                            "--batches", str(args.batches), "--k", str(args.k),
                            "--n", str(args.n), "--seed", str(args.seed),
                            "--model", args.model])


def cmd_prompt(args):
    return subprocess.call([sys.executable, str(ROOT / "experiments/select_prompt.py"),
                            "--n-problems", str(args.n_problems), "--model", args.model])


def cmd_figures(args):
    rc = 0
    for script, need in (("plot_synthetic.py", "results/synthetic_mse.json"),
                         ("plot_grad_variance.py", "results/grad_variance_k8n8_long.json"),
                         ("plot_training.py", "results")):
        if (ROOT / need).exists():
            rc |= subprocess.call([sys.executable, str(ROOT / "experiments" / script)])
        else:
            print(f"skip {script}: {need} not found")
    return rc


def cmd_eval(args):
    from grpo_eva.evaluate import evaluate
    tok, model, _ = build(args)
    r = evaluate(model, tok, n_problems=args.n_problems,
                 max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature, progress=True)
    print(f"\naccuracy {r['accuracy']:.1%} +/- {r['stderr']:.1%}  (n={r['n']}, "
          f"T={r['temperature']})\nmean_len {r['mean_len']:.0f}  "
          f"truncated {r['truncated_frac']:.1%}  format {r['format_frac']:.1%}")
    return 0


def _train_one(args, tok, model, ref, tag):
    """One training run; returns the path of its jsonl log."""
    from grpo_eva.data import load_gsm8k
    from grpo_eva.evaluate import evaluate
    from grpo_eva.grpo import GRPO, GRPOConfig

    cfg = GRPOConfig(baseline=args.baseline, k_prompts=args.k, n_generations=args.n,
                     lr=args.lr, beta=args.beta, num_iterations=args.num_iterations,
                     epsilon=args.epsilon, scale=args.scale, loss_norm=args.loss_norm,
                     precision=args.precision,
                     track_update_precision=args.track_update_precision,
                     track_grad_var=args.track_grad_var,
                     max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                     repetition_penalty=args.repetition_penalty,
                     micro_batch=args.micro_batch, gen_micro_batch=args.k * args.n,
                     seed=args.seed)
    import torch
    _stage(f"{tag}: reset CUDA peak memory stats")
    torch.cuda.reset_peak_memory_stats()      # else arm 2 reports arm 1's peak
    _stage(f"{tag}: initializing GRPO trainer")
    trainer = GRPO(model, tok, cfg, ref_model=ref)
    _stage(f"{tag}: loading GSM8K train split")
    data = load_gsm8k(tok, "train")
    _stage(f"{tag}: loaded {len(data)} train prompts")
    rng = random.Random(args.seed)

    out = ROOT / f"results/train_{tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    _stage(f"{tag}: writing log to {out}")
    t0 = time.time()
    print(f"\n=== {tag} === baseline={args.baseline} K={args.k} N={args.n} "
          f"beta={args.beta} mu={args.num_iterations} scale={args.scale}", flush=True)

    with out.open("w") as f:
        conf = {k: v for k, v in vars(args).items()
                if isinstance(v, (str, int, float, bool, list, type(None)))}
        f.write(json.dumps({"record": "config", **conf, "tag": tag}) + "\n")
        progress = _step_progress(args, tag)
        for step in progress:
            if args.eval_every and step % args.eval_every == 0:
                _stage(f"{tag}: starting eval@{step} on {args.eval_problems} test problems", progress)
                e = evaluate(model, tok, n_problems=args.eval_problems,
                             max_new_tokens=args.max_new_tokens, temperature=0.0,
                             progress=getattr(args, "progress", True))
                f.write(json.dumps({"record": "eval", "step": step, **e}) + "\n")
                _progress_write(
                    progress,
                    f"       eval@{step}: {e['accuracy']:.1%} +/- {e['stderr']:.1%}",
                )

            m = trainer.step(rng.sample(data, args.k))
            m.update(record="step", step=step)
            f.write(json.dumps(m) + "\n")
            f.flush()
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    acc=f"{m['accuracy']:.3f}",
                    deg=f"{m['degenerate_frac']:.2f}",
                    shrink=f"{m['shrink']:.3f}",
                    gnorm=f"{m['grad_norm']:.2f}",
                    sec=f"{m['sec_total']:.0f}",
                )
            if step % args.log_every == 0:
                _progress_write(
                    progress,
                    f"[{step:4d}] acc={m['accuracy']:.3f} deg={m['degenerate_frac']:.2f} "
                    f"shrink={m['shrink']:.3f} advvar={m['adv_var']:.4f} "
                    f"gnorm={m['grad_norm']:6.3f} kl={m['kl']:.4f} "
                    + (f"upd={m['update_frac']:.3f} " if "update_frac" in m else "")
                    + (f"nsr={m['grad_noise_ratio']:.1f} " if "grad_noise_ratio" in m else "")
                    + f"clip={m['clip_frac']:.3f} {m['sec_total']:.0f}s",
                )

        _stage(f"{tag}: starting final eval on {args.eval_problems} test problems", progress)
        e = evaluate(model, tok, n_problems=args.eval_problems,
                     max_new_tokens=args.max_new_tokens, temperature=0.0,
                     progress=getattr(args, "progress", True))
        f.write(json.dumps({"record": "eval", "step": args.steps, **e}) + "\n")
        _progress_write(progress, f"       eval@final: {e['accuracy']:.1%} +/- {e['stderr']:.1%}")

    if args.save_checkpoint:
        d = ROOT / "checkpoints" / tag
        _stage(f"{tag}: saving checkpoint to {d}")
        model.save_pretrained(d); tok.save_pretrained(d)
        print(f"       checkpoint -> {d}")

    _stage(f"{tag}: done in {(time.time() - t0) / 60:.1f} min -> {out}")
    return out


def cmd_train(args):
    tok, model, ref = build(args, need_ref=args.beta > 0)
    tag = args.tag or f"{args.baseline}_k{args.k}n{args.n}_s{args.seed}"
    _train_one(args, tok, model, ref, tag)
    return 0


def cmd_sweep(args):
    """Same seed, same prompts, one arm per baseline -- a paired A/B."""
    import copy, gc

    import torch

    seeds = args.seeds or [args.seed]
    arms = [(seed, b) for seed in seeds for b in args.baselines]     # seed-major
    _stage(f"sweep start: seeds={seeds}, arms={', '.join(args.baselines)}")
    for arm_idx, (seed, b) in enumerate(arms, start=1):
        a = copy.copy(args)
        a.baseline, a.seed = b, seed
        tag = f"{b}_k{a.k}n{a.n}_s{a.seed}"
        _stage(f"sweep arm {arm_idx}/{len(arms)}: {tag} - building fresh policy")
        # Fresh policy per arm from the same seed, so every arm walks the same
        # prompt stream and starts from the same rollouts -- a paired A/B.
        tok, model, ref = build(a, need_ref=a.beta > 0)
        try:
            _train_one(a, tok, model, ref, tag)
            _stage(f"sweep arm {arm_idx}/{len(arms)}: {tag} complete")
        finally:
            del model, ref, tok
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"       vram released: {free / 2**30:.1f} of "
                  f"{total / 2**30:.1f} GiB free\n", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="grpo-eva", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("smoke", help="check the GPU can run bf16 on this arch")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("synthetic", help="phase 1: MSE of each baseline, ground truth known")
    p.add_argument("--k", type=int, nargs="+", default=[32])
    p.add_argument("--n", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--trials", type=int, default=2000)
    p.set_defaults(fn=cmd_synthetic)

    p = sub.add_parser("train", help="phase 2: GRPO on GSM8K")
    add_model_args(p); add_grpo_args(p)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50, help="0 disables mid-run eval")
    p.add_argument("--eval-problems", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--tag", default=None)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="write the final policy so it can be re-evaluated later")
    p.add_argument("--no-progress", action="store_false", dest="progress",
                   help="disable the tqdm training progress bar")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("sweep", help="one training run per baseline, paired by seed")
    add_model_args(p); add_grpo_args(p)
    p.add_argument("--baselines", nargs="+",
                   default=["vanilla", "js_fixed", "js_pooled", "global"])
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run every arm once per seed, seed-major (overrides --seed); "
                        "`compare` then pairs arms within a seed")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-problems", type=int, default=200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="write the final policy so it can be re-evaluated later")
    p.add_argument("--no-progress", action="store_false", dest="progress",
                   help="disable the tqdm training progress bar")
    p.set_defaults(fn=cmd_sweep, tag=None)

    p = sub.add_parser("gradvar", help="phase 3: paired gradient-noise measurement")
    add_model_args(p)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n", type=int, default=8)
    p.set_defaults(fn=cmd_gradvar)

    p = sub.add_parser("eval", help="pass@1 on the GSM8K test split")
    add_model_args(p)
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("prompt", help="compare candidate system prompts")
    add_model_args(p)
    p.add_argument("--n-problems", type=int, default=60)
    p.set_defaults(fn=cmd_prompt)

    p = sub.add_parser("report", help="compile report/report.tex to PDF")
    p.set_defaults(fn=lambda a: subprocess.call(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=ROOT / "report"))

    p = sub.add_parser("compare", help="paired McNemar across sweep arms")
    p.add_argument("--reference", default="vanilla")
    p.set_defaults(fn=lambda a: subprocess.call(
        [sys.executable, str(ROOT / "experiments/compare_arms.py"),
         "--reference", a.reference]))

    p = sub.add_parser("figures", help="regenerate every figure from saved results")
    p.set_defaults(fn=cmd_figures)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
