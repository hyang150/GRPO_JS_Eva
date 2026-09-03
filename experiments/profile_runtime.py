#!/usr/bin/env python3
"""Profile the existing GRPO path without changing runs or checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def union_duration(intervals):
    """Union of half-open time intervals, so concurrent kernels count once."""
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def summarize_trace(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    if not kernels:
        return {"kernel_count": 0, "warning": "No CUDA kernel events were captured."}
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernels]
    first, last = min(a for a, _ in intervals), max(b for _, b in intervals)
    busy = union_duration(intervals)
    totals, counts = Counter(), Counter()
    for event in kernels:
        totals[event["name"]] += event["dur"]
        counts[event["name"]] += 1
    runtime = Counter()
    for event in events:
        if event.get("cat") == "cuda_runtime" and event.get("ph") == "X":
            runtime[event["name"]] += event["dur"]
    return {
        "kernel_count": len(kernels),
        "kernel_span_ms": (last - first) / 1000,
        "kernel_union_ms": busy / 1000,
        "kernel_coverage_of_span": busy / (last - first) if last > first else None,
        "kernel_duration_median_us": statistics.median(e["dur"] for e in kernels),
        "top_kernels": [{"name": name, "sum_ms": duration / 1000, "count": counts[name]}
                        for name, duration in totals.most_common(15)],
        "top_cuda_runtime": [{"name": name, "sum_ms": duration / 1000}
                             for name, duration in runtime.most_common(10)],
        "caveat": "Coverage is not SM utilization or achieved FLOPS/bandwidth; profiling adds overhead.",
    }


def diagnostic_prefix(seq, attn, width, pad_id, eos_id):
    """Repack completed short rows on the left before a diagnostic continuation."""
    import torch

    ids = torch.full_like(seq[:, :width], pad_id)
    mask = torch.zeros_like(attn[:, :width])
    for i in range(seq.shape[0]):
        valid = seq[i, :width][attn[i, :width].bool()]
        if valid.numel() and valid[-1].item() == eos_id:
            valid = valid[:-1]
        if not valid.numel():
            raise ValueError("A diagnostic prefix must contain at least one valid token")
        ids[i, -valid.numel():] = valid
        mask[i, -valid.numel():] = 1
    return ids.contiguous(), mask.contiguous()


@contextmanager
def stage_instrumentation(trainer, *, synchronize):
    import torch
    import grpo_eva.grpo as module

    rows = defaultdict(list)

    def wrapper(label, fn):
        def call(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            if synchronize:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function("grpo/" + name):
                result = fn(*args, **kwargs)
                if synchronize:
                    torch.cuda.synchronize()
            rows[name].append(time.perf_counter() - start)
            return result
        return call

    def logprob_label(seq, attn, prompt_len, grad, model=None):
        return "forward_with_grad" if grad else "old_or_reference_logprobs"

    with ExitStack() as stack:
        for obj, name, label in [
            (trainer, "generate", "rollout"),
            (trainer, "_logprobs", logprob_label),
            (module, "correctness_reward", "reward"),
            (module, "compute_advantages", "advantages"),
            (torch.Tensor, "backward", "backward"),
            (torch.nn.utils, "clip_grad_norm_", "gradient_clipping"),
            (trainer.opt, "step", "optimizer"),
        ]:
            stack.enter_context(patch.object(obj, name, wrapper(label, getattr(obj, name))))
        yield rows


def capture_profile(label, fn, out):
    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        start = time.perf_counter()
        with torch.profiler.record_function("experiment/" + label):
            fn()
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
    trace = out / (label + ".trace.json")
    prof.export_chrome_trace(str(trace))
    averages = prof.key_averages()
    (out / (label + ".operators.txt")).write_text(
        averages.table(sort_by="self_device_time_total", row_limit=25)
        + "\n\nCPU self time:\n"
        + averages.table(sort_by="self_cpu_time_total", row_limit=25)
    )
    return {"profiled_wall_sec": wall, "trace": str(trace), **summarize_trace(trace)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--stage-steps", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.stage_steps, args.decode_tokens) < 1 or args.warmup < 0:
        parser.error("measurement counts must be positive and warmup nonnegative")
    out = args.out or ROOT / "results" / ("runtime_profile_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    import main as entry
    from grpo_eva.baselines import BASELINE_REGISTRY, compute_advantages
    from grpo_eva.data import load_gsm8k
    from grpo_eva.grpo import GRPO, GRPOConfig
    from grpo_eva.rewards import correctness_reward

    cfg = GRPOConfig(k_prompts=8, n_generations=8, micro_batch=8,
                     gen_micro_batch=64, seed=args.seed)
    sources = ["main.py", "src/grpo_eva/grpo.py", "src/grpo_eva/baselines.py",
               "src/grpo_eva/data.py", "src/grpo_eva/rewards.py"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "config": asdict(cfg),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Disposable pretrained policy, no checkpoint saved, no evaluation, no existing logs modified.",
    }

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Loading the original training path...", flush=True)
    start = time.perf_counter()
    tok, model, _ = entry.build(args)
    summary["model_architecture"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "parameter_dtype": str(next(model.parameters()).dtype),
    }
    trainer = GRPO(model, tok, cfg)
    data = load_gsm8k(tok, "train")
    summary["load_and_data_sec"] = time.perf_counter() - start
    rng = random.Random(args.seed)
    torch.cuda.reset_peak_memory_stats()
    summary["unprofiled_steps"] = []
    for step in range(args.warmup + args.steps):
        batch = rng.sample(data, cfg.k_prompts)
        torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = trainer.step(batch)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        print(f"{'warmup' if step < args.warmup else 'timed'} {step}: "
              f"{wall:.3f}s, generation {metrics['sec_gen']:.3f}s, "
              f"mean length {metrics['completion_len']:.1f}", flush=True)
        if step >= args.warmup:
            summary["unprofiled_steps"].append({"wall_sec": wall, **metrics})
        save()
    summary["unprofiled_mean_sec"] = statistics.mean(r["wall_sec"] for r in summary["unprofiled_steps"])

    summary["synchronized_stage_steps"] = []
    for step in range(args.stage_steps):
        with stage_instrumentation(trainer, synchronize=True) as timings:
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = trainer.step(rng.sample(data, cfg.k_prompts))
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
        row = {"wall_sec": wall, "stages_sec": {k: sum(v) for k, v in timings.items()},
               "calls": {k: len(v) for k, v in timings.items()}, "completion_len": metrics["completion_len"]}
        summary["synchronized_stage_steps"].append(row)
        print("stages: " + json.dumps(row), flush=True)
        save()

    batch = rng.sample(data, cfg.k_prompts)
    model.eval()
    with torch.no_grad():
        rollout = trainer.generate([b["prompt"] for b in batch])
    seq, attn, plen, cmask, texts = rollout
    summary["replay_shape"] = {"sequences": seq.shape[0], "prompt_tokens": plen,
                               "padded_total_tokens": seq.shape[1], "valid_completion_tokens": cmask.sum().item()}

    # Use the same 64 existing prefixes, with a bounded generation window to
    # keep the trace small. This is not an end-to-end 512-token benchmark.
    prefix_width = min(plen + 128, seq.shape[1])
    ids, mask = diagnostic_prefix(seq, attn, prefix_width, tok.pad_token_id, tok.eos_token_id)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    @torch.no_grad()
    def decode_sample():
        torch.manual_seed(args.seed + 991)
        return model.generate(input_ids=ids, attention_mask=mask,
                              do_sample=True, temperature=cfg.temperature,
                              top_p=cfg.top_p, top_k=0, use_cache=True,
                              max_new_tokens=args.decode_tokens, min_new_tokens=args.decode_tokens,
                              pad_token_id=tok.pad_token_id)

    for _ in range(2):
        decode_sample()
    short_times = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        decode_sample()
        torch.cuda.synchronize()
        short_times.append(time.perf_counter() - start)
    print("Capturing a bounded generation trace...", flush=True)
    summary["generation_profile"] = {
        "prefix_tokens": prefix_width, "new_tokens": args.decode_tokens,
        "sequences": ids.shape[0], "unprofiled_sec": short_times,
        "scope": "Prefill plus fixed-length decode of left-padded cached prefixes, terminal EOS removed; not a full rollout or quality eval.",
        **capture_profile("generation_sample", decode_sample, out),
    }
    save()

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Capturing an update trace on an existing full-length rollout...", flush=True)
    with patch.object(trainer, "generate", return_value=rollout):
        with stage_instrumentation(trainer, synchronize=False):
            summary["update_profile"] = capture_profile("update_replay", lambda: trainer.step(batch), out)
    save()

    golds = [b["gold"] for b in batch for _ in range(cfg.n_generations)]
    rewards = torch.tensor(correctness_reward(texts, golds), device="cuda").view(8, 8)
    summary["baseline_microbenchmark_ms"] = {}
    for name in BASELINE_REGISTRY:
        for _ in range(10):
            compute_advantages(rewards, baseline=name, scale=cfg.scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            compute_advantages(rewards, baseline=name, scale=cfg.scale)
        torch.cuda.synchronize()
        summary["baseline_microbenchmark_ms"][name] = (time.perf_counter() - start) * 10
    summary["memory_gib"] = {
        "peak_allocated": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved": torch.cuda.max_memory_reserved() / 2**30,
        "total_device": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }
    save()
    print("DONE: " + str(out / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
