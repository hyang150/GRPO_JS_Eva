"""A minimal GRPO loop, written so the baseline is the only moving part.

Everything that could confound a baseline A/B is fixed by default:
  * no KL term (beta = 0), so no reference model in memory and no second
    force acting on the policy;
  * no division by the within-group std (``scale="none"``), because that
    term interacts with the baseline -- it is the difficulty bias Dr.GRPO
    (arXiv:2503.20783) identifies;
  * constant loss normalisation, so completion length does not reweight
    the update.
Change those only on purpose, and change them for every arm at once.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time

import torch
import torch.nn.functional as F

from .baselines import compute_advantages, shrink_factor
from .rewards import correctness_reward


@dataclasses.dataclass
class GRPOConfig:
    baseline: str = "vanilla"
    precision: str = "bf16"         # bf16 | mixed -- see GRPO.autocast
    track_update_precision: bool = False
    beta: float = 0.0               # KL coefficient; > 0 loads a reference model
    num_iterations: int = 1         # mu -- inner updates per rollout batch
    k_prompts: int = 8              # groups per step
    n_generations: int = 8          # completions per group
    lr: float = 1e-6
    max_prompt_len: int = 320
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    epsilon: float = 0.2            # PPO clip
    grad_clip: float = 1.0
    scale: str = "none"             # none | group | batch
    loss_norm: str = "constant"     # constant | per_seq | per_token
    micro_batch: int = 4            # sequences per forward; the memory knob
    gen_micro_batch: int = 32       # sequences per generate() call
    seed: int = 0


def _completion_mask(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """1 up to and including the first EOS, 0 after it."""
    b, c = completion_ids.shape
    is_eos = completion_ids == eos_id
    idx = torch.full((b,), c - 1, dtype=torch.long, device=completion_ids.device)
    has = is_eos.any(dim=1)
    idx[has] = is_eos.int().argmax(dim=1)[has]
    ar = torch.arange(c, device=completion_ids.device).expand(b, -1)
    return (ar <= idx.unsqueeze(1)).to(torch.float32)


class GRPO:
    def autocast(self):
        """bf16 compute, with or without fp32 master weights.

        ``precision="bf16"`` keeps parameters, gradients *and* AdamW's moments
        in bf16.  That is not the usual mixed-precision recipe, and at small
        learning rates it silently discards most of the update: bf16 carries 8
        mantissa bits, so a weight of 0.0145 has a ULP of 1.1e-4, while an
        AdamW step at lr=1e-6 moves it by ~1e-6 -- a hundred times below the
        smallest representable increment, so it rounds away.  Measured on
        Qwen2.5-0.5B: only 2.3% of weights change per step at lr=1e-6, against
        90% at lr=1e-4.  Set ``track_update_precision`` to watch this.

        ``precision="mixed"`` holds fp32 master weights and casts only the
        forward pass, which is the standard recipe and costs roughly 2x the
        parameter and optimizer memory -- it does not fit a 16 GB card at
        K=8 x N=8, which is why it is not the default here.
        """
        if self.cfg.precision == "mixed":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def __init__(self, model, tokenizer, cfg: GRPOConfig, ref_model=None):
        self.model, self.tok, self.cfg = model, tokenizer, cfg
        if cfg.precision not in ("bf16", "mixed"):
            raise ValueError(f"precision must be bf16 or mixed, got {cfg.precision!r}")
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                                     weight_decay=0.0, eps=1e-8)
        self.device = next(model.parameters()).device
        # a fixed sample of tensors, so the update-precision probe costs
        # ~1e-3 of a step instead of comparing all 494M parameters
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        self._probe = [q for _, q in named[::max(1, len(named) // 8)]][:8]
        self.ref_model = ref_model
        if cfg.beta > 0 and ref_model is None:
            raise ValueError("beta > 0 requires a reference model")
        if ref_model is not None:
            ref_model.eval().requires_grad_(False)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def generate(self, prompts: list[str]):
        """Returns (seq_ids, attn_mask, prompt_len, completion_mask, texts).

        Gradient checkpointing forces ``use_cache=False`` on the config, which
        would make generation quadratic in length -- it must be turned off
        around generate() and back on for the backward pass.
        """
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       padding_side="left", truncation=True,
                       max_length=self.cfg.max_prompt_len).to(self.device)
        prompt_len = enc.input_ids.shape[1]

        outs = []
        # generate() with num_return_sequences=N expands the batch N-fold;
        # chunk over prompts so the KV cache stays bounded.
        per = max(1, self.cfg.gen_micro_batch // self.cfg.n_generations)
        for i in range(0, enc.input_ids.shape[0], per):
            with self.autocast():
                out = self.model.generate(
                input_ids=enc.input_ids[i:i + per],
                attention_mask=enc.attention_mask[i:i + per],
                num_return_sequences=self.cfg.n_generations,
                do_sample=True, temperature=self.cfg.temperature,
                top_p=self.cfg.top_p, top_k=0,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )
            outs.append(out)
        width = max(o.shape[1] for o in outs)
        outs = [F.pad(o, (0, width - o.shape[1]), value=self.tok.pad_token_id)
                for o in outs]
        seq = torch.cat(outs, dim=0)

        completion_ids = seq[:, prompt_len:]
        cmask = _completion_mask(completion_ids, self.tok.eos_token_id)
        pmask = enc.attention_mask.repeat_interleave(self.cfg.n_generations, dim=0)
        attn = torch.cat([pmask, cmask.to(pmask.dtype)], dim=1)
        texts = self.tok.batch_decode(completion_ids, skip_special_tokens=True)

        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        return seq, attn, prompt_len, cmask, texts

    # -------------------------------------------------------------- logprobs
    def _logprobs(self, seq, attn, prompt_len, grad: bool, model=None):
        """Per-token logprob of the completion. Chunked: the logits tensor is
        (B, L, 151936) and materialising it for the whole batch will OOM."""
        model = model or self.model
        chunks = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(0, seq.shape[0], self.cfg.micro_batch):
                ids, am = seq[i:i + self.cfg.micro_batch], attn[i:i + self.cfg.micro_batch]
                with self.autocast():
                    logits = model(input_ids=ids, attention_mask=am).logits
                logits = logits[:, prompt_len - 1:-1, :]      # predicts completion
                tgt = ids[:, prompt_len:]
                # cross_entropy == -logsoftmax.gather, without the extra tensor
                lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                      tgt.reshape(-1), reduction="none")
                chunks.append(lp.view(tgt.shape))
                del logits
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------ step
    def step(self, batch: list[dict]) -> dict:
        cfg = self.cfg
        t0 = time.time()
        self.model.eval()
        seq, attn, plen, cmask, texts = self.generate([b["prompt"] for b in batch])
        t_gen = time.time() - t0

        golds = [b["gold"] for b in batch for _ in range(cfg.n_generations)]
        rewards = torch.tensor(correctness_reward(texts, golds),
                               device=self.device).view(len(batch), cfg.n_generations)

        adv = compute_advantages(rewards, baseline=cfg.baseline, scale=cfg.scale)
        adv_flat = adv.reshape(-1)

        self.model.train()
        old_lp = self._logprobs(seq, attn, plen, grad=False).detach()
        ref_lp = (self._logprobs(seq, attn, plen, grad=False,
                                 model=self.ref_model).detach()
                  if cfg.beta > 0 else None)

        mb = cfg.micro_batch
        denom = {"constant": cmask.shape[0] * cfg.max_new_tokens,
                 "per_token": cmask.sum().clamp(min=1),
                 "per_seq": cmask.shape[0]}[cfg.loss_norm]

        total_loss = kl_sum = clip_frac = 0.0
        n_micro = 0
        for _ in range(cfg.num_iterations):        # mu: ratio != 1 after the first
            self.opt.zero_grad(set_to_none=True)
            for i in range(0, seq.shape[0], mb):
                sl = slice(i, i + mb)
                lp = self._logprobs(seq[sl], attn[sl], plen, grad=True)
                ratio = torch.exp(lp - old_lp[sl])
                a = adv_flat[sl].unsqueeze(1)
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - cfg.epsilon, 1 + cfg.epsilon) * a
                per_tok = -torch.min(unclipped, clipped)

                if cfg.beta > 0:
                    # k3 estimator: unbiased, non-negative, low variance
                    # (Schulman) -- exp(d) - d - 1 with d = ref_lp - lp
                    d = ref_lp[sl] - lp
                    kl = torch.exp(d) - d - 1.0
                    per_tok = per_tok + cfg.beta * kl
                    kl_sum += float((kl.detach() * cmask[sl]).sum()
                                    / cmask[sl].sum().clamp(min=1))

                per_tok = per_tok * cmask[sl]
                if cfg.loss_norm == "per_seq":
                    loss = (per_tok.sum(1) / cmask[sl].sum(1).clamp(min=1)).sum() / denom
                else:
                    loss = per_tok.sum() / denom
                loss.backward()

                with torch.no_grad():
                    clip_frac += float((((ratio < 1 - cfg.epsilon) & (a < 0)) |
                                        ((ratio > 1 + cfg.epsilon) & (a > 0)))
                                       .float().mul(cmask[sl]).sum()
                                       / cmask[sl].sum().clamp(min=1))
                total_loss += loss.item()
                n_micro += 1

            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if cfg.track_update_precision:
                probe_before = [q.detach().clone() for q in self._probe]
            self.opt.step()
            if cfg.track_update_precision:
                moved = sum(int((a != b).sum()) for a, b in zip(probe_before, self._probe))
                total = sum(q.numel() for q in self._probe)
                update_frac = moved / total

        grp = rewards.sum(dim=1)
        out = {
            "loss": total_loss,
            "reward_mean": rewards.mean().item(),
            "accuracy": rewards.mean().item(),
            "reward_std_within": rewards.std(dim=1).mean().item(),
            "adv_var": adv.var().item(),
            "adv_abs_mean": adv.abs().mean().item(),
            "degenerate_frac": (((grp == 0) | (grp == cfg.n_generations))
                                .float().mean().item()),
            "shrink": shrink_factor(rewards, cfg.baseline).mean().item(),
            "grad_norm": gnorm.item(),
            "kl": kl_sum / max(n_micro, 1),
            "clip_frac": clip_frac / max(n_micro, 1),
            "entropy_proxy": -old_lp.mul(cmask).sum().item() / cmask.sum().item(),
            "completion_len": cmask.sum(1).mean().item(),
            "sec_gen": t_gen,
            "sec_total": time.time() - t0,
            "vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        if cfg.track_update_precision:
            # fraction of sampled weights the optimizer step actually moved;
            # under pure bf16 at a small lr most updates round to nothing
            out["update_frac"] = update_frac
        return out
