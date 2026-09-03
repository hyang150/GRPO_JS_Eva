"""CPU checks on the rollout bookkeeping in grpo.py."""

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from grpo_eva.grpo import GRPO, GRPOConfig, _completion_mask

IM_END, ENDOFTEXT = 151645, 151643     # Qwen2.5: eos_token, and pad == 2nd terminator


def test_completion_mask_stops_on_any_terminator():
    ids = torch.tensor([
        [5, 6, IM_END, ENDOFTEXT, ENDOFTEXT],        # chat stop, then padding
        [5, ENDOFTEXT, ENDOFTEXT, ENDOFTEXT, ENDOFTEXT],  # stopped on <|endoftext|>
        [5, 6, 7, 8, 9],                              # truncated: all valid
    ])
    m = _completion_mask(ids, {IM_END, ENDOFTEXT})
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert m.dtype == torch.float32


def test_completion_mask_with_only_im_end_leaks_padding():
    """The defect the stop-id set fixes: a completion that ended on
    <|endoftext|> (== pad) was treated as 512 valid tokens."""
    ids = torch.tensor([[5, ENDOFTEXT, ENDOFTEXT, ENDOFTEXT]])
    assert _completion_mask(ids, {IM_END}).sum() == 4
    assert _completion_mask(ids, {IM_END, ENDOFTEXT}).sum() == 2


def _stub_model(eos):
    m = torch.nn.Linear(2, 2)
    m.generation_config = SimpleNamespace(eos_token_id=eos)
    return m


@pytest.mark.parametrize("eos", [[IM_END, ENDOFTEXT], IM_END, None])
def test_trainer_collects_every_stop_id(eos):
    tok = SimpleNamespace(eos_token_id=IM_END, pad_token_id=ENDOFTEXT)
    t = GRPO(_stub_model(eos), tok, GRPOConfig())
    assert t._stop_ids == {IM_END, ENDOFTEXT}


def test_rollout_sampler_is_the_policy_by_default():
    """Qwen2.5's generation_config ships repetition_penalty=1.1; the loop must
    override it or every rollout is off-policy."""
    assert GRPOConfig().repetition_penalty == 1.0


# ------------------------------------------ per-step gradient variance (paper eq. 17-18)
from grpo_eva.grpo import micro_batch_grad_stats


def test_micro_batch_grad_stats_matches_the_direct_formula():
    torch.manual_seed(2)
    g = torch.randn(6, 7, dtype=torch.float64) + 3.0          # m = 6 contributions
    gbar = g.mean(0)
    direct_var = ((g - gbar) ** 2).sum() / (6 * 5)              # eq. 17
    s = micro_batch_grad_stats([(r ** 2).sum().item() for r in g],
                               (g.sum(0) ** 2).sum().item())
    assert s["grad_var"] == pytest.approx(direct_var.item(), rel=1e-9)
    assert s["grad_signal_sq"] == pytest.approx(((gbar ** 2).sum() - direct_var).item(), rel=1e-9)
    assert s["grad_noise_ratio"] == pytest.approx(s["grad_var"] / s["grad_signal_sq"])


def test_micro_batch_grad_stats_is_scale_free_and_needs_two_micro_batches():
    sq = [4.0, 9.0, 16.0]
    a = micro_batch_grad_stats(sq, 50.0)
    b = micro_batch_grad_stats([4 * x for x in sq], 200.0)
    assert a["grad_noise_ratio"] == pytest.approx(b["grad_noise_ratio"])
    assert all(v != v for v in micro_batch_grad_stats([1.0], 1.0).values())


def test_contribution_sq_differences_the_accumulated_gradient():
    lin = torch.nn.Linear(3, 1, bias=False)
    params = list(lin.parameters())
    lin.weight.grad = torch.tensor([[1.0, 2.0, 2.0]])           # G_1 = g_1, ||g_1||^2 = 9
    assert GRPO._contribution_sq(params, None) == pytest.approx(9.0)
    prev = [p.grad.clone() for p in params]
    lin.weight.grad += torch.tensor([[0.0, 3.0, 4.0]])          # G_2 = G_1 + g_2, ||g_2||^2 = 25
    assert GRPO._contribution_sq(params, prev) == pytest.approx(25.0)


def test_grad_var_tracking_is_off_by_default():
    assert GRPOConfig().track_grad_var is False
