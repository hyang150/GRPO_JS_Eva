"""CPU checks for the random projections used by the gradient measurement."""

import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

from grad_variance import project


@pytest.mark.parametrize("shape", [(4,), (2, 2), (16,)])
def test_opposite_parameter_blocks_do_not_cancel(shape):
    gradients = [torch.ones(shape), -torch.ones(shape)]
    projections = [project(gradients, m) for m in range(24)]
    assert any(abs(value) > 1e-6 for value in projections)


def test_projected_covariance_matches_exact_covariance():
    generator = torch.Generator().manual_seed(19)
    blocks = torch.randn(8, 4, generator=generator)
    gradients = torch.cat((blocks, -blocks, 0.5 * blocks), dim=1)
    exact_trace = gradients.var(dim=0, unbiased=True).sum().item()

    projections = np.array([
        [project(list(row.split(4)), m) for m in range(2048)]
        for row in gradients
    ])
    estimated_trace = projections.var(axis=0, ddof=1).mean()
    assert estimated_trace == pytest.approx(exact_trace, rel=0.10)


def test_projection_is_repeatable_without_changing_global_rng():
    gradients = [torch.arange(4, dtype=torch.float32), torch.ones(4)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        initial_state = torch.get_rng_state().clone()
        first = project(gradients, 3)
        assert torch.equal(torch.get_rng_state(), initial_state)
        torch.rand(37)
        assert project(gradients, 3) == first


def test_paired_projections_preserve_gradient_scaling():
    gradients = [torch.arange(4, dtype=torch.float32), -torch.ones(4)]
    scaled = [2.0 * gradient for gradient in gradients]
    for m in range(24):
        assert project(scaled, m) == pytest.approx(2.0 * project(gradients, m))


# ------------------------------------------------ pair estimator (no probes)
from grad_variance import GradAccumulator


def test_pair_estimator_recovers_trace_and_signal():
    """g_b = mu + noise with known ||mu||^2 and tr Cov: disjoint consecutive
    pairs give unbiased estimates of both without any probe."""
    torch.manual_seed(5)
    d, n = 40, 400
    mu = torch.full((d,), 0.5)                          # ||mu||^2 = 10
    scale = torch.linspace(0.05, 0.5, d)                # tr Cov = sum(scale^2)
    acc = GradAccumulator(params=None)
    for _ in range(n):
        g = mu + scale * torch.randn(d)
        acc.add([g[:25], g[25:]])                       # two "parameter tensors"
    s = acc.summary(n_boot=200)
    assert s["n_pairs"] == n // 2
    assert s["pair_signal_sq"] == pytest.approx(float((mu ** 2).sum()), rel=0.05)
    assert s["pair_trace_cov"] == pytest.approx(float((scale ** 2).sum()), rel=0.10)
    assert s["pair_noise_ratio"] == pytest.approx(s["pair_trace_cov"] / s["pair_signal_sq"])
    lo, hi = s["pair_signal_sq_ci"]
    assert lo < float((mu ** 2).sum()) < hi
    assert s["trace_cov_probe_se"] > 0


def test_pairs_are_disjoint_and_the_held_gradient_is_released():
    acc = GradAccumulator(params=None)
    for i in range(5):
        acc.add([torch.full((3,), float(i))])
    assert len(acc.dots) == 2                           # (0,1), (2,3); 4 waits
    assert acc.dots == [0.0, 3 * 2.0 * 3.0]             # <g0,g1> = 0, <g2,g3> = 3*2*3
    assert acc._prev is not None                        # g4 held for a partner
    acc.add([torch.full((3,), 5.0)])
    assert acc._prev is None and len(acc.dots) == 3


def test_pair_summary_degrades_without_enough_pairs():
    acc = GradAccumulator(params=None)
    for i in range(3):
        acc.add([torch.randn(4)])
    s = acc.summary(n_boot=10)
    assert s["n_pairs"] == 1 and "pair_noise_ratio" not in s
