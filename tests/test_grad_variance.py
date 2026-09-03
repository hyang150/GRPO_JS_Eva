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
