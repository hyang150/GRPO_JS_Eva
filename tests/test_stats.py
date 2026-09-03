"""The hand-rolled statistics, checked against known values.

scipy is not a dependency, so the t-distribution tail and the exact McNemar
test are implemented here -- which means they need pinning.
"""

import importlib.util
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from grpo_eva.evaluate import mcnemar

_spec = importlib.util.spec_from_file_location(
    "compare_arms", pathlib.Path(__file__).resolve().parents[1] / "experiments/compare_arms.py")
compare_arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_arms)


@pytest.mark.parametrize("t,df,two_sided", [
    (2.5, 9, 0.0339), (1.0, 4, 0.3739), (4.2, 2, 0.0523),
    (0.5, 20, 0.6226), (0.0, 8, 1.0), (10.0, 30, 4.575252e-11),
])
def test_t_tail_matches_scipy(t, df, two_sided):
    """scipy.stats.t.sf(t, df) * 2; the far tail was cross-checked against
    Simpson integration of the density, agreeing to 13 significant figures."""
    assert compare_arms._t_sf(t, df) * 2 == pytest.approx(two_sided, rel=2e-3, abs=1e-12)


def test_t_tail_is_symmetric_and_bounded():
    for df in (1, 5, 50):
        assert compare_arms._t_sf(0.0, df) == pytest.approx(0.5)
        for t in (0.1, 1.0, 5.0):
            assert 0.0 <= compare_arms._t_sf(t, df) <= 0.5


def test_mcnemar_balanced_discordance_gives_p_one():
    """b == c is maximal agreement with the null, in any two-sided convention.

    Doubling the smaller tail exceeds 1 here (it already covers more than half
    the mass), which is why the result is clamped rather than reported raw.
    """
    m = mcnemar([0] * 11 + [1] * 11 + [0] * 178, [1] * 11 + [0] * 11 + [0] * 178)
    assert (m["fixed"], m["broken"]) == (11, 11)
    assert m["p_value"] == 1.0
    assert m["delta"] == 0.0


def test_mcnemar_ignores_agreements():
    """Only discordant pairs carry information."""
    a = mcnemar([0, 1], [1, 0])
    b = mcnemar([0, 1] + [1] * 500, [1, 0] + [1] * 500)
    assert a["p_value"] == b["p_value"]
    assert a["n_discordant"] == b["n_discordant"] == 2


@pytest.mark.parametrize("b,c,expect", [
    (17, 7, 0.0639), (14, 10, 0.5413), (9, 10, 1.0), (25, 5, 0.0003),
])
def test_mcnemar_exact_values(b, c, expect):
    m = mcnemar([0] * b + [1] * c, [1] * b + [0] * c)
    assert m["p_value"] == pytest.approx(expect, abs=5e-4)


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar([0, 1], [1])
