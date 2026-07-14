import numpy as np
import pytest
from mc.bootstrap import bootstrap_roi, ci


def test_determinism_with_seed():
    pnls, stakes = [0.2, -1.0, 0.3, 0.25], [1.0] * 4
    a = bootstrap_roi(pnls, stakes, n_sims=500, seed=0)
    b = bootstrap_roi(pnls, stakes, n_sims=500, seed=0)
    assert a.shape == (500,) and np.allclose(a, b)


def test_all_same_outcome_has_zero_variance():
    vals = bootstrap_roi([0.25] * 50, [1.0] * 50, n_sims=200, seed=0)
    assert np.allclose(vals, 0.25)


def test_ci_bounds_ordered_and_hand_computed():
    vals = np.arange(1001.0) / 1000.0  # uniform 0..1
    c = ci(vals)
    assert c["lower"] == pytest.approx(0.025, abs=0.002)
    assert c["upper"] == pytest.approx(0.975, abs=0.002)
    assert c["lower"] < c["mean"] < c["upper"]
