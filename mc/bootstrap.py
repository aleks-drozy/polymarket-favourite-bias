"""Bootstrap (resample-with-replacement) distribution of aggregate ROI (spec §6)."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from mc.metrics import roi


def bootstrap_roi(pnls: Sequence[float], stakes: Sequence[float],
                  n_sims: int = 10000, seed: int = 0) -> np.ndarray:
    pn = np.asarray(pnls, dtype=float)
    st = np.asarray(stakes, dtype=float)
    n = len(pn)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for i in range(n_sims):
        idx = rng.integers(0, n, size=n)
        out[i] = roi(pn[idx], st[idx])
    return out


def ci(values: np.ndarray, level: float = 0.95) -> dict:
    alpha = (1.0 - level) / 2.0
    return {"mean": float(values.mean()),
            "lower": float(np.percentile(values, 100 * alpha)),
            "upper": float(np.percentile(values, 100 * (1 - alpha)))}
