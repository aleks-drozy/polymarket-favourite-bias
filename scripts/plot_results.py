"""Render the two figures the README embeds, straight from committed results.

    python scripts/plot_results.py        # -> charts/bootstrap_roi.png, charts/strategy_comparison.png

Inputs: results/mc_results.json (favourite study), results/mc_results_underdog.json
(mirror study), and results/results.json (the per-bet ledger).

Why the ledger is needed: mc_results.json stores the bootstrap *summary* (mean and
the 2.5/97.5 percentiles), not the 10,000 draws behind it, and a distribution can't
be honestly drawn from three summary numbers -- fitting a bell curve to them would
be inventing data the study never produced. So the draws are recomputed here with
the study's own `mc.bootstrap.bootstrap_roi` at the same seed and sim count that
`run_mc.py` used, and the resulting percentiles are asserted equal to the ones
stored in mc_results.json. If that assertion ever fails, the chart and the reported
CI have diverged and the chart is the thing that's wrong.

Style is deliberately plain: opaque near-white background (a transparent background
renders as dark-on-dark for anyone reading GitHub in dark mode), one accent colour,
horizontal gridlines only, no gradients.
"""
from __future__ import annotations
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mc.bootstrap import bootstrap_roi, ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CHARTS = ROOT / "charts"

# Must match run_mc.py's defaults, or the recomputed draws are not the study's draws.
N_SIMS = 10000
SEED = 0

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
ACCENT = "#1f4e79"      # favourite
ACCENT_FILL = "#9fc0dd"  # CI band / bar fill
NEUTRAL = "#9a9a9a"      # random side
WARN = "#a33a3a"         # underdog
ZERO_LINE = "#1a1a1a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#8a8a8a",
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=9)


def bootstrap_chart(mc: dict, out: pathlib.Path) -> None:
    bets = _load("results.json")["bets"]
    draws = bootstrap_roi([b["pnl"] for b in bets], [b["stake"] for b in bets],
                          n_sims=N_SIMS, seed=SEED) * 100.0

    stored = mc["bootstrap_ci"]
    check = ci(draws / 100.0)
    for key in ("mean", "lower", "upper"):
        assert abs(check[key] - stored[key]) < 1e-12, (
            f"recomputed bootstrap {key} ({check[key]}) != mc_results.json "
            f"({stored[key]}) -- the chart would misreport the study")

    lo, hi = stored["lower"] * 100, stored["upper"] * 100
    observed = mc["observed"]["roi"] * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(lo, hi, color=ACCENT_FILL, alpha=0.45, zorder=1,
               label=f"95% CI  {lo:.2f}% to {hi:+.2f}%")
    ax.hist(draws, bins=60, color=ACCENT, alpha=0.85, zorder=2, linewidth=0)
    ax.axvline(0.0, color=ZERO_LINE, linestyle="--", linewidth=1.4, zorder=4,
               label="break-even (0% ROI)")
    ax.axvline(observed, color=WARN, linewidth=1.4, zorder=5,
               label=f"observed ROI {observed:.2f}%")

    ax.set_xlabel("ROI per dollar staked (%)", labelpad=7)
    ax.set_ylabel("bootstrap resamples", labelpad=7)
    ax.set_title("Always bet the favourite: bootstrap distribution of ROI",
                 fontsize=12.5, pad=34, loc="left")
    ax.text(0.0, 1.045,
            f"{N_SIMS:,} resamples with replacement over "
            f"{mc['observed']['n_bets']:,} resolved PolyMarket markets.\n"
            f"The 95% interval spans break-even, so the pre-registered gate fails.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom",
            linespacing=1.5)

    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _frame(ax)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.5,
                    borderpad=0.6, handlelength=1.6)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def comparison_chart(mc: dict, mc_dog: dict, out: pathlib.Path) -> None:
    n_bets = mc["observed"]["n_bets"]
    assert n_bets == mc_dog["observed"]["n_bets"], (
        "the mirror study ran on a different sample -- 'same markets' would be a lie")
    rows = [
        ("Favourite\n(always the >50% side)", mc["observed"]["roi"] * 100, ACCENT),
        ("Random side\n(coin flip, null)", mc["null"]["null_mean_roi"] * 100, NEUTRAL),
        ("Underdog\n(always the <50% side)", mc_dog["observed"]["roi"] * 100, WARN),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], width=0.55,
           color=[r[2] for r in rows], zorder=2, linewidth=0)
    ax.axhline(0.0, color=ZERO_LINE, linewidth=1.2, zorder=3)

    for x, (_, value, _) in zip(xs, rows):
        ax.annotate(f"{value:.2f}%", (x, value), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=10.5, color=INK,
                    fontweight="bold", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("ROI per dollar staked (%)", labelpad=7)
    ax.set_ylim(min(r[1] for r in rows) * 1.22, 8)
    ax.set_title(f"Same {n_bets:,} markets, same prices, same fees "
                 f"-- only the side changes",
                 fontsize=12.5, pad=34, loc="left")
    ax.text(0.0, 1.045,
            "Observed ROI after fees. Random side is the mean of the "
            "10,000-sim coin-flip null.\n"
            "The favourite-longshot bias shows up as \"don't buy longshots\", "
            "not \"buy favourites for profit\".",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom",
            linespacing=1.5)
    ax.set_xlim(-0.62, 2.62)
    ax.annotate("break-even", (2.58, 0), textcoords="offset points",
                xytext=(0, 6), ha="right", va="bottom", fontsize=8.5, color=MUTED)

    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _frame(ax)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    mc = _load("mc_results.json")
    mc_dog = _load("mc_results_underdog.json")

    targets = [
        (CHARTS / "bootstrap_roi.png", lambda p: bootstrap_chart(mc, p)),
        (CHARTS / "strategy_comparison.png", lambda p: comparison_chart(mc, mc_dog, p)),
    ]
    for path, render in targets:
        render(path)
        print(f"wrote {path.relative_to(ROOT).as_posix()} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
