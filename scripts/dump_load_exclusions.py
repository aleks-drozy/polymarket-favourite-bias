"""Persist the metadata-load exclusion breakdown (Task 13 review follow-up).

`load_from_gamma`'s `load_exclusions` (not_binary, outside_study_window,
missing_field:*, bad_outcome -- everything dropped before the volume floor
is even applied) are computed on every `run_backtest.py` run but never
written to disk -- only the post-floor `results/exclusions.csv` funnel
(no_candle_in_window) is persisted. This script re-runs the same Gamma load
against the warm on-disk cache (`data/api_client.py` caches every response
by URL hash; the study-window params below match `run_backtest.py`'s
exactly, so every URL is a cache hit) and writes `results/load_exclusions.csv`
so the writeup's exclusion-funnel table is fully sourced from disk, not
re-derived from a log line.

Instant, zero network calls, given a warm cache/ directory.
"""
from __future__ import annotations
import collections
import csv
import pathlib
import sys

# Repo root isn't on sys.path when this is invoked as `python scripts/dump_load_exclusions.py`
# (only the script's own directory is) -- add it so the project-root imports below resolve
# regardless of cwd, same as running `python run_backtest.py` from the repo root does implicitly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data.api_client import PolymarketClient
from data.dataset_loader import load_from_gamma
from run_backtest import STUDY_WINDOW_END_DATE_MIN, STUDY_WINDOW_END_DATE_MAX


def main() -> None:
    client = PolymarketClient()
    records, load_exclusions = load_from_gamma(
        client, end_date_min=STUDY_WINDOW_END_DATE_MIN, end_date_max=STUDY_WINDOW_END_DATE_MAX)

    out_dir = pathlib.Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "load_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["market_id", "reason"])
        for e in load_exclusions:
            w.writerow([e.market_id, e.reason])

    counts = collections.Counter(e.reason for e in load_exclusions)
    print(f"records: {len(records)}, load_exclusions: {len(load_exclusions)}")
    for reason, n in counts.most_common():
        print(f"  {reason}: {n}")


if __name__ == "__main__":
    main()
