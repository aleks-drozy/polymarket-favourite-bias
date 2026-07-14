import pathlib
from data.dataset_loader import load_dataset_csv

FIX = pathlib.Path(__file__).parent / "fixtures" / "dataset_sample.csv"


def test_loads_valid_binary_resolved_rows():
    records, exclusions = load_dataset_csv(str(FIX))
    assert len(records) > 0
    r = records[0]
    assert r.resolved_outcome in ("YES", "NO")
    assert r.resolved_ts > r.created_ts
    assert r.volume >= 0


def test_non_binary_rows_become_exclusions_with_reason():
    _, exclusions = load_dataset_csv(str(FIX))
    reasons = {e.reason.split(":")[0] for e in exclusions}
    assert reasons <= {"not_binary", "not_resolved", "missing_field", "bad_outcome"}


def test_nothing_silently_dropped():
    import pandas as pd
    total_rows = len(pd.read_csv(FIX))
    records, exclusions = load_dataset_csv(str(FIX))
    assert len(records) + len(exclusions) == total_rows
