import pathlib
from data.dataset_loader import (
    load_dataset_csv,
    load_from_gamma,
    _classify_outcome,
    _gamma_to_record,
    RESOLVED_PRICE_THRESHOLD,
)
from data.schema import MarketRecord, Exclusion

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
    # Exact-count assertion: pin the fixture to 3 records, 5 exclusions, so any
    # future threshold drift (e.g. reverting RESOLVED_PRICE_THRESHOLD back to 0.9)
    # breaks loudly instead of silently reclassifying records. The fixture row
    # 0xbbb222 with prices [0.001, 0.999] is specifically the boundary case:
    # it resolves to NO under 0.999 threshold (after round-2 fix) but would
    # become an exclusion under any looser threshold.
    assert len(records) == 3
    assert len(exclusions) == 5


# --- Boundary-pinning tests (review round 1) ---------------------------------
# The reviewer's finding: any RESOLVED_PRICE_THRESHOLD between 0.51 and 0.98
# passed the original 3 tests, because none of them exercised the boundary.
# These pin the constant itself (currently 0.999) rather than just "some
# threshold in a big range" -- if someone silently loosens or tightens
# RESOLVED_PRICE_THRESHOLD, one of the two assertions below flips.

def test_classify_outcome_at_threshold_resolves():
    # Exactly at the live constant -- must resolve. Sanity check that the
    # constant hasn't drifted out from under this test.
    assert RESOLVED_PRICE_THRESHOLD == 0.999
    outcome, reason = _classify_outcome(["Yes", "No"], [0.999, 0.001])
    assert outcome == "YES"
    assert reason is None


def test_classify_outcome_just_below_threshold_is_excluded():
    # One float step below the boundary -- must NOT resolve. Together with
    # the test above, this pins RESOLVED_PRICE_THRESHOLD to exactly 0.999:
    # raising it would flip the first test, lowering it would flip this one.
    outcome, reason = _classify_outcome(["Yes", "No"], [0.9989, 0.0011])
    assert outcome is None
    assert reason == "bad_outcome"


def test_classify_outcome_boundary_applies_to_either_side():
    outcome, reason = _classify_outcome(["Yes", "No"], [0.0011, 0.9989])
    assert outcome is None
    assert reason == "bad_outcome"


# --- load_from_gamma / _gamma_to_record coverage (review round 1) ------------
# Realistic Gamma market dicts, grounded in live `GET /markets?closed=...`
# responses pulled during the round-1 investigation (see
# .superpowers/sdd/task-11-report.md, "Fix round 1"). Gamma encodes
# outcomes/outcomePrices as JSON *strings* (not native lists), which these
# dicts reproduce faithfully -- a shape difference from the CSV fixture that
# is otherwise untested.

# Real market (conditionId 0x9b946f5...), closed 2021, resolved NO. The
# winning side is 0.9999989... rather than exactly 1.0 -- real UMA/AMM
# settlement noise, not synthetic.
GAMMA_VALID_RESOLVED = {
    "conditionId": "0x9b946f54f3428aafc308c33aa04a943fe13a011bdac9a9b66e1ba16c416ca256",
    "category": "Pop-Culture",
    "createdAt": "2020-10-02T19:20:04.234Z",
    "closedTime": "2021-01-02 21:35:34+00",
    "closed": True,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.000001011082052522541417308141468657552", "0.9999989889179474774585826918585313"]',
    "volumeNum": 22067.48,
}

# Real market (conditionId 0xd903891...), category "Crypto", non-binary
# Long/Short outcome labels -- a real pattern from Gamma's crypto-price
# markets, not the brief's Yes/No shape.
GAMMA_NOT_BINARY = {
    "conditionId": "0xd903891c2b9046cae14615afc4c5245370143503f7b2dfc13919acee07a1696d",
    "category": "Crypto",
    "createdAt": "2020-10-02T20:00:04.856Z",
    "closedTime": "2020-11-05 16:21:29+00",
    "closed": True,
    "outcomes": '["Long", "Short"]',
    "outcomePrices": '["0", "0"]',
    "volumeNum": 59755.8,
}

# Live open markets (verified via a real `closed=false` pull) always return
# closedTime=null while open -- there is no real example of closed=False with
# a populated closedTime. That null closedTime would itself trip the
# missing_field:resolved_ts check before ever reaching the not_resolved
# check, so a genuinely-live open market never actually reaches the
# not_resolved branch through _gamma_to_record as currently ordered. This
# dict isolates the not_resolved branch on its own (closedTime populated,
# closed=False) the same way the hand-built CSV fixture's ddd555 row already
# does; the closedTime=null real-world case is covered separately below.
GAMMA_NOT_RESOLVED = {
    "conditionId": "0x1fad72fae204143ff1c3035e99e7c0f65ea8d5cd9bd1070987bd1a3316f772be",
    "category": "Crypto",
    "createdAt": "2025-05-02T15:04:43.762151Z",
    "closedTime": "2026-12-31 00:00:00+00",
    "closed": False,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.505", "0.495"]',
    "volumeNum": 856613.0,
}

# Real market (conditionId 0xe3b423d...), closed 2020 but never resolved to
# either side -- outcomePrices=["0","0"] is the real degenerate/void pattern,
# not synthetic.
GAMMA_BAD_OUTCOME = {
    "conditionId": "0xe3b423dfad8c22ff75c9899c4e8176f628cf4ad4caa00481764d320e7415f7a9",
    "category": "US-current-affairs",
    "createdAt": "2020-10-02T16:10:01.467Z",
    "closedTime": "2020-11-02 16:31:01+00",
    "closed": True,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0", "0"]',
    "volumeNum": 32257.45,
}

# Real market (conditionId 0xa9096ff...): category is genuinely null on this
# live closed market ("Games Total: O/U 2.5"), which is a real, recurring
# Gamma pattern (also seen on real open markets during this investigation).
# Its outcomes are also non-binary (Over/Under) -- so under Task 13's fix
# (null category no longer excludes on its own, see
# data/dataset_loader.py's "Task 13 real-API discovery" note), this record
# still ends up excluded, but for "not_binary", not "missing_field:category".
GAMMA_MISSING_FIELD = {
    "conditionId": "0xa9096ff7e25f808b537e7f95e4d6b690c88f7dc4a49cf01c05ff13e9b401468a",
    "category": None,
    "createdAt": "2026-05-11T12:00:04.714394Z",
    "closedTime": "2026-05-12 06:44:09+00",
    "closed": True,
    "outcomes": '["Over", "Under"]',
    "outcomePrices": '["0", "1"]',
    "volumeNum": 99.99999999999999,
}

# Real market (conditionId 0xdd22472e...): Polymarket's highest-volume market
# ever ($1.53B), the 2024 US Presidential Election winner market. category is
# null here too -- Task 13's live investigation found this is universal for
# the entire CLOB-covered era (2022-09 onward), not an isolated data-quality
# blip: the flat `category` field simply stopped being populated Gamma-side.
# Clean binary Yes/No, cleanly resolved -- isolates the "null category alone
# must not exclude" behavior from any other exclusion reason.
GAMMA_NULL_CATEGORY_VALID = {
    "conditionId": "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917",
    "category": None,
    "createdAt": "2024-01-04T17:33:51.332Z",
    "closedTime": "2024-11-06 15:17:41+00",
    "closed": True,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["1", "0"]',
    "volumeNum": 1531479284.504353,
}

# Synthetic: a genuinely-missing *other* required field (volumeNum absent)
# must still exclude -- confirms the category exemption in the required-field
# loop didn't accidentally weaken the check for every other column.
GAMMA_MISSING_VOLUME = {
    "conditionId": "0xtest_missing_volume",
    "category": "Politics",
    "createdAt": "2023-01-01T00:00:00Z",
    "closedTime": "2023-02-01 00:00:00+00",
    "closed": True,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["1", "0"]',
}


def test_gamma_to_record_valid_resolved_binary_market():
    r = _gamma_to_record(GAMMA_VALID_RESOLVED)
    assert isinstance(r, MarketRecord)
    assert r.resolved_outcome == "NO"
    assert r.category == "pop-culture"
    assert r.volume == 22067.48


def test_gamma_to_record_not_binary():
    r = _gamma_to_record(GAMMA_NOT_BINARY)
    assert isinstance(r, Exclusion)
    assert r.reason == "not_binary"


def test_gamma_to_record_not_resolved():
    r = _gamma_to_record(GAMMA_NOT_RESOLVED)
    assert isinstance(r, Exclusion)
    assert r.reason == "not_resolved"


def test_gamma_to_record_bad_outcome():
    r = _gamma_to_record(GAMMA_BAD_OUTCOME)
    assert isinstance(r, Exclusion)
    assert r.reason == "bad_outcome"


def test_gamma_to_record_missing_field():
    # Category-nullness no longer excludes on its own (Task 13 fix); this
    # fixture's outcomes are Over/Under, so it now falls through to
    # not_binary instead of missing_field:category.
    r = _gamma_to_record(GAMMA_MISSING_FIELD)
    assert isinstance(r, Exclusion)
    assert r.reason == "not_binary"


def test_gamma_to_record_null_category_defaults_to_unknown_not_excluded():
    r = _gamma_to_record(GAMMA_NULL_CATEGORY_VALID)
    assert isinstance(r, MarketRecord)
    assert r.category == "unknown"
    assert r.resolved_outcome == "YES"


def test_gamma_to_record_missing_non_category_field_still_excludes():
    r = _gamma_to_record(GAMMA_MISSING_VOLUME)
    assert isinstance(r, Exclusion)
    assert r.reason == "missing_field:volume"


def test_gamma_to_record_strips_whitespace_from_category():
    # Real Gamma data includes trailing-space category values (e.g.
    # "Pop-Culture ", verified live) that would otherwise miss a clean
    # fees.py CATEGORY_RATES key match even when one exists.
    m = dict(GAMMA_VALID_RESOLVED, category="Politics ")
    r = _gamma_to_record(m)
    assert isinstance(r, MarketRecord)
    assert r.category == "politics"


class _StubClient:
    """Serves pre-built pages via keyset (cursor) pagination -- each call's
    incoming cursor is the `next_cursor` the previous call returned, and the
    stub returns next_cursor=None once its pages are exhausted, mirroring live
    Gamma keyset semantics (see PolymarketClient.fetch_markets_keyset). No
    network access -- `fetch_markets_keyset` is the only method
    load_from_gamma calls on the client."""

    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)
        self.calls = []

    def fetch_markets_keyset(self, cursor=None, limit: int = 100,
                             end_date_min=None, end_date_max=None):
        self.calls.append(cursor)
        if not self._pages:
            return [], None
        page = self._pages.pop(0)
        next_cursor = f"cursor{len(self.calls)}" if self._pages else None
        return page, next_cursor


def test_load_from_gamma_paginates_and_classifies_with_stub_client():
    pages = [
        [GAMMA_VALID_RESOLVED, GAMMA_NOT_BINARY],
        [GAMMA_NOT_RESOLVED, GAMMA_BAD_OUTCOME, GAMMA_MISSING_FIELD],
    ]
    client = _StubClient(pages)
    records, exclusions = load_from_gamma(client)

    assert len(records) == 1
    assert records[0].resolved_outcome == "NO"
    reasons = sorted(e.reason for e in exclusions)
    # GAMMA_MISSING_FIELD now excludes as "not_binary" (its Over/Under
    # outcomes), not "missing_field:category" -- null category alone no
    # longer excludes (Task 13 fix, see data/dataset_loader.py).
    assert reasons == ["bad_outcome", "not_binary", "not_binary", "not_resolved"]
    # cursor threaded through: None (page 1), then "cursor1" (page 2, whose
    # next_cursor is None -- pages exhausted -- so the loop stops there)
    assert client.calls == [None, "cursor1"]


def test_load_from_gamma_stops_on_empty_page():
    client = _StubClient([[GAMMA_VALID_RESOLVED]])
    records, exclusions = load_from_gamma(client)
    assert len(records) == 1
    assert exclusions == []
    # single page whose own next_cursor is None -- one call suffices, no
    # wasted extra call to discover the end (an efficiency improvement over
    # the old offset-based always-one-more-call pattern)
    assert client.calls == [None]


class _KwargsSpyClient:
    """Records the kwargs each fetch_markets_keyset call received; serves one
    page then stops."""

    def __init__(self, page):
        self._page = page
        self.kwargs_seen = []

    def fetch_markets_keyset(self, cursor=None, limit=100, end_date_min=None, end_date_max=None):
        self.kwargs_seen.append({"cursor": cursor, "end_date_min": end_date_min,
                                 "end_date_max": end_date_max})
        if self._page is None:
            return [], None
        page, self._page = self._page, None
        return page, None


def test_load_from_gamma_forwards_date_bounds_to_client():
    client = _KwargsSpyClient([GAMMA_VALID_RESOLVED])
    load_from_gamma(client, end_date_min="2023-01-01", end_date_max="2023-06-01")
    assert client.kwargs_seen[0]["end_date_min"] == "2023-01-01"
    assert client.kwargs_seen[0]["end_date_max"] == "2023-06-01"


def test_load_from_gamma_excludes_records_outside_study_window():
    # GAMMA_VALID_RESOLVED is a real market resolved 2021-01-02 (closedTime
    # "2021-01-02 21:35:34+00") -- well before a 2023+ study window. Gamma's
    # server-side end_date_min/max filtering should normally prevent this
    # from even being fetched, but the client-side closedTime check is a
    # drift guard (endDate != closedTime is possible) that must still catch
    # it and log it as an exclusion rather than silently dropping it.
    client = _KwargsSpyClient([GAMMA_VALID_RESOLVED])
    records, exclusions = load_from_gamma(
        client, end_date_min="2023-01-01", end_date_max="2023-06-01")
    assert records == []
    assert len(exclusions) == 1
    assert exclusions[0].reason == "outside_study_window"
