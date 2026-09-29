"""Tests for data exclusions: masking, config parsing, and the soundness of
the contamination rule against the REAL design-matrix builder."""

import numpy as np
import pandas as pd
import pytest
from gridcast.models.exclusions import (
    LOOKBACK_HOURS,
    Exclusion,
    apply_exclusions,
    contaminated_origins,
    load_exclusions,
)
from gridcast.models.ml_features import (
    CALENDAR_COLS,
    EXTERNAL_FORECAST_COL,
    TARGET_COL,
    WEATHER_COLS,
    build_design_matrix,
    daily_origins,
)

HOUR = pd.Timedelta("1h")
LOAD_DERIVED = ["last_obs", "roll24_mean", "roll24_min", "roll24_max", "roll168_mean", "y_true"]

EXCL = Exclusion(
    start=pd.Timestamp("2025-01-31", tz="UTC"),
    end=pd.Timestamp("2025-02-02", tz="UTC"),
    reason="test",
    evidence="test",
)


def _frame(days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="1h", tz="UTC", name="timestamp")
    df = pd.DataFrame(index=idx)
    for col in CALENDAR_COLS + WEATHER_COLS + [EXTERNAL_FORECAST_COL]:
        df[col] = 0.0
    hours = np.arange(len(idx))
    df[TARGET_COL] = 12000.0 + 2000.0 * np.sin(2 * np.pi * hours / 24)
    return df


def test_masks_exactly_the_period_and_does_not_mutate_input():
    df = _frame()
    out = apply_exclusions(df, [EXCL])
    inside = (df.index >= EXCL.start) & (df.index < EXCL.end)
    assert out.loc[inside, TARGET_COL].isna().all()
    assert out.loc[~inside, TARGET_COL].notna().all()
    assert len(out) == len(df)  # grid kept intact, nothing dropped
    assert df[TARGET_COL].notna().all()  # input untouched


def test_open_ended_exclusion_masks_to_the_end():
    df = _frame()
    ex = Exclusion(pd.Timestamp("2025-02-20", tz="UTC"), None, "t", "t")
    out = apply_exclusions(df, [ex])
    assert out.loc[df.index >= ex.start, TARGET_COL].isna().all()
    assert out.loc[df.index < ex.start, TARGET_COL].notna().all()


def test_clean_origins_have_no_masked_inputs():
    """Soundness: an origin NOT flagged contaminated must have no NaN in any
    load-derived column of the real design matrix."""
    df = apply_exclusions(_frame(), [EXCL])
    origins = daily_origins(df.index)
    origins = origins[origins >= df.index[0] + LOOKBACK_HOURS * HOUR]  # full history
    origins = origins[origins + 24 * HOUR <= df.index[-1] + HOUR]  # labels fit

    matrix = build_design_matrix(df, origins)
    cols = [c for c in matrix.columns if c.startswith("lag_")] + LOAD_DERIVED
    has_nan = matrix[cols].isna().any(axis=1).groupby(matrix["origin"]).any()
    has_nan = has_nan.reindex(origins)
    flag = pd.Series(contaminated_origins(origins, [EXCL]), index=origins)

    assert not (has_nan & ~flag).any(), "an origin marked clean reads masked data"
    assert (has_nan & flag).any(), "masking never reached the design matrix"
    assert (~flag).any(), "no clean origins left to score"


def test_overlap_boundaries_are_half_open():
    # read window [o - 336h, o + 24h); exclusion [start, end)
    end_touch = EXCL.end + LOOKBACK_HOURS * HOUR  # window starts exactly at end
    start_touch = EXCL.start - 24 * HOUR  # window ends exactly at start
    origins = pd.DatetimeIndex([end_touch, end_touch - HOUR, start_touch, start_touch + HOUR])
    assert contaminated_origins(origins, [EXCL]).tolist() == [False, True, False, True]


def test_load_exclusions_parses_and_handles_bom(tmp_path):
    body = (
        '[[exclusion]]\nstart = "2026-07-01T00:00:00Z"\nend = "2026-08-01T00:00:00Z"\n'
        'reason = "r"\nevidence = "e"\n\n'
        '[[exclusion]]\nstart = "2026-09-01T00:00:00Z"\nreason = "r2"\nevidence = "e2"\n'
    )
    path = tmp_path / "ex.toml"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))  # with BOM
    ex = load_exclusions(path)
    assert len(ex) == 2
    assert ex[0].end == pd.Timestamp("2026-08-01", tz="UTC")
    assert ex[1].end is None
    assert str(ex[0].start.tz) == "UTC"


@pytest.mark.parametrize(
    "body",
    [
        '[[exclusion]]\nstart = "2026-07-01T00:00:00"\nreason = "r"\nevidence = "e"\n',
        '[[exclusion]]\nstart = "2026-08-01T00:00:00Z"\nend = "2026-07-01T00:00:00Z"\n'
        'reason = "r"\nevidence = "e"\n',
        '[[exclusion]]\nstart = "2026-07-01T00:00:00Z"\nreason = "r"\n',
    ],
    ids=["naive-timestamp", "end-before-start", "missing-evidence"],
)
def test_load_exclusions_rejects_bad_config(tmp_path, body):
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_exclusions(path)


def test_repo_config_loads():
    ex = load_exclusions()
    assert len(ex) >= 2
    assert all(e.reason and e.evidence for e in ex)
