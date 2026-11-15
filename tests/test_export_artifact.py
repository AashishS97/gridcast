"""Serving export: config contract, masking, calibration window, model
serialization, and a cutoff canary against the REAL training code."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from gridcast.models.exclusions import Exclusion
from gridcast.models.export_artifact import (
    calibration_origins,
    load_config,
    mask_from,
    save_models,
)
from gridcast.models.ml_features import WEATHER_COLS, build_design_matrix, daily_origins
from gridcast.models.quantile_lgbm import fit_quantile_models

CUTOFF = pd.Timestamp("2024-10-01", tz="UTC")
Q50 = {"q50": 0.5}


def _frame(days: int = 300) -> pd.DataFrame:
    """Synthetic hourly table with the real schema: daily and weekly shape,
    temperature dependence, noise."""
    ts = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    n = len(ts)
    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    weekend = np.asarray(ts.dayofweek >= 5)
    temp = 10 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "hour": hour.astype("int8"),
            "dow": ts.dayofweek.to_numpy().astype("int8"),
            "is_weekend": weekend.astype("int8"),
            "month": ts.month.to_numpy().astype("int8"),
            "day_of_year": doy.astype("int16"),
            "is_holiday": np.zeros(n, dtype="int8"),
            "temperature_2m": temp,
            "shortwave_radiation": np.clip(600 * np.sin(np.pi * (hour - 6) / 12), 0, None),
            "wind_speed_10m": rng.gamma(2.0, 3.0, n),
        }
    )
    df["temp_mean_24h"] = df["temperature_2m"].rolling(24, min_periods=1).mean()
    load = (
        12000
        + 1500 * np.sin(2 * np.pi * (hour - 8) / 24)
        - 120 * (temp - 10)
        - 800 * weekend
        + rng.normal(0, 150, n)
    )
    df["load_mw"] = load
    df["da_forecast_mw"] = load * 0.8
    return df


def _origins(frame: pd.DataFrame) -> pd.DatetimeIndex:
    o = daily_origins(pd.DatetimeIndex(frame["timestamp"]))
    return o[np.asarray(o <= CUTOFF)]


def _write(tmp_path, cutoff: str):
    p = tmp_path / "serving.toml"
    p.write_text(
        "[artifact]\n"
        f'train_cutoff = "{cutoff}"\n'
        "calibration_window_days = 60\n"
        "min_calibration_origins = 6\n"
        'output_root = "models"\n',
        encoding="utf-8",
    )
    return p


def test_load_config(tmp_path):
    cfg = load_config(_write(tmp_path, "2026-10-01T00:00:00Z"))
    assert cfg.train_cutoff == pd.Timestamp("2026-10-01", tz="UTC")
    assert cfg.calibration_window == pd.Timedelta(days=60)
    assert cfg.min_calibration_origins == 6
    assert cfg.name == "lgbm_quantile_2026-10-01"
    assert cfg.output_root.is_absolute()


@pytest.mark.parametrize("cutoff", ["2026-10-01T00:00:00", "2026-10-01T05:00:00Z"])
def test_load_config_rejects_bad_cutoff(tmp_path, cutoff):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, cutoff))


def test_repo_serving_config_loads():
    cfg = load_config()
    assert cfg.train_cutoff == cfg.train_cutoff.floor("D")


def test_mask_from_masks_only_at_and_after_cutoff():
    frame = _frame()
    out = mask_from(frame, CUTOFF)
    late = np.asarray(pd.DatetimeIndex(frame["timestamp"]) >= CUTOFF)
    assert len(out) == len(frame)
    assert out.loc[late, "load_mw"].isna().all()
    pd.testing.assert_series_equal(out.loc[~late, "load_mw"], frame.loc[~late, "load_mw"])
    assert frame["load_mw"].notna().all(), "input must not be modified"


def test_calibration_origins_window_and_exclusions():
    origins = pd.date_range("2026-07-01", "2026-10-01", freq="1D", tz="UTC")
    cutoff = pd.Timestamp("2026-10-01", tz="UTC")
    window = pd.Timedelta(days=60)
    clean = calibration_origins(origins, cutoff, window, [])
    assert clean[0] == pd.Timestamp("2026-08-02", tz="UTC")
    assert clean[-1] == pd.Timestamp("2026-09-30", tz="UTC")
    assert len(clean) == 60
    ex = Exclusion(
        pd.Timestamp("2026-08-20", tz="UTC"), pd.Timestamp("2026-08-21", tz="UTC"), "r", "e"
    )
    kept = calibration_origins(origins, cutoff, window, [ex])
    # [o - 336h, o + 24h) overlaps [Aug 20, Aug 21) for o in Aug 20 .. Sep 3
    assert len(kept) == 45
    gone = clean.difference(kept)
    assert gone[0] == pd.Timestamp("2026-08-20", tz="UTC")
    assert gone[-1] == pd.Timestamp("2026-09-03", tz="UTC")


def test_saved_models_roundtrip(tmp_path):
    frame = _frame()
    matrix = build_design_matrix(mask_from(frame, CUTOFF), _origins(frame))
    models, feats = fit_quantile_models(matrix, CUTOFF, quantiles=Q50)
    paths = save_models(models, tmp_path)
    booster = lgb.Booster(model_file=str(paths["q50"]))
    x = matrix[feats]
    expected = models["q50"].predict(x, num_iteration=models["q50"].best_iteration_)
    np.testing.assert_allclose(booster.predict(x), expected, rtol=0, atol=1e-6)


def test_training_ignores_everything_at_or_after_cutoff():
    """Canary: corrupt every time-varying input from the cutoff on. If any
    of it leaks into training through any feature path, predictions move."""
    frame = _frame()
    corrupted = frame.copy()
    late = np.asarray(pd.DatetimeIndex(corrupted["timestamp"]) >= CUTOFF)
    for col in ["load_mw", "da_forecast_mw", *WEATHER_COLS]:
        corrupted.loc[late, col] = corrupted.loc[late, col] * 3 + 1000

    x_matrix = build_design_matrix(mask_from(frame, CUTOFF), _origins(frame))
    x_matrix = x_matrix[x_matrix["origin"] < CUTOFF]
    preds = []
    for f in (frame, corrupted):
        matrix = build_design_matrix(mask_from(f, CUTOFF), _origins(f))
        models, feats = fit_quantile_models(matrix, CUTOFF, quantiles=Q50)
        m = models["q50"]
        preds.append(m.predict(x_matrix[feats], num_iteration=m.best_iteration_))
    np.testing.assert_allclose(preds[0], preds[1], rtol=0, atol=1e-9)
