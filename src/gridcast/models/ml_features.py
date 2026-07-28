"""Design matrix for the direct multi-horizon LightGBM model.

Golden rule: ONE builder, parameterized by forecast origins, generates both
training rows (by replaying historical origins) and inference rows. Every
feature for a row (origin, h) uses only:
  - target-series data strictly before the origin (lags, rolling stats),
  - deterministic calendar functions of the target timestamp,
  - exogenous series legitimately available at the origin (weather under the
    stated perfect-prognosis caveat; the TSO day-ahead forecast, published
    ~noon D-1, before our midnight origin).
Because training and serving share this function, train/serve mismatch is
impossible by construction.

Lag availability: target t = origin + (h-1)h; lag L references t - L, which
is observed iff h <= L. Unavailable lags become NaN (LightGBM handles NaN
natively), in training AND inference alike — consistent, therefore learnable.
With the default lags (>= 24) masking never triggers at a 24h horizon, but
the rule is enforced generically so adding short lags later stays safe.

Rolling stats are ORIGIN-anchored ("state of the system when the forecast
was made"), ending at origin - 1h — they cannot touch the forecast window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")

DEFAULT_LAGS = (24, 48, 168, 336)
N_HORIZONS = 24

CALENDAR_COLS = ["hour", "dow", "is_weekend", "month", "day_of_year", "is_holiday"]
WEATHER_COLS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "temp_mean_24h"]
EXTERNAL_FORECAST_COL = "da_forecast_mw"
TARGET_COL = "load_mw"


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("timestamp")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted, unique timestamps")
    return frame


def build_design_matrix(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Long-format matrix: one row per (origin, horizon).

    Columns: origin, timestamp, horizon, features..., y_true.
    y_true is NaN for genuinely future timestamps (inference rows) — the
    builder does not care whether it is producing training or serving data.
    """
    frame = _indexed(frame)
    y = frame[TARGET_COL].astype(float)

    # --- origin-anchored state (one value per origin, shared by all 24 rows).
    # rolling(...) at position origin-1h covers [origin-Wh, origin-1h]:
    # strictly pre-origin by construction.
    anchor = origins - HOUR
    origin_feats = pd.DataFrame(
        {
            "origin": origins,
            "last_obs": y.reindex(anchor).to_numpy(),
            "roll24_mean": y.rolling(24).mean().reindex(anchor).to_numpy(),
            "roll24_min": y.rolling(24).min().reindex(anchor).to_numpy(),
            "roll24_max": y.rolling(24).max().reindex(anchor).to_numpy(),
            "roll168_mean": y.rolling(168).mean().reindex(anchor).to_numpy(),
        }
    )

    parts: list[pd.DataFrame] = []
    for h in range(1, N_HORIZONS + 1):
        ts = origins + (h - 1) * HOUR
        part = pd.DataFrame({"origin": origins, "timestamp": ts})
        part["horizon"] = np.int16(h)

        for lag in lags:
            vals = y.reindex(ts - lag * HOUR).to_numpy()
            if h > lag:  # references the forecast window -> not observed yet
                vals = np.full(len(ts), np.nan)
            part[f"lag_{lag}"] = vals

        # target-time joins: calendar (deterministic), weather (perfect-prog
        # caveat), TSO day-ahead forecast (published before our origin).
        target_side = frame.reindex(ts)
        for col in CALENDAR_COLS + WEATHER_COLS + [EXTERNAL_FORECAST_COL]:
            part[col] = target_side[col].to_numpy()

        part["y_true"] = y.reindex(ts).to_numpy()
        parts.append(part)

    matrix = pd.concat(parts, ignore_index=True)
    matrix = matrix.merge(origin_feats, on="origin", how="left")
    return matrix.sort_values(["origin", "horizon"]).reset_index(drop=True)


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and the target."""
    return [c for c in matrix.columns if c not in ("origin", "timestamp", "y_true")]


def daily_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every midnight in the span of `index` usable as a replayed origin."""
    first = index[0].ceil("D")
    last = index[-1].floor("D")
    return pd.date_range(first, last, freq="1D", tz=index.tz)
