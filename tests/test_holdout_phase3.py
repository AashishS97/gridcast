"""Test the calibration-pool rule: trailing window, and a forecast is usable
only once its whole target day has been observed."""

import pandas as pd
from gridcast.models.run_holdout_phase3 import calibration_pool


def test_calibration_pool_window_and_closure():
    origins = pd.date_range("2025-01-01", "2025-01-10", freq="1D", tz="UTC")
    res = pd.DataFrame({"origin": origins.repeat(24)})
    pool = calibration_pool(res, pd.Timestamp("2025-01-10", tz="UTC"), window=pd.Timedelta("5D"))
    got = sorted(pool["origin"].unique())
    expected = list(pd.date_range("2025-01-05", "2025-01-09", freq="1D", tz="UTC"))
    assert got == expected  # Jan 4 too old; Jan 10 not yet observed
