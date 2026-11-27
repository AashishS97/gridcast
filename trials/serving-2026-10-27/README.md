# GridCast daily shadow log

Written by `.github/workflows/daily.yml` on `main`. Forecasts are retrospective
(~30 days behind): ENTSO-E NL actual load is provisional for ~28 days, so the
serving gates refuse anything newer. Each forecast is scored once its own
target day is final. Scores are of the served, conformally calibrated forecast.

## Latest run

- run at: 2026-11-27 17:32 UTC, status `ok`
- artifact: `lgbm_quantile_2026-10-27`
- data vintage: 2026-11-27 16:00 UTC (provisional from 2026-10-29 16:00)
- servable origins: 3, newest 2026-10-29
- this run: 1 new forecast(s), 1 new score(s)

## Shadow evaluation so far

| metric | value |
|---|---|
| origins scored | 2 (2026-10-27 to 2026-10-28) |
| MAE p50 (MW) | 241.5 |
| mean pinball (MW) | 80.5 |
| 80% interval coverage | 79.2% (target 80%) |
| mean interval width (MW) | 939 |

These are the served (calibrated) quantiles. For comparisons with the backtest
and calibration-pool baselines (which are raw, pre-calibration), rolling windows
with uncertainty bands, and the per-hour breakdown, see ACCURACY.md.

Every origin weighs equally (24 hours each). A few dozen origins is a small
sample: read coverage with a margin of several points.
