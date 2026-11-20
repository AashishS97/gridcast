# GridCast daily shadow log

Written by `.github/workflows/daily.yml` on `main`. Forecasts are retrospective
(~30 days behind): ENTSO-E NL actual load is provisional for ~28 days, so the
serving gates refuse anything newer. Each forecast is scored once its own
target day is final. Scores are of the served, conformally calibrated forecast.

## Latest run

- run at: 2026-11-20 14:55 UTC, status `ok`
- artifact: `lgbm_quantile_2026-10-01`
- data vintage: 2026-11-20 13:00 UTC (provisional from 2026-10-22 13:00)
- servable origins: 22, newest 2026-10-22
- this run: 0 new forecast(s), 0 new score(s)

## Shadow evaluation so far

| metric | value |
|---|---|
| origins scored | 21 (2026-10-01 to 2026-10-21) |
| MAE p50 (MW) | 230.2 |
| mean pinball (MW) | 79.6 |
| 80% interval coverage | 82.7% (target 80%) |
| mean interval width (MW) | 882 |

These are the served (calibrated) quantiles. For comparisons with the backtest
and calibration-pool baselines (which are raw, pre-calibration), rolling windows
with uncertainty bands, and the per-hour breakdown, see ACCURACY.md.

Every origin weighs equally (24 hours each). A few dozen origins is a small
sample: read coverage with a margin of several points.
