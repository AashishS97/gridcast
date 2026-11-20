# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-20 16:26 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 21 origins (2026-10-01 to 2026-10-21)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 82.7% | [75.8%, 89.1%] | green |
| p50 MAE (MW) | 230.2 | [196.2, 268.9] | |
| seasonal-naive MAE (MW) | 455.6 | | |
| skill vs naive (1 - MAE ratio) | 0.495 | [0.342, 0.600] | green |
| mean interval width (MW) | 882 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 269.7 | [231.5, 313.9] | 246.6 | 256.9 |
| 80% coverage | 64.7% | [55.0%, 74.0%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 21 of 21 origins

## Per hour, all 21 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 142 | [102, 189] | 81% | [62%, 95%] |
| 2 | 01 | 187 | [136, 242] | 81% | [62%, 95%] |
| 3 | 02 | 140 | [97, 190] | 95% | [86%, 100%] |
| 4 | 03 | 127 | [79, 182] | 90% | [76%, 100%] |
| 5 | 04 | 143 | [95, 199] | 95% | [86%, 100%] |
| 6 | 05 | 160 | [116, 203] | 90% | [76%, 100%] |
| 7 | 06 | 174 | [122, 235] | 76% | [57%, 95%] |
| 8 | 07 | 324 | [218, 431] | 62% | [43%, 81%] |
| 9 | 08 | 365 | [262, 468] | 71% | [52%, 90%] |
| 10 | 09 | 272 | [183, 368] | 76% | [57%, 95%] |
| 11 | 10 | 272 | [185, 370] | 86% | [71%, 100%] |
| 12 | 11 | 384 | [277, 497] | 76% | [57%, 95%] |
| 13 | 12 | 335 | [228, 451] | 81% | [62%, 95%] |
| 14 | 13 | 347 | [238, 469] | 81% | [62%, 95%] |
| 15 | 14 | 399 | [284, 521] | 71% | [52%, 90%] |
| 16 | 15 | 407 | [298, 528] | 62% | [43%, 81%] |
| 17 | 16 | 232 | [168, 298] | 81% | [62%, 95%] |
| 18 | 17 | 122 | [86, 159] | 95% | [86%, 100%] |
| 19 | 18 | 144 | [98, 191] | 95% | [86%, 100%] |
| 20 | 19 | 112 | [77, 150] | 95% | [86%, 100%] |
| 21 | 20 | 125 | [88, 164] | 90% | [76%, 100%] |
| 22 | 21 | 124 | [82, 174] | 95% | [86%, 100%] |
| 23 | 22 | 256 | [174, 351] | 86% | [71%, 100%] |
| 24 | 23 | 231 | [148, 322] | 71% | [52%, 90%] |
