# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-24 18:57 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 25 origins (2026-10-01 to 2026-10-25)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 79.8% | [73.2%, 85.8%] | green |
| p50 MAE (MW) | 266.4 | [222.3, 313.7] | |
| seasonal-naive MAE (MW) | 539.8 | | |
| skill vs naive (1 - MAE ratio) | 0.506 | [0.397, 0.589] | green |
| mean interval width (MW) | 890 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 310.1 | [262.5, 362.9] | 246.6 | 256.9 |
| 80% coverage | 61.7% | [53.2%, 69.8%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 25 of 25 origins

## Per hour, all 25 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 277 | [125, 528] | 76% | [60%, 92%] |
| 2 | 01 | 212 | [149, 290] | 80% | [64%, 92%] |
| 3 | 02 | 186 | [123, 266] | 88% | [76%, 100%] |
| 4 | 03 | 144 | [89, 212] | 88% | [72%, 100%] |
| 5 | 04 | 161 | [112, 219] | 92% | [80%, 100%] |
| 6 | 05 | 154 | [113, 196] | 92% | [80%, 100%] |
| 7 | 06 | 196 | [143, 255] | 76% | [60%, 92%] |
| 8 | 07 | 331 | [231, 432] | 64% | [44%, 84%] |
| 9 | 08 | 364 | [261, 471] | 72% | [52%, 88%] |
| 10 | 09 | 311 | [216, 421] | 76% | [60%, 92%] |
| 11 | 10 | 378 | [255, 524] | 84% | [68%, 96%] |
| 12 | 11 | 535 | [372, 720] | 68% | [48%, 84%] |
| 13 | 12 | 422 | [296, 557] | 72% | [52%, 88%] |
| 14 | 13 | 389 | [280, 505] | 76% | [60%, 92%] |
| 15 | 14 | 427 | [324, 536] | 72% | [52%, 88%] |
| 16 | 15 | 434 | [330, 550] | 60% | [40%, 80%] |
| 17 | 16 | 248 | [183, 316] | 72% | [52%, 88%] |
| 18 | 17 | 130 | [94, 170] | 92% | [80%, 100%] |
| 19 | 18 | 160 | [111, 215] | 92% | [80%, 100%] |
| 20 | 19 | 136 | [96, 182] | 96% | [88%, 100%] |
| 21 | 20 | 137 | [97, 182] | 92% | [80%, 100%] |
| 22 | 21 | 144 | [103, 190] | 92% | [80%, 100%] |
| 23 | 22 | 276 | [205, 355] | 72% | [52%, 88%] |
| 24 | 23 | 241 | [168, 323] | 72% | [52%, 88%] |
