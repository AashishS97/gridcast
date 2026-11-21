# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-21 15:39 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 22 origins (2026-10-01 to 2026-10-22)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 81.6% | [74.8%, 88.1%] | green |
| p50 MAE (MW) | 243.7 | [202.8, 290.3] | |
| seasonal-naive MAE (MW) | 477.8 | | |
| skill vs naive (1 - MAE ratio) | 0.490 | [0.349, 0.592] | green |
| mean interval width (MW) | 879 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 283.7 | [239.0, 335.1] | 246.6 | 256.9 |
| 80% coverage | 63.6% | [54.4%, 72.5%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 22 of 22 origins

## Per hour, all 22 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 137 | [97, 182] | 82% | [64%, 95%] |
| 2 | 01 | 190 | [141, 242] | 82% | [64%, 95%] |
| 3 | 02 | 172 | [106, 256] | 91% | [77%, 100%] |
| 4 | 03 | 152 | [90, 225] | 86% | [73%, 100%] |
| 5 | 04 | 142 | [96, 196] | 95% | [86%, 100%] |
| 6 | 05 | 156 | [113, 199] | 91% | [77%, 100%] |
| 7 | 06 | 186 | [131, 247] | 73% | [55%, 91%] |
| 8 | 07 | 346 | [236, 460] | 59% | [41%, 77%] |
| 9 | 08 | 393 | [283, 506] | 68% | [50%, 86%] |
| 10 | 09 | 314 | [202, 443] | 73% | [55%, 91%] |
| 11 | 10 | 327 | [206, 478] | 82% | [64%, 95%] |
| 12 | 11 | 436 | [299, 589] | 73% | [55%, 91%] |
| 13 | 12 | 359 | [246, 478] | 77% | [59%, 95%] |
| 14 | 13 | 356 | [249, 475] | 82% | [64%, 95%] |
| 15 | 14 | 407 | [297, 525] | 73% | [55%, 91%] |
| 16 | 15 | 414 | [307, 529] | 64% | [45%, 82%] |
| 17 | 16 | 222 | [157, 287] | 82% | [64%, 95%] |
| 18 | 17 | 124 | [90, 160] | 95% | [86%, 100%] |
| 19 | 18 | 143 | [100, 189] | 95% | [86%, 100%] |
| 20 | 19 | 115 | [81, 151] | 95% | [86%, 100%] |
| 21 | 20 | 129 | [93, 167] | 91% | [77%, 100%] |
| 22 | 21 | 130 | [87, 178] | 95% | [86%, 100%] |
| 23 | 22 | 263 | [185, 353] | 82% | [64%, 95%] |
| 24 | 23 | 235 | [155, 323] | 73% | [55%, 91%] |
