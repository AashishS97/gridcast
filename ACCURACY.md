# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-22 15:46 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 23 origins (2026-10-01 to 2026-10-23)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 81.0% | [74.1%, 87.3%] | green |
| p50 MAE (MW) | 250.6 | [209.3, 295.9] | |
| seasonal-naive MAE (MW) | 496.0 | | |
| skill vs naive (1 - MAE ratio) | 0.495 | [0.370, 0.587] | green |
| mean interval width (MW) | 879 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 290.5 | [245.7, 341.2] | 246.6 | 256.9 |
| 80% coverage | 63.0% | [54.0%, 71.7%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 23 of 23 origins

## Per hour, all 23 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 144 | [104, 191] | 83% | [65%, 96%] |
| 2 | 01 | 194 | [146, 245] | 83% | [65%, 96%] |
| 3 | 02 | 174 | [112, 255] | 91% | [78%, 100%] |
| 4 | 03 | 147 | [87, 218] | 87% | [74%, 100%] |
| 5 | 04 | 144 | [99, 195] | 96% | [87%, 100%] |
| 6 | 05 | 154 | [112, 195] | 91% | [78%, 100%] |
| 7 | 06 | 185 | [133, 242] | 74% | [57%, 91%] |
| 8 | 07 | 333 | [226, 444] | 61% | [39%, 78%] |
| 9 | 08 | 384 | [280, 495] | 70% | [52%, 87%] |
| 10 | 09 | 314 | [210, 435] | 74% | [57%, 91%] |
| 11 | 10 | 355 | [225, 512] | 83% | [65%, 96%] |
| 12 | 11 | 500 | [333, 698] | 70% | [52%, 87%] |
| 13 | 12 | 381 | [266, 503] | 74% | [57%, 91%] |
| 14 | 13 | 377 | [266, 494] | 78% | [61%, 96%] |
| 15 | 14 | 421 | [310, 536] | 70% | [52%, 87%] |
| 16 | 15 | 433 | [326, 552] | 61% | [39%, 83%] |
| 17 | 16 | 228 | [166, 294] | 78% | [61%, 96%] |
| 18 | 17 | 119 | [85, 154] | 96% | [87%, 100%] |
| 19 | 18 | 139 | [99, 183] | 96% | [87%, 100%] |
| 20 | 19 | 116 | [84, 150] | 96% | [87%, 100%] |
| 21 | 20 | 127 | [93, 166] | 91% | [78%, 100%] |
| 22 | 21 | 134 | [94, 180] | 96% | [87%, 100%] |
| 23 | 22 | 268 | [193, 353] | 78% | [61%, 96%] |
| 24 | 23 | 244 | [166, 330] | 70% | [52%, 87%] |
