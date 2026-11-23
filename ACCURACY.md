# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-23 15:58 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 24 origins (2026-10-01 to 2026-10-24)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 80.4% | [73.8%, 86.5%] | green |
| p50 MAE (MW) | 257.0 | [215.9, 302.7] | |
| seasonal-naive MAE (MW) | 527.5 | | |
| skill vs naive (1 - MAE ratio) | 0.513 | [0.396, 0.595] | green |
| mean interval width (MW) | 886 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 299.9 | [253.6, 351.0] | 246.6 | 256.9 |
| 80% coverage | 62.2% | [53.3%, 70.7%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 24 of 24 origins

## Per hour, all 24 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 165 | [114, 226] | 79% | [62%, 96%] |
| 2 | 01 | 221 | [159, 299] | 79% | [62%, 96%] |
| 3 | 02 | 189 | [122, 270] | 88% | [75%, 100%] |
| 4 | 03 | 149 | [93, 217] | 88% | [71%, 100%] |
| 5 | 04 | 160 | [108, 219] | 92% | [79%, 100%] |
| 6 | 05 | 159 | [119, 200] | 92% | [79%, 100%] |
| 7 | 06 | 198 | [141, 258] | 75% | [58%, 92%] |
| 8 | 07 | 333 | [231, 440] | 62% | [42%, 79%] |
| 9 | 08 | 379 | [277, 487] | 71% | [50%, 88%] |
| 10 | 09 | 315 | [215, 436] | 75% | [58%, 92%] |
| 11 | 10 | 376 | [247, 526] | 83% | [67%, 96%] |
| 12 | 11 | 510 | [349, 699] | 71% | [50%, 88%] |
| 13 | 12 | 384 | [274, 501] | 75% | [58%, 92%] |
| 14 | 13 | 368 | [264, 484] | 79% | [62%, 92%] |
| 15 | 14 | 430 | [324, 544] | 71% | [50%, 88%] |
| 16 | 15 | 446 | [339, 563] | 58% | [38%, 79%] |
| 17 | 16 | 232 | [171, 295] | 75% | [58%, 92%] |
| 18 | 17 | 120 | [87, 154] | 96% | [88%, 100%] |
| 19 | 18 | 143 | [102, 186] | 96% | [88%, 100%] |
| 20 | 19 | 122 | [90, 159] | 96% | [88%, 100%] |
| 21 | 20 | 124 | [91, 162] | 92% | [79%, 100%] |
| 22 | 21 | 142 | [100, 188] | 92% | [79%, 100%] |
| 23 | 22 | 269 | [196, 353] | 75% | [58%, 92%] |
| 24 | 23 | 235 | [160, 320] | 71% | [50%, 88%] |
