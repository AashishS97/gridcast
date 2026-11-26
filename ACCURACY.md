# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-26 06:40 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 27 origins (2026-10-01 to 2026-10-27)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 79.6% | [73.6%, 85.2%] | green |
| p50 MAE (MW) | 269.9 | [227.9, 315.1] | |
| seasonal-naive MAE (MW) | 563.3 | | |
| skill vs naive (1 - MAE ratio) | 0.521 | [0.424, 0.593] | green |
| mean interval width (MW) | 888 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 311.1 | [264.7, 362.2] | 246.6 | 256.9 |
| 80% coverage | 61.1% | [52.9%, 69.0%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 27 of 27 origins

## Per hour, all 27 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 295 | [142, 539] | 74% | [56%, 89%] |
| 2 | 01 | 209 | [150, 281] | 81% | [67%, 96%] |
| 3 | 02 | 199 | [138, 274] | 85% | [70%, 96%] |
| 4 | 03 | 139 | [87, 200] | 89% | [74%, 100%] |
| 5 | 04 | 167 | [119, 221] | 93% | [81%, 100%] |
| 6 | 05 | 190 | [131, 265] | 85% | [70%, 96%] |
| 7 | 06 | 205 | [152, 262] | 70% | [52%, 85%] |
| 8 | 07 | 319 | [226, 417] | 67% | [48%, 85%] |
| 9 | 08 | 356 | [262, 454] | 74% | [56%, 89%] |
| 10 | 09 | 309 | [217, 416] | 78% | [63%, 93%] |
| 11 | 10 | 395 | [273, 533] | 81% | [67%, 96%] |
| 12 | 11 | 561 | [403, 741] | 67% | [48%, 85%] |
| 13 | 12 | 456 | [331, 590] | 70% | [52%, 85%] |
| 14 | 13 | 366 | [260, 480] | 78% | [63%, 93%] |
| 15 | 14 | 410 | [314, 513] | 74% | [56%, 89%] |
| 16 | 15 | 411 | [309, 521] | 63% | [44%, 81%] |
| 17 | 16 | 238 | [176, 302] | 74% | [56%, 89%] |
| 18 | 17 | 131 | [97, 167] | 93% | [81%, 100%] |
| 19 | 18 | 159 | [114, 209] | 93% | [81%, 100%] |
| 20 | 19 | 140 | [101, 185] | 96% | [89%, 100%] |
| 21 | 20 | 142 | [101, 189] | 89% | [78%, 100%] |
| 22 | 21 | 167 | [113, 231] | 89% | [78%, 100%] |
| 23 | 22 | 277 | [211, 353] | 74% | [56%, 89%] |
| 24 | 23 | 237 | [168, 312] | 74% | [56%, 89%] |
