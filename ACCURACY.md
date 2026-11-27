# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-27 17:32 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 28 origins (2026-10-01 to 2026-10-28)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 80.2% | [74.3%, 85.6%] | green |
| p50 MAE (MW) | 266.9 | [226.8, 310.7] | |
| seasonal-naive MAE (MW) | 561.4 | | |
| skill vs naive (1 - MAE ratio) | 0.525 | [0.429, 0.596] | green |
| mean interval width (MW) | 891 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 307.5 | [262.0, 356.8] | 246.6 | 256.9 |
| 80% coverage | 61.9% | [54.0%, 69.5%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 28 of 28 origins

## Per hour, all 28 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 286 | [136, 522] | 75% | [57%, 89%] |
| 2 | 01 | 209 | [153, 278] | 82% | [68%, 96%] |
| 3 | 02 | 195 | [135, 265] | 86% | [71%, 96%] |
| 4 | 03 | 139 | [89, 199] | 89% | [75%, 100%] |
| 5 | 04 | 174 | [126, 228] | 93% | [82%, 100%] |
| 6 | 05 | 200 | [138, 273] | 82% | [68%, 96%] |
| 7 | 06 | 209 | [157, 264] | 71% | [54%, 86%] |
| 8 | 07 | 319 | [230, 415] | 68% | [50%, 86%] |
| 9 | 08 | 358 | [267, 455] | 75% | [57%, 89%] |
| 10 | 09 | 299 | [212, 403] | 79% | [64%, 93%] |
| 11 | 10 | 400 | [283, 532] | 82% | [68%, 96%] |
| 12 | 11 | 559 | [407, 735] | 68% | [50%, 86%] |
| 13 | 12 | 449 | [326, 578] | 71% | [54%, 86%] |
| 14 | 13 | 355 | [253, 466] | 79% | [64%, 93%] |
| 15 | 14 | 397 | [301, 498] | 75% | [57%, 89%] |
| 16 | 15 | 397 | [296, 506] | 64% | [46%, 82%] |
| 17 | 16 | 234 | [173, 297] | 75% | [57%, 89%] |
| 18 | 17 | 127 | [94, 163] | 93% | [82%, 100%] |
| 19 | 18 | 160 | [115, 209] | 93% | [82%, 100%] |
| 20 | 19 | 137 | [98, 181] | 96% | [89%, 100%] |
| 21 | 20 | 138 | [97, 183] | 89% | [75%, 100%] |
| 22 | 21 | 164 | [112, 227] | 89% | [75%, 100%] |
| 23 | 22 | 268 | [202, 343] | 75% | [57%, 89%] |
| 24 | 23 | 232 | [164, 306] | 75% | [57%, 89%] |
