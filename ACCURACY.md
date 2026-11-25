# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-25 18:01 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 26 origins (2026-10-01 to 2026-10-26)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 79.6% | [73.4%, 85.4%] | green |
| p50 MAE (MW) | 271.3 | [228.0, 318.6] | |
| seasonal-naive MAE (MW) | 555.5 | | |
| skill vs naive (1 - MAE ratio) | 0.512 | [0.405, 0.589] | green |
| mean interval width (MW) | 893 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 314.8 | [267.8, 367.3] | 246.6 | 256.9 |
| 80% coverage | 60.9% | [52.7%, 68.9%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 26 of 26 origins

## Per hour, all 26 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 298 | [140, 550] | 73% | [54%, 88%] |
| 2 | 01 | 215 | [154, 287] | 81% | [65%, 96%] |
| 3 | 02 | 193 | [130, 269] | 85% | [69%, 96%] |
| 4 | 03 | 141 | [88, 205] | 88% | [77%, 100%] |
| 5 | 04 | 161 | [113, 217] | 92% | [81%, 100%] |
| 6 | 05 | 184 | [124, 261] | 88% | [77%, 100%] |
| 7 | 06 | 206 | [150, 264] | 73% | [54%, 88%] |
| 8 | 07 | 329 | [234, 430] | 65% | [46%, 85%] |
| 9 | 08 | 360 | [259, 462] | 73% | [58%, 88%] |
| 10 | 09 | 315 | [220, 421] | 77% | [62%, 92%] |
| 11 | 10 | 391 | [269, 532] | 85% | [69%, 96%] |
| 12 | 11 | 544 | [385, 730] | 69% | [50%, 85%] |
| 13 | 12 | 441 | [315, 578] | 73% | [54%, 88%] |
| 14 | 13 | 378 | [270, 494] | 77% | [62%, 92%] |
| 15 | 14 | 419 | [320, 525] | 73% | [54%, 88%] |
| 16 | 15 | 423 | [320, 537] | 62% | [42%, 81%] |
| 17 | 16 | 247 | [184, 311] | 73% | [54%, 88%] |
| 18 | 17 | 132 | [97, 169] | 92% | [81%, 100%] |
| 19 | 18 | 160 | [114, 214] | 92% | [81%, 100%] |
| 20 | 19 | 143 | [103, 189] | 96% | [88%, 100%] |
| 21 | 20 | 147 | [104, 193] | 88% | [73%, 100%] |
| 22 | 21 | 168 | [114, 236] | 88% | [77%, 100%] |
| 23 | 22 | 280 | [212, 357] | 73% | [54%, 88%] |
| 24 | 23 | 235 | [164, 314] | 73% | [54%, 88%] |
