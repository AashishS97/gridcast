# GridCast accuracy

Written by `gridcast.monitoring.accuracy` at 2026-11-28 16:17 UTC from
hourly.csv. Bands resample whole days (origins), not hours: a forecast's 24
hours fail together. Thresholds: `configs/monitoring.toml`. Bands are 95%.

## Trailing window: 28 origins (2026-10-02 to 2026-10-29)

| served forecast | value | band | zone |
|---|---|---|---|
| 80% interval coverage | 80.1% | [74.3%, 85.4%] | green |
| p50 MAE (MW) | 269.5 | [230.6, 312.8] | |
| seasonal-naive MAE (MW) | 574.0 | | |
| skill vs naive (1 - MAE ratio) | 0.531 | [0.439, 0.603] | green |
| mean interval width (MW) | 894 | | |

## Raw quantiles against the baselines

The pool and the backtest gate measure raw (pre-calibration) quantiles, so the
shadow is compared on its raw quantiles too. A baseline inside the shadow band
is consistent with it; outside, the live period differs from the reference.

| raw quantiles | shadow window | band | pool | backtest gate |
|---|---|---|---|---|
| p50 MAE (MW) | 309.1 | [265.3, 358.7] | 246.6 | 256.9 |
| 80% coverage | 61.8% | [53.9%, 69.3%] | 64.2% | 65.5% |

- pool: calibration pool (60 origins 2026-08-02 to 2026-09-30; out-of-sample, one retrain per origin, not a sealed holdout)
- backtest gate: backtest gate (14 pinned dev folds, daily retrain, an earlier period)
- shadow raw values available for 28 of 28 origins

## Per hour, all 29 scored origins

Horizon h is UTC hour h-1 (origins are midnight UTC): hour-of-day and
horizon effects are confounded. Per-hour bands rest on few days; read them
as where to look, not as findings.

| h | UTC | p50 MAE (MW) | band | coverage | band |
|---|---|---|---|---|---|
| 1 | 00 | 280 | [136, 514] | 76% | [59%, 90%] |
| 2 | 01 | 208 | [152, 275] | 83% | [69%, 97%] |
| 3 | 02 | 205 | [144, 275] | 83% | [69%, 97%] |
| 4 | 03 | 149 | [97, 210] | 86% | [72%, 97%] |
| 5 | 04 | 194 | [137, 261] | 90% | [76%, 100%] |
| 6 | 05 | 214 | [149, 290] | 79% | [66%, 93%] |
| 7 | 06 | 224 | [167, 287] | 69% | [52%, 86%] |
| 8 | 07 | 317 | [230, 409] | 69% | [52%, 86%] |
| 9 | 08 | 353 | [266, 448] | 76% | [59%, 90%] |
| 10 | 09 | 293 | [207, 393] | 79% | [66%, 93%] |
| 11 | 10 | 389 | [276, 518] | 83% | [69%, 97%] |
| 12 | 11 | 542 | [394, 712] | 69% | [52%, 86%] |
| 13 | 12 | 445 | [326, 570] | 72% | [55%, 86%] |
| 14 | 13 | 355 | [255, 465] | 79% | [62%, 93%] |
| 15 | 14 | 389 | [297, 488] | 76% | [59%, 90%] |
| 16 | 15 | 385 | [285, 493] | 66% | [48%, 83%] |
| 17 | 16 | 227 | [168, 289] | 76% | [59%, 90%] |
| 18 | 17 | 124 | [91, 160] | 93% | [83%, 100%] |
| 19 | 18 | 159 | [116, 206] | 93% | [83%, 100%] |
| 20 | 19 | 136 | [98, 177] | 97% | [90%, 100%] |
| 21 | 20 | 140 | [101, 183] | 90% | [76%, 100%] |
| 22 | 21 | 162 | [113, 225] | 90% | [76%, 100%] |
| 23 | 22 | 262 | [197, 336] | 76% | [59%, 90%] |
| 24 | 23 | 230 | [163, 301] | 76% | [59%, 90%] |
