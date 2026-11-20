# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-20 15:35 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.53 | [-2, 1.22] | green | 2026-10-17 to 2026-11-13; percentile 19% of 220 same-season reference windows (~8 independent) |
| support | temperature_2m | 0.0104 |  | green | 2026-10-17 to 2026-11-13; 7 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 9.99 | [-15.7, 6.73] | amber | 2026-10-17 to 2026-11-13; percentile 98% of 220 same-season reference windows (~8 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-17 to 2026-11-13; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -3.49 | [-6.55, 3.82] | green | 2026-10-17 to 2026-11-13; percentile 12% of 220 same-season reference windows (~8 independent) |
| support | wind_speed_10m | 0.0699 |  | amber | 2026-10-17 to 2026-11-13; 47 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 392.7 | [-561.8, 616.7] | green | 2026-09-24 to 2026-10-21; percentile 84% of 212 same-season reference windows (~8 independent) |
| support | load_mw | 0.0536 |  | amber | 2026-09-24 to 2026-10-21; 36 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0774 | [0.0298, 0.143] | green | nominal 10%; 21 origins |
| tail_rate | above p90 | 0.0952 | [0.0417, 0.159] | green | nominal 10%; 21 origins |
| median_error | raw p50 | 161.7 | [114.3, 214.7] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 1 | [28.0, 28.0] | info | 1 day(s) revised by > 5% in the ledger; ages at revision 28-28 days, median change +11.0% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.04 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.84 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

