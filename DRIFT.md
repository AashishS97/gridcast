# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-22 15:46 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.64 | [-2, 1.23] | green | 2026-10-19 to 2026-11-15; percentile 14% of 221 same-season reference windows (~8 independent) |
| support | temperature_2m | 0.0104 |  | green | 2026-10-19 to 2026-11-15; 7 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 9.74 | [-15.7, 6.72] | amber | 2026-10-19 to 2026-11-15; percentile 98% of 221 same-season reference windows (~8 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-19 to 2026-11-15; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -3.32 | [-6.54, 4.44] | green | 2026-10-19 to 2026-11-15; percentile 14% of 221 same-season reference windows (~8 independent) |
| support | wind_speed_10m | 0.0699 |  | amber | 2026-10-19 to 2026-11-15; 47 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 462.9 | [-560.7, 639.7] | green | 2026-09-26 to 2026-10-23; percentile 84% of 214 same-season reference windows (~8 independent) |
| support | load_mw | 0.0893 |  | amber | 2026-09-26 to 2026-10-23; 60 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0725 | [0.029, 0.134] | green | nominal 10%; 23 origins |
| tail_rate | above p90 | 0.118 | [0.0598, 0.183] | green | nominal 10%; 23 origins |
| median_error | raw p50 | 168.1 | [118.7, 216.4] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 5 | [27.0, 28.0] | info | 5 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +12.6% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.04 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.84 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

