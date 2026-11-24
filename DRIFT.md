# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-24 18:57 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.85 | [-2.29, 1.79] | green | 2026-10-21 to 2026-11-17; percentile 12% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.00893 |  | green | 2026-10-21 to 2026-11-17; 6 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 4.41 | [-14.5, 3.36] | amber | 2026-10-21 to 2026-11-17; percentile 99% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-21 to 2026-11-17; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -1.99 | [-4.77, 6.14] | green | 2026-10-21 to 2026-11-17; percentile 28% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-21 to 2026-11-17; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 681.9 | [-550.0, 694.1] | green | 2026-09-28 to 2026-10-25; percentile 93% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.0967 |  | amber | 2026-09-28 to 2026-10-25; 65 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0683 | [0.0283, 0.125] | green | nominal 10%; 25 origins |
| tail_rate | above p90 | 0.133 | [0.0767, 0.195] | green | nominal 10%; 25 origins |
| median_error | raw p50 | 185.5 | [128.8, 242.9] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 10.0 | [27.0, 28.0] | info | 10 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +13.2% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.06 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.847 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

