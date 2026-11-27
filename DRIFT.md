# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-27 17:32 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.66 | [-2.26, 1.79] | green | 2026-10-24 to 2026-11-20; percentile 15% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.00893 |  | green | 2026-10-24 to 2026-11-20; 6 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 3.95 | [-14.5, 3.47] | amber | 2026-10-24 to 2026-11-20; percentile 97% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-24 to 2026-11-20; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -2.61 | [-4.77, 6.27] | green | 2026-10-24 to 2026-11-20; percentile 18% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-24 to 2026-11-20; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 704.2 | [-550.0, 698.4] | amber | 2026-10-01 to 2026-10-28; percentile 95% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.116 |  | amber | 2026-10-01 to 2026-10-28; 78 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0685 | [0.0327, 0.118] | green | nominal 10%; 28 origins |
| tail_rate | above p90 | 0.129 | [0.0774, 0.187] | green | nominal 10%; 28 origins |
| median_error | raw p50 | 175.8 | [124.9, 228.2] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 14.0 | [27.0, 28.0] | info | 14 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +15.2% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.05 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.845 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

