# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-25 18:02 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.73 | [-2.29, 1.79] | green | 2026-10-22 to 2026-11-18; percentile 12% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.00893 |  | green | 2026-10-22 to 2026-11-18; 6 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 4.19 | [-14.5, 3.41] | amber | 2026-10-22 to 2026-11-18; percentile 98% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-22 to 2026-11-18; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -1.94 | [-4.77, 6.14] | green | 2026-10-22 to 2026-11-18; percentile 29% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-22 to 2026-11-18; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 701.7 | [-550.0, 694.1] | amber | 2026-09-29 to 2026-10-26; percentile 96% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.106 |  | amber | 2026-09-29 to 2026-10-26; 71 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0689 | [0.0288, 0.122] | green | nominal 10%; 26 origins |
| tail_rate | above p90 | 0.135 | [0.0785, 0.194] | green | nominal 10%; 26 origins |
| median_error | raw p50 | 188.5 | [131.3, 244.7] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 11.0 | [27.0, 28.0] | info | 11 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +13.8% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.06 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.845 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

