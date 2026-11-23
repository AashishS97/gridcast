# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-23 15:58 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.96 | [-2.29, 1.79] | green | 2026-10-20 to 2026-11-16; percentile 10% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.0104 |  | green | 2026-10-20 to 2026-11-16; 7 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 6.61 | [-14.5, 3.37] | amber_unprecedented | 2026-10-20 to 2026-11-16; percentile 100% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-20 to 2026-11-16; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -2.16 | [-4.77, 6.11] | green | 2026-10-20 to 2026-11-16; percentile 25% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-20 to 2026-11-16; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 648.8 | [-550.0, 694.1] | green | 2026-09-27 to 2026-10-24; percentile 92% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.0952 |  | amber | 2026-09-27 to 2026-10-24; 64 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0694 | [0.0278, 0.127] | green | nominal 10%; 24 origins |
| tail_rate | above p90 | 0.127 | [0.0694, 0.191] | green | nominal 10%; 24 origins |
| median_error | raw p50 | 180.2 | [124.9, 233.9] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 7 | [27.0, 28.0] | info | 7 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +13.8% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.05 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.84 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

