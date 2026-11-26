# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-26 06:17 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.62 | [-2.28, 1.79] | green | 2026-10-23 to 2026-11-19; percentile 17% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.00893 |  | green | 2026-10-23 to 2026-11-19; 6 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 3.47 | [-14.5, 3.43] | amber | 2026-10-23 to 2026-11-19; percentile 95% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-23 to 2026-11-19; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -2.13 | [-4.77, 6.19] | green | 2026-10-23 to 2026-11-19; percentile 27% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-23 to 2026-11-19; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 718.6 | [-550.0, 695.4] | amber | 2026-09-30 to 2026-10-27; percentile 96% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.113 |  | amber | 2026-09-30 to 2026-10-27; 76 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0694 | [0.0324, 0.12] | green | nominal 10%; 27 origins |
| tail_rate | above p90 | 0.134 | [0.0802, 0.191] | green | nominal 10%; 27 origins |
| median_error | raw p50 | 178.9 | [125.1, 238.0] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 12.0 | [27.0, 28.0] | info | 12 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +13.2% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.04 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.84 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

