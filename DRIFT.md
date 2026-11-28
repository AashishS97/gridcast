# GridCast drift and data checks

Written by `gridcast.monitoring.drift` at 2026-11-28 16:17 UTC. Thresholds:
`configs/monitoring.toml`. Feature and target drift compare the recent window with
the same season in the training years (before 2026-10-01), as
anomalies from the seasonal average; they are amber at most, because unusual
inputs are a reason to look, not evidence of worse forecasts.

## Feature drift (weather inputs)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | temperature_2m | -1.67 | [-2.25, 1.79] | green | 2026-10-25 to 2026-11-21; percentile 13% of 273 same-season reference windows (~10 independent) |
| support | temperature_2m | 0.00893 |  | green | 2026-10-25 to 2026-11-21; 6 of 672 hours outside the seasonal training range |
| shift | shortwave_radiation | 4.09 | [-14.5, 3.57] | amber | 2026-10-25 to 2026-11-21; percentile 97% of 273 same-season reference windows (~10 independent) |
| support | shortwave_radiation | 0 |  | green | 2026-10-25 to 2026-11-21; 0 of 672 hours outside the seasonal training range |
| shift | wind_speed_10m | -3.25 | [-4.77, 6.31] | green | 2026-10-25 to 2026-11-21; percentile 12% of 273 same-season reference windows (~10 independent) |
| support | wind_speed_10m | 0.067 |  | amber | 2026-10-25 to 2026-11-21; 45 of 672 hours outside the seasonal training range |

## Target drift (load)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| shift | load_mw | 691.3 | [-550.0, 710.3] | green | 2026-10-02 to 2026-10-29; percentile 94% of 289 same-season reference windows (~10 independent) |
| support | load_mw | 0.113 |  | amber | 2026-10-02 to 2026-10-29; 76 of 672 hours outside the seasonal training range |

## Concept drift (residuals of the served model)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| tail_rate | below p10 | 0.0759 | [0.0387, 0.125] | green | nominal 10%; 28 origins |
| tail_rate | above p90 | 0.124 | [0.0729, 0.182] | green | nominal 10%; 28 origins |
| median_error | raw p50 | 176.2 | [125.8, 227.6] | amber | y - raw p50; the calibration pool's median (the p50 shift) is +109.9 MW |

## Data checks (vintage ledger)

| check | subject | value | band | zone | note |
|---|---|---|---|---|---|
| revisions_seen | load | 16.0 | [27.0, 28.0] | info | 16 day(s) revised by > 5% in the ledger; ages at revision 27-28 days, median change +14.0% |
| late_revision | load | 0 |  | green | no day older than 29 days changed in the latest run |
| final_level | load | 1.05 |  | green | mean of final days aged 30-43 / aged > 43; red below 0.85 |
| provisional | load | 0.846 |  | info | days aged 1-29 / final days aged 30-43: well below 1 means under-publication is active |
| final_gaps | load | 0 |  | green | every final day has 24 hours |
| weather_lag | temperature | 6 |  | green | newest day with temperature is 6 days old |

