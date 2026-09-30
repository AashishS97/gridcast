# Phase 3 — Probabilistic Forecasting & Backtesting: Results

All numbers below come from runs in this phase. Backtest = the 140 pinned
Phase 2 folds (2024-06-29 -> 2026-05-25, 5-day step, 24 h horizon, expanding
window); calibrated comparisons use the 134 folds with calibration history.
Holdout = 15 sealed August 2026 origins, scored exactly once.

## 1. Data state (step 0)

- **Two corrupted months of ENTSO-E NL actual load.** Compared with the same ISO
  week and weekday in 2023-25: July 2026 ran at 0.65x normal (monthly mean
  8,015 MW vs 11,920-12,375 MW), September 2026 at 0.73x (9,400 MW vs
  12,561-12,857 MW). August 2026 was clean, sitting between the two bad months.
  The corruption aligns with calendar months, which matches the monthly
  fetch granularity. July is traced to upstream (Phase 2); **September's raw XML
  has not been traced** yet.
- **Handling:** `configs/data_exclusions.toml` masks the target to NaN
  (1,415 newly masked hours) and never drops rows, so the hourly grid stays
  intact for the row-based rolling windows. Every one of the 83 missing load
  hours in the dataset lies inside an exclusion. Any origin whose read window
  (336 h back, 24 h forward) overlaps an exclusion is not scored: 77 origins.
- **Correction to Phase 2:** the TSO day-ahead forecast is not only
  amplitude-damped but also **biased low**. The monthly median actual/TSO ratio
  is 1.06-1.36, seasonal, and the May-June offset grows year over year. Phase 2
  numbers are unaffected (the bias-corrected baseline absorbs the level; the
  production model excludes `da_forecast_mw`).
- The weather cache never refreshed on its own (a Phase 1 flaw). Refetched,
  coverage went from 93.8% to 99.5%. A proper staleness check is Phase 5 work.

## 2. Pinned evaluation windows

Phase 2 derived its cutoff from "end of data", so it moved on every refetch.
The Phase 2 dev folds and holdout had been computed from different data builds,
leaving an 8-day unscored gap: harmless, but only by luck.
`configs/evaluation.toml` now pins the fold set itself (140 folds, 20 per
weekday, verified identical to Phase 2's scored folds 0-139) and gives
holdouts a status (sealed/spent). The splitter's default step moved from 7D
(which aliases onto one weekday) to 5D.

## 3. Quantile model (p10/p50/p90)

One LightGBM model per quantile (`objective="quantile"`), Phase 2's fixed
hyperparameters, early stopping on each quantile's own pinball loss,
`da_forecast_mw` dropped.

- **The median reproduces Phase 2 exactly:** raw q50 MAE 269.9 MW, equal to
  Phase 2 `lightgbm_no_da` (269.9) over 3,360 hours. At tau = 0.5 the quantile
  objective is L1, so this validates the whole Phase 3 pipeline.
- **The raw intervals are badly overconfident** (140 folds):

| | nominal | actual |
|---|---|---|
| below p10 | 0.10 | 0.151 |
| below p50 | 0.50 | 0.375 |
| below p90 | 0.90 | 0.781 |
| inside [p10, p90] | 0.80 | **0.630** |

- The raw crossing rate was 13.6% (the p90 model stops at a median of 152 trees,
  against 1,362 for p50). Rearrangement (sorting each row) never increases the
  summed pinball loss (a pointwise rearrangement-inequality argument; tested) and
  lowered the q50 MAE to 268.3.

## 4. Why rolling-origin backtesting and not one holdout

1. **A single window only contains its own conditions.** The August holdout had
   no cold, no holidays and no winter evening peaks; scored on August alone,
   the model looks fine. The backtest spans two winters, and that is where the
   cold-tail bias surfaced.
2. **Short windows are noisy.** Uncalibrated monthly interval coverage ranged
   0.46-0.76 across the backtest; within the 15 holdout days, one week covered
   0.67 and the next 0.80. One window is one draw from that spread.
3. **It evaluates the process.** Each fold retrains only on its own past,
   exactly as production would.
4. **Calibration needs it.** Conformal calibration consumes a sequence of
   out-of-sample residuals, which a single split cannot provide.
5. **It still needs a sealed holdout.** The backtest informed decisions
   (calibration window, flag thresholds), so it is optimistic by selection. The
   holdout influenced no decision.

Costs: 420 model fits (~14 min). Folds are **not independent**: neighbouring
folds share almost all their training data, so the effective sample size is
well below 140. Step choice matters: a 7-day step aliases every fold onto one
weekday (Phase 2 bug; a 5-day step is coprime with 7).

## 5. Failure analysis (uncalibrated model)

- **The bias is episodic, not a growing trend.** The load-growth hypothesis
  (bias rising over time) was falsified. There is a persistent +0.5-1% bias in
  most months, plus three episodes: Nov 2024 (+2.2%), Jan 2026 (+2.7%),
  Mar 2026 (+2.6%), in which actuals fell below p50 only 11-14% of the time.
- **Afternoons and evenings are worst:** interval coverage was 0.45-0.57 over
  15:00-18:00 local, and at 17:00 only 58% of actuals fell below p90.
- **The cold tail is thin data, not extrapolation.** No test hour fell below its
  fold's training minimum, yet the bias grows steadily with coldness
  (bottom 2%: +383 MW; 2-10%: +164 MW; middle: +92 MW; 90-98%: +17 MW).
  **Being rare inside the training range is enough to fail.** Interval coverage
  was low in every band (0.61-0.66), so the width problem is global.
- **Holidays:** the model widens its intervals (1,299 vs 648 MW) and p90
  coverage is 0.89. It expresses uncertainty when given an explicit signal.
  Bridge days are missing from the calendar features (2025-05-30 was
  over-forecast by 416 MW).
- **The TSO as an independent witness:** the worst days were mostly missed by
  the TSO too, often by more (2026-03-11: model +6.1%, TSO +11.3%). The Spearman
  correlation between model misses and TSO surprise is 0.34, so some failures are
  shared and some are not. A trust flag cannot anticipate shocks nobody saw.
- **Design limit:** origins are midnight UTC, so horizon and hour of day are
  perfectly confounded.

## 6. Conformal calibration (asymmetric CQR-style)

Each quantile is shifted by the tau-quantile of its recent out-of-sample
residuals, using the trailing 12 folds (~60 days). W=12 was declared in advance;
W=24 is a sensitivity check only. There is no formal guarantee, because time
series are not exchangeable. Same 134 folds:

| | uncalibrated | conformal W=12 | W=24 (sensitivity) |
|---|---|---|---|
| interval coverage (0.80) | 0.630 | **0.791** | 0.786 |
| below p10/p50/p90 | 0.15/0.37/0.78 | 0.10/0.49/0.89 | 0.10/0.48/0.88 |
| mean pinball | 96.3 | **91.4** (-5.1%) | 91.7 |
| interval score | 1,541 | **1,448** (-6.1%) | 1,442 |
| q50 MAE (MW) | 269.5 | **259.0** (-3.9%) | 261.9 |
| mean width (MW) | 680 | 985 (+45%) | 962 |

Limits: calibration holds on average, not conditionally. Nights are now
over-covered (0.84-0.90) and afternoons under-covered (0.66-0.76). After an
episode the trailing window over-corrects (the month after each bad month shows
a negative bias: Dec 2024 -80, Feb 2026 -117, Apr 2026 -125 MW).

## 7. Trust flag

Signals known at forecast time: `cold` (target-hour temperature in the bottom
10% of the training history), `calendar` (holiday, bridge day, Dec 24/31),
`instability` (|conformal median shift| > 2% of the forecast), and
`integrity` (a missing or excluded load hour in the 336 h lookback). The
thresholds were declared before evaluation but motivated by the same backtest,
so these results are **descriptive**.

- Flags 21.5% of hours (cold 14.9%, calendar 7.9%, instability 0.4%,
  integrity 0%).
- Flagged hours versus unflagged: q50 MAE 338 vs 237 MW, interval score
  1,900 vs 1,323 (1.44x); flagged hours capture 39.4% of the worst 10% of errors
  (lift 1.83). All three pre-declared criteria pass.
- The flag predicts **lower precision, not interval failure**: only 24% of
  out-of-interval hours are flagged (lift 1.13).
- `calendar` is the strongest single signal (coverage 0.70, bias -243 MW).
  Calibration pushes forecasts up, which is the wrong direction on holidays.
- `instability` was inactive in the backtest; its per-hour threshold should be
  per-origin (Phase 4 fix). `integrity` is covered by unit tests only.

## 8. Sealed holdout (Aug 16-30 2026, 15 origins, scored once)

| | backtest calibrated | holdout uncalibrated | holdout calibrated |
|---|---|---|---|
| q50 MAE (MW) | 259.0 | 242.3 | **243.2** |
| interval coverage (0.80) | 0.791 | 0.703 | **0.736** |
| below p10/p50/p90 | 0.10/0.49/0.89 | 0.08/0.33/0.78 | 0.15/0.57/0.88 |
| mean pinball | 91.4 | 79.3 | **77.2** |
| interval score | 1,448 | 1,166 | **1,101** |

For scale, the weekly seasonal naive scores 559 MW MAE. All three declared
confirmation criteria pass. Calibration fixed the upper tail but shifted the
distribution up too far (below p50 went from 0.33 to 0.57).

**After the outage:** Aug 16-22 was calibrated only on June residuals (54 days
old; June ran 7-11% above prior years): MAE 281 MW, coverage 0.67. Aug 23-30,
with mostly fresh August residuals (7 days old): MAE 210 MW, coverage 0.80.
Early August also had degraded training (NaN lags, a mostly masked
early-stopping window), so the two causes cannot be separated. The
`instability` signal, inactive in the backtest, fired on 13.9% of holdout hours,
all in the stale week, including all of Aug 16, the worst-calibrated day. That
is one day of evidence, not validation.

## 9. Known limitations

- Weather features are observed weather (perfect prognosis); production
  forecasts will be worse.
- Calibration is marginal only: hour-of-day and cold-tail conditional
  calibration are not solved.
- The trust-flag design is in-sample; its only out-of-sample test (August)
  contained no cold or calendar cases.
- September 2026's upstream corruption is not yet traced to raw XML.
- Horizon and hour of day are confounded (midnight-UTC origins).
