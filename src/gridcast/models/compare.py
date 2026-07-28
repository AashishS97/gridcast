"""Phase 2 comparison report: where does each model fail?

Reads the merged backtest results and produces per-horizon, per-local-hour,
and per-fold breakdowns, plus a worst-days table for the top models.
Averages hide failures; distributions and worst cases reveal them.

Run:  uv run python -m gridcast.models.compare
Writes markdown + csv to reports/phase2/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridcast.models.metrics import summarize

RESULTS_PATH = Path("data/backtests/baselines.parquet")
REPORT_DIR = Path("reports/phase2")

MAIN_MODELS = ["lightgbm", "seasonal_naive_168", "sarimax_fourier"]


def fmt(df: pd.DataFrame, floatfmt: str = "{:,.1f}") -> str:
    return df.to_string(float_format=lambda v: floatfmt.format(v))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_parquet(RESULTS_PATH)
    sections: list[str] = []

    def emit(title: str, body: str) -> None:
        print(f"\n=== {title} ===\n{body}")
        sections.append(f"## {title}\n\n```\n{body}\n```\n")

    # 1. Overall, with MAE/RMSE ratio (rare-but-severe failure indicator)
    overall = summarize(results).sort_values("mae")
    overall["rmse/mae"] = overall["rmse"] / overall["mae"]
    emit("Overall (all folds, all horizons)", fmt(overall.set_index("model")))

    # 2. MAE per horizon, every horizon, main models
    by_h = summarize(results[results["model"].isin(MAIN_MODELS)], by=["model", "horizon"])
    emit(
        "MAE by horizon (1-24)",
        fmt(by_h.pivot(index="horizon", columns="model", values="mae"), "{:,.0f}"),
    )

    # 3. MAE per local hour of day (Europe/Amsterdam)
    by_hr = summarize(results[results["model"].isin(MAIN_MODELS)], by=["model", "hour_local"])
    emit(
        "MAE by local hour (Europe/Amsterdam)",
        fmt(by_hr.pivot(index="hour_local", columns="model", values="mae"), "{:,.0f}"),
    )

    # 4. Per-fold MAE distribution: does the mean hide catastrophes?
    per_fold = (
        results[results["model"].isin(MAIN_MODELS)]
        .groupby(["model", "fold"])
        .apply(lambda g: (g["y_true"] - g["y_pred"]).abs().mean(), include_groups=False)
        .rename("fold_mae")
        .reset_index()
    )
    dist = per_fold.groupby("model")["fold_mae"].describe(percentiles=[0.5, 0.9, 0.99])[
        ["mean", "50%", "90%", "99%", "max"]
    ]
    emit("Per-fold MAE distribution", fmt(dist))

    # 5. Worst 10 folds for the top model, with dates and the naive's score
    #    on the same folds — failures clustering on holidays/DST/weather?
    top = per_fold[per_fold["model"] == "lightgbm"].nlargest(10, "fold_mae")
    fold_dates = (
        results[results["model"] == "lightgbm"]
        .groupby("fold")["timestamp"]
        .min()
        .rename("forecast_day")
    )
    naive = (
        per_fold[per_fold["model"] == "seasonal_naive_168"]
        .set_index("fold")["fold_mae"]
        .rename("naive168_mae")
    )
    worst = top.set_index("fold").join([fold_dates, naive])
    worst["forecast_day"] = worst["forecast_day"].dt.tz_convert("Europe/Amsterdam").dt.date
    worst["dow"] = pd.to_datetime(worst["forecast_day"]).dt.day_name()
    emit("Worst 10 LightGBM folds", fmt(worst[["forecast_day", "dow", "fold_mae", "naive168_mae"]]))

    # 6. Win rate: on what fraction of folds does lightgbm beat the naive?
    wide = per_fold.pivot(index="fold", columns="model", values="fold_mae")
    wins = (wide["lightgbm"] < wide["seasonal_naive_168"]).mean()
    emit(
        "Head-to-head", f"lightgbm beats seasonal_naive_168 on " f"{wins:.1%} of {len(wide)} folds"
    )

    (REPORT_DIR / "comparison.md").write_text(
        "# Phase 2 model comparison\n\n" + "\n".join(sections), encoding="utf-8"
    )
    overall.to_csv(REPORT_DIR / "overall.csv", index=False)
    per_fold.to_csv(REPORT_DIR / "per_fold_mae.csv", index=False)
    print(f"\nWritten to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
