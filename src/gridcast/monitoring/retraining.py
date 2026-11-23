"""Retraining policy: when to build a challenger, and whether it may replace
the champion. Pure decisions over the monitoring logs; the workflows act on
them.

WHEN TO BUILD (trigger):
  schedule      the champion has not seen max_unseen_days of final data.
  performance   a red accuracy zone (coverage, skill) or a red concept-drift
  / concept     check (tail rates, raw median error), or an amber one that
                has held for persistent_amber_days distinct days.
  not drift     feature and target drift never trigger on their own: if the
                inputs are unusual, a model refreshed on data that does not
                contain them cannot know them either. They are a reason to
                watch the residuals, which do trigger.
  blocked       a red data check (late or missing revision): training on data
                that may not be final would bake the defect into the model;
                fewer than min_new_days of new final data: nothing to learn.

WHETHER TO PROMOTE (champion / challenger):
  The challenger's cutoff is today's provisional line, so at build time none
  of its out-of-sample days is final -- the hard problem. It is not solved by
  evaluating on older days: every day before the cutoff is in its training
  or calibration data (leakage), and the only older out-of-sample days left
  are the spent holdouts, which are never re-scored. Instead both models
  forecast every new origin from the challenger's cutoff on, and each pair is
  scored once its target day is final -- about one new paired day per day.
  The comparison is PAIRED (same origins, same actuals), so day-to-day
  difficulty cancels and the band on the difference is far narrower than
  either model's own band.

  wait      fewer than min_paired paired origins.
  reject    the challenger is worse beyond doubt (the band on the relative
            pinball difference lies above 0), or its coverage zone is red
            where the champion's is not (a fault it would introduce).
  promote   non-inferior: the band's upper end is below +margin.
  at max_paired, neither: promote if the challenger's mean pinball is not
            above the champion's, else reject.

Non-inferiority, not superiority: a refresh of the same recipe on more data
is expected to be at least as good (the backtest retrained daily); the
burden of proof is on keeping a stale model, not on refreshing it. A recipe
change is a different case and goes through the backtest gate first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.export_artifact import load_config as load_serving_config
from gridcast.models.metrics import DEFAULT_QUANTILES
from gridcast.monitoring.accuracy import (
    CONFIG_PATH,
    MonitoringConfig,
    band,
    coverage_zone,
    resample_index,
)
from gridcast.serving.daily_log import VINTAGE_COLS, read_log
from gridcast.serving.forecaster import load_provisional_days

TAUS = {
    "p10": DEFAULT_QUANTILES["q10"],
    "p50": DEFAULT_QUANTILES["q50"],
    "p90": DEFAULT_QUANTILES["q90"],
}
CONCEPT_CHECKS = {"tail_rate", "median_error"}


@dataclass(frozen=True)
class RetrainingConfig:
    max_unseen_days: int
    min_new_days: int
    persistent_amber_days: int
    min_paired: int
    max_paired: int
    margin: float


def load_config(path: str | Path = CONFIG_PATH) -> RetrainingConfig:
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))["retraining"]
    cfg = RetrainingConfig(
        max_unseen_days=int(raw["max_unseen_days"]),
        min_new_days=int(raw["min_new_days"]),
        persistent_amber_days=int(raw["persistent_amber_days"]),
        min_paired=int(raw["min_paired"]),
        max_paired=int(raw["max_paired"]),
        margin=float(raw["margin"]),
    )
    if not 1 <= cfg.min_paired <= cfg.max_paired:
        raise ValueError("need 1 <= min_paired <= max_paired")
    if not 0 <= cfg.min_new_days <= cfg.max_unseen_days:
        raise ValueError("need 0 <= min_new_days <= max_unseen_days")
    return cfg


# --- trigger ------------------------------------------------------------------


@dataclass
class Trigger:
    build: bool
    candidate_cutoff: pd.Timestamp
    unseen_days: int
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "BUILD" if self.build else "no build"
        lines = [
            f"{head}: candidate cutoff {self.candidate_cutoff:%Y-%m-%d}, "
            f"{self.unseen_days} day(s) of final data unseen by the champion"
        ]
        lines += [f"  reason : {r}" for r in self.reasons]
        lines += [f"  blocked: {b}" for b in self.blockers]
        return "\n".join(lines)


def _red_or_persistent(
    rows: pd.DataFrame, label: str, persistent_days: int
) -> tuple[list[str], list[str]]:
    """rows: one zone per (run day, subject). Red on the latest run day
    triggers; amber triggers if it held on each of the last N run days."""
    if rows.empty:
        return [], []
    days = sorted(rows["run_day"].unique())
    latest = rows[rows["run_day"] == days[-1]]
    red = [
        f"{label} {s}: {z}"
        for s, z in latest[["subject", "zone"]].itertuples(index=False)
        if str(z).startswith("red")
    ]
    persistent = []
    recent = days[-persistent_days:]
    if len(recent) == persistent_days:
        window = rows[rows["run_day"].isin(recent)]
        for subject, g in window.groupby("subject"):
            amber_days = g.loc[g["zone"].astype(str).str.startswith("amber"), "run_day"].nunique()
            if amber_days == persistent_days:
                persistent.append(f"{label} {subject}: amber on {persistent_days} consecutive days")
    return red, persistent


def _last_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only each day's last run (manual re-runs must not count twice)."""
    df = df.copy()
    df["run_day"] = pd.to_datetime(df["run_at"], utc=True).dt.floor("D")
    last = df.groupby("run_day")["run_at"].transform("max")
    return df[df["run_at"] == last]


def trigger(
    champion_cutoff: pd.Timestamp,
    provisional_from: pd.Timestamp,
    zones: pd.DataFrame,
    drift: pd.DataFrame,
    cfg: RetrainingConfig,
) -> Trigger:
    """Decide whether to build a challenger now. zones: zones.csv; drift:
    drift.csv (both may be empty)."""
    cutoff = provisional_from.floor("D")
    unseen = int((cutoff - champion_cutoff).days)
    t = Trigger(build=False, candidate_cutoff=cutoff, unseen_days=unseen)

    if unseen >= cfg.max_unseen_days:
        t.reasons.append(f"schedule: {unseen} unseen days >= {cfg.max_unseen_days}")

    if len(zones):
        z = _last_per_day(zones)
        long = pd.concat(
            [
                z[["run_day", "coverage_zone"]]
                .rename(columns={"coverage_zone": "zone"})
                .assign(subject="coverage"),
                z[["run_day", "skill_zone"]]
                .rename(columns={"skill_zone": "zone"})
                .assign(subject="skill"),
            ]
        )
        red, persistent = _red_or_persistent(long, "accuracy", cfg.persistent_amber_days)
        t.reasons += red + persistent

    if len(drift):
        d = _last_per_day(drift)
        concept = d[d["check"].isin(CONCEPT_CHECKS)]
        red, persistent = _red_or_persistent(concept, "concept", cfg.persistent_amber_days)
        t.reasons += red + persistent
        latest = d[d["run_day"] == d["run_day"].max()]
        for check, subject, zone in latest.loc[
            latest["family"] == "data", ["check", "subject", "zone"]
        ].itertuples(index=False):
            if str(zone).startswith("red"):
                t.blockers.append(f"data check {check} ({subject}) is red")

    if unseen < cfg.min_new_days:
        t.blockers.append(f"only {unseen} new final day(s) (< {cfg.min_new_days})")
    t.build = bool(t.reasons) and not t.blockers
    return t


# --- champion / challenger ----------------------------------------------------


def daily_pinball(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per origin: mean pinball over p10/p50/p90 (served), 80% coverage."""
    h = hourly.sort_values(["origin", "horizon"])
    y = h["y"].to_numpy(float)
    loss = np.zeros(len(h))
    for col, tau in TAUS.items():
        d = y - h[col].to_numpy(float)
        loss += np.maximum(tau * d, (tau - 1) * d)
    h = h.assign(pinball=loss / len(TAUS), hit=(h["y"] >= h["p10"]) & (h["y"] <= h["p90"]))
    g = h.groupby("origin")
    return pd.DataFrame(
        {"pinball": g["pinball"].mean(), "coverage": g["hit"].mean(), "n_hours": g.size()}
    )


@dataclass
class Comparison:
    decision: str  # wait | promote | reject
    n_paired: int
    rel_diff: float = np.nan  # challenger / champion - 1, mean pinball
    rel_band: tuple[float, float] = (np.nan, np.nan)
    champion_pinball: float = np.nan
    challenger_pinball: float = np.nan
    challenger_coverage: float = np.nan
    challenger_coverage_zone: str = ""
    champion_coverage_zone: str = ""
    reason: str = ""


def compare(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    challenger_cutoff: pd.Timestamp,
    cfg: RetrainingConfig,
    acc: MonitoringConfig,
) -> Comparison:
    """Paired comparison on origins both have scored, from the challenger's
    cutoff on (earlier origins were inside its training data)."""
    a, b = daily_pinball(champion), daily_pinball(challenger)
    common = a.index.intersection(b.index)
    common = common[common >= challenger_cutoff].sort_values()[: cfg.max_paired]
    common = common[
        (a.loc[common, "n_hours"] == 24).to_numpy() & (b.loc[common, "n_hours"] == 24).to_numpy()
    ]
    n = len(common)
    if n < cfg.min_paired:
        return Comparison("wait", n, reason=f"{n} paired origin(s), need {cfg.min_paired}")

    pa, pb = a.loc[common, "pinball"].to_numpy(), b.loc[common, "pinball"].to_numpy()
    idx = resample_index(n, acc.n_boot, acc.seed)
    rel = pb.sum() / pa.sum() - 1
    rel_band = band(pb[idx].sum(axis=1) / pa[idx].sum(axis=1) - 1, acc.amber_level)

    def zone(cov: np.ndarray) -> str:
        boot = cov[idx].mean(axis=1)
        return coverage_zone(
            cov.mean(),
            band(boot, acc.amber_level),
            band(boot, acc.red_level),
            acc.nominal_coverage,
        )

    cov_b = b.loc[common, "coverage"].to_numpy()
    cz, champ_z = zone(cov_b), zone(a.loc[common, "coverage"].to_numpy())
    out = Comparison(
        "wait",
        n,
        float(rel),
        rel_band,
        float(pa.mean()),
        float(pb.mean()),
        float(cov_b.mean()),
        cz,
        champ_z,
    )

    if rel_band[0] > 0:
        out.decision, out.reason = "reject", "challenger worse: whole band above 0"
    elif cz.startswith("red") and not champ_z.startswith("red"):
        out.decision, out.reason = "reject", f"challenger coverage {cz}, champion {champ_z}"
    elif rel_band[1] < cfg.margin:
        out.decision = "promote"
        out.reason = f"non-inferior: band upper {rel_band[1]:+.1%} < margin {cfg.margin:+.0%}"
    elif n >= cfg.max_paired:
        if rel <= 0:
            out.decision, out.reason = "promote", "trial ended undecided; not worse on average"
        else:
            out.decision, out.reason = "reject", "trial ended undecided; worse on average"
    else:
        out.reason = (
            f"undecided after {n} paired origins (band {rel_band[0]:+.1%} to "
            f"{rel_band[1]:+.1%}); trial runs to {cfg.max_paired}"
        )
    return out


CHALLENGER_FILE = "challenger.json"


def read_challenger(log_dir: Path) -> dict | None:
    path = log_dir / CHALLENGER_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def register_challenger(log_dir: Path, cutoff: pd.Timestamp, tag: str, now: pd.Timestamp) -> dict:
    """Record a new challenger on the log branch: the daily job runs its
    trial while status is "trial"."""
    state = {
        "tag": tag,
        "cutoff": cutoff.isoformat(),
        "artifact": f"lgbm_quantile_{cutoff:%Y-%m-%d}",
        "status": "trial",
        "registered_at": now.isoformat(),
        "code_sha": (os.environ.get("GITHUB_SHA") or "local")[:12],
    }
    (log_dir / CHALLENGER_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


TRIALS_FILE = "trials.csv"
TRIAL_COLS = ["tag", "cutoff", "registered_at", "closed_at", "status", "reason"]


def close_challenger(log_dir: Path, status: str, reason: str, now: pd.Timestamp) -> dict:
    """End the running trial: challenger.json keeps the verdict, and the
    trial is appended to trials.csv (the retraining history)."""
    if status not in {"promoted", "rejected"}:
        raise ValueError("status must be promoted or rejected")
    state = read_challenger(log_dir)
    if not state or state.get("status") != "trial":
        raise SystemExit("no trial in progress")
    state.update({"status": status, "closed_at": now.isoformat(), "reason": reason})
    (log_dir / CHALLENGER_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    path = log_dir / TRIALS_FILE
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=TRIAL_COLS)
    row = pd.DataFrame([{c: state.get(c, "") for c in TRIAL_COLS}], columns=TRIAL_COLS)
    pd.concat([old, row], ignore_index=True).to_csv(path, index=False)
    return state


def decide(log_dir: Path, force: bool, champion_cutoff: pd.Timestamp | None = None) -> Trigger:
    """The trigger on a log directory. A running trial blocks a new build;
    --force supplies a reason (manual request) but never lifts a blocker."""
    v = read_log(log_dir / "vintages.csv", VINTAGE_COLS)
    if v.empty:
        raise SystemExit("no vintages logged yet: cannot place a cutoff")
    provisional_from = v["data_as_of"].max() - pd.Timedelta(days=load_provisional_days())

    def csv(name: str) -> pd.DataFrame:
        path = log_dir / name
        return pd.read_csv(path, parse_dates=["run_at"]) if path.exists() else pd.DataFrame()

    champion = (
        champion_cutoff if champion_cutoff is not None else load_serving_config().train_cutoff
    )
    t = trigger(
        champion,
        provisional_from,
        csv("zones.csv"),
        csv("drift.csv"),
        load_config(),
    )
    if force:
        t.reasons.append("manual: forced from workflow_dispatch")
    state = read_challenger(log_dir)
    if state and state.get("status") == "trial":
        t.blockers.append(f"trial in progress: {state['tag']}")
    t.build = bool(t.reasons) and not t.blockers
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="retraining decisions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="should a challenger be built now?")
    d.add_argument("--log-dir", type=Path, required=True)
    d.add_argument("--force", action="store_true")
    d.add_argument(
        "--github-output",
        action="store_true",
        help="also write build= and cutoff= to $GITHUB_OUTPUT",
    )
    c = sub.add_parser("close", help="end the running trial with a verdict")
    c.add_argument("--log-dir", type=Path, required=True)
    c.add_argument("--status", required=True, choices=["promoted", "rejected"])
    c.add_argument("--reason", required=True)
    r = sub.add_parser("register", help="record a published challenger")
    r.add_argument("--log-dir", type=Path, required=True)
    r.add_argument("--cutoff", required=True)
    r.add_argument("--tag", required=True)
    args = ap.parse_args()

    if args.cmd == "close":
        state = close_challenger(
            args.log_dir, args.status, args.reason, pd.Timestamp.now(tz="UTC").floor("s")
        )
        print(json.dumps(state, indent=2))
        return 0

    if args.cmd == "register":
        state = register_challenger(
            args.log_dir,
            pd.Timestamp(args.cutoff, tz="UTC"),
            args.tag,
            pd.Timestamp.now(tz="UTC").floor("s"),
        )
        print(json.dumps(state, indent=2))
        return 0

    t = decide(args.log_dir, args.force)
    print(t.summary())
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"build={'true' if t.build else 'false'}\n")
            f.write(f"cutoff={t.candidate_cutoff:%Y-%m-%d}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
