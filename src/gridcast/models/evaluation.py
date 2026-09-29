"""Pinned evaluation windows: the contract every phase is scored on.

Why pinned: Phase 2 computed its cutoff as "end of data minus 56 days", so
every refetch silently moved the dev/holdout boundary (the dev folds and
the holdout ended up computed from different builds, leaving an 8-day
unscored gap -- harmless, but only by luck). Phase 3 must score on exactly
Phase 2's 140 folds or comparisons to its numbers are void.

Folds are built from the pinned ORIGINS, not from a cutoff: the fold set
is the contract, so that is what gets recorded and checked.

Holdouts carry a status. "sealed" and "spent" holdouts only release their
origins with one_shot=True -- a speed bump, not security, but it makes
scoring a holdout a deliberate, greppable act.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gridcast.models.splits import Fold

HOUR = pd.Timedelta("1h")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "configs" / "evaluation.toml"


def _utc(value: str, field: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        raise ValueError(f"{field}={value!r} has no timezone; write UTC with a trailing Z")
    return ts.tz_convert("UTC")


def load_config(path: str | Path = DEFAULT_PATH) -> dict:
    # utf-8-sig: Windows PowerShell may write a BOM, which tomllib rejects.
    return tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))


def backtest_folds(path: str | Path = DEFAULT_PATH) -> list[Fold]:
    """The pinned rolling-origin folds. Fails loudly if the config is
    internally inconsistent (unreachable last origin, wrong fold count)."""
    cfg = load_config(path)["backtest"]
    first = _utc(cfg["first_origin"], "first_origin")
    last = _utc(cfg["last_origin"], "last_origin")
    train_start = _utc(cfg["train_start"], "train_start")
    step = pd.Timedelta(cfg["step"])
    horizon = int(cfg["horizon_hours"]) * HOUR

    origins = pd.date_range(first, last, freq=step)
    if len(origins) == 0 or origins[-1] != last:
        raise ValueError(f"last_origin {last} is not reachable from {first} in steps of {step}")
    if len(origins) != cfg["expected_folds"]:
        raise ValueError(f"config yields {len(origins)} folds, expected {cfg['expected_folds']}")
    return [Fold(i, train_start, o, o + horizon) for i, o in enumerate(origins)]


@dataclass(frozen=True)
class Holdout:
    name: str
    first_origin: pd.Timestamp
    last_origin: pd.Timestamp
    status: str
    note: str


def holdout_info(name: str, path: str | Path = DEFAULT_PATH) -> Holdout:
    """Metadata only -- safe to read any time."""
    cfg = load_config(path)["holdout"][name]
    return Holdout(
        name=name,
        first_origin=_utc(cfg["first_origin"], "first_origin"),
        last_origin=_utc(cfg["last_origin"], "last_origin"),
        status=cfg["status"],
        note=cfg["note"],
    )


def holdout_origins(
    name: str, *, one_shot: bool = False, path: str | Path = DEFAULT_PATH
) -> pd.DatetimeIndex:
    """Daily origins of a holdout, for SCORING. Requires one_shot=True."""
    info = holdout_info(name, path)
    if info.status in ("sealed", "spent") and not one_shot:
        raise PermissionError(
            f"holdout {name!r} is {info.status}. Pass one_shot=True only for the "
            f"single final evaluation. Note: {info.note}"
        )
    return pd.date_range(info.first_origin, info.last_origin, freq="1D")


def _demo() -> None:
    from gridcast.models.exclusions import contaminated_origins, load_exclusions

    folds = backtest_folds()
    origins = pd.DatetimeIndex([f.origin for f in folds])
    print(f"backtest folds     : {len(folds)}")
    print(f"origins            : {origins[0]} -> {origins[-1]}")
    print(f"span               : {(origins[-1] - origins[0]).days} days")
    dows = pd.Series(origins.day_name()).value_counts()
    print(f"origins per weekday: {dict(dows)}")

    ex = load_exclusions()
    print(f"contaminated folds : {int(contaminated_origins(origins, ex).sum())}")

    for name in ("phase2", "phase3"):
        h = holdout_info(name)
        n = len(pd.date_range(h.first_origin, h.last_origin, freq="1D"))
        print(
            f"\nholdout {name} [{h.status}]: {h.first_origin.date()} -> "
            f"{h.last_origin.date()} ({n} origins)"
        )
        print(f"  {h.note}")


if __name__ == "__main__":
    _demo()
