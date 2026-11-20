"""Static checks on an exported challenger, then its serving bundle.

A challenger is the same recipe at a newer cutoff (export_artifact
--cutoff). Before it starts its paired trial it must be a sound artifact:

  integrity     every file matches its manifest sha256; feature order matches
  data          the feature table being bundled is the one it was exported
                from (sha256 in the manifest)
  parity        the serving code (Forecaster) reproduces the export's own
                reference forecast for the cutoff origin, to 1e-6 MW
  objective     every model's recorded params say objective="quantile" with
                its own alpha (the champion's manifest said "l1")
  shifts        conformal shifts strictly increase p10 < p50 < p90: the
                monitoring recovers raw quantiles as served minus shift,
                which is exact only then
  pool          calibration-pool raw coverage and q50 MAE inside sanity
                bounds (configs/monitoring.toml [candidate])

These catch a broken export or broken data, not a worse model: quality is
the paired trial's job. Only if every check passes is the bundle written:
models/<name>/ and data/processed/features.parquet in one tar.gz, the same
layout as the champion's release, so CI and the daily job unpack it the
same way.

Run:  uv run python -m gridcast.models.candidate --cutoff 2026-10-22 --out dist
      prints the release tag and bundle path; exit 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.models.backtest import FEATURES_PATH
from gridcast.models.exclusions import load_exclusions
from gridcast.models.export_artifact import PROJECT_ROOT, load_config, parse_cutoff
from gridcast.models.metrics import DEFAULT_QUANTILES
from gridcast.monitoring.accuracy import CONFIG_PATH, GATE_BASELINE_PATH
from gridcast.serving.forecaster import (
    ArtifactError,
    Forecaster,
    file_sha256,
    load_artifact,
    load_provisional_days,
)

PARITY_ATOL_MW = 1e-6


@dataclass(frozen=True)
class CandidateConfig:
    pool_coverage_min: float
    pool_coverage_max: float
    pool_mae_max_ratio: float


def load_candidate_config(path: str | Path = CONFIG_PATH) -> CandidateConfig:
    raw = tomllib.loads(Path(path).read_bytes().decode("utf-8-sig"))["candidate"]
    return CandidateConfig(
        pool_coverage_min=float(raw["pool_coverage_min"]),
        pool_coverage_max=float(raw["pool_coverage_max"]),
        pool_mae_max_ratio=float(raw["pool_mae_max_ratio"]),
    )


def release_tag(cutoff: pd.Timestamp) -> str:
    return f"serving-{cutoff:%Y-%m-%d}"


def bundle_name(cutoff: pd.Timestamp) -> str:
    return f"serving_bundle_{cutoff:%Y-%m-%d}.tar.gz"


def validate(
    artifact_dir: Path,
    features_path: Path,
    cfg: CandidateConfig,
    gate_baseline: dict,
    provisional_days: int,
) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for every check; later checks are skipped
    when the artifact cannot even be loaded."""
    out: list[tuple[str, bool, str]] = []
    try:
        artifact = load_artifact(artifact_dir)
    except ArtifactError as e:
        return [("integrity", False, str(e))]
    m = artifact.manifest
    out.append(("integrity", True, f"{len(m['files'])} files match their sha256"))

    sha = file_sha256(features_path)
    expected = m.get("inputs", {}).get("features_sha256")
    out.append(
        ("data", sha == expected, f"features sha256 {sha[:12]} vs manifest {str(expected)[:12]}")
    )

    if sha == expected:
        fc = Forecaster(
            artifact=artifact,
            features=pd.read_parquet(features_path),
            exclusions=load_exclusions(),
            provisional_days=provisional_days,
            features_sha256=sha,
        )
        ref = pd.read_parquet(artifact_dir / "reference_forecast.parquet")
        got = fc.forecast(artifact.train_cutoff)
        cols = [f"{n}_cal" for n in artifact.names]
        diff = float(np.max(np.abs(got[cols].to_numpy() - ref[cols].to_numpy())))
        out.append(("parity", diff <= PARITY_ATOL_MW, f"max |serving - export| {diff:.2e} MW"))
    else:
        out.append(("parity", False, "skipped: not the export's data"))

    params = m.get("training", {}).get("params", {})
    bad = [
        n
        for n in artifact.names
        if not isinstance(params.get(n), dict)
        or params[n].get("objective") != "quantile"
        or params[n].get("alpha") != DEFAULT_QUANTILES[n]
    ]
    out.append(
        (
            "objective",
            not bad,
            "quantile objective with own alpha"
            if not bad
            else f"wrong or missing params for {bad}",
        )
    )

    shifts = [artifact.shifts[n] for n in artifact.names]
    increasing = all(a < b for a, b in zip(shifts, shifts[1:], strict=False))
    out.append(("shifts", increasing, " < ".join(f"{s:+.1f}" for s in shifts)))

    cal = m.get("calibration", {})
    cov, mae = cal.get("pool_raw_interval_coverage"), cal.get("pool_q50_mae_mw")
    ref_mae = gate_baseline["metrics"]["mae_q50"]
    ok_cov = cov is not None and cfg.pool_coverage_min <= cov <= cfg.pool_coverage_max
    ok_mae = mae is not None and mae <= cfg.pool_mae_max_ratio * ref_mae
    out.append(
        (
            "pool",
            ok_cov and ok_mae,
            f"raw coverage {cov:.1%} in [{cfg.pool_coverage_min:.0%}, "
            f"{cfg.pool_coverage_max:.0%}]; q50 MAE {mae:.1f} <= "
            f"{cfg.pool_mae_max_ratio} x {ref_mae:.1f}",
        )
    )
    return out


def build_bundle(artifact_dir: Path, features_path: Path, out_path: Path) -> Path:
    """tar.gz with models/<name>/... and data/processed/features.parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(artifact_dir, arcname=f"models/{artifact_dir.name}")
        tar.add(features_path, arcname="data/processed/features.parquet")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="validate a challenger and bundle it")
    ap.add_argument("--cutoff", required=True, help="the challenger's train cutoff (YYYY-MM-DD)")
    ap.add_argument("--out", type=Path, required=True, help="directory for the bundle")
    args = ap.parse_args(argv)
    cutoff = parse_cutoff(args.cutoff)
    serving = load_config()
    artifact_dir = serving.output_root / f"lgbm_quantile_{cutoff:%Y-%m-%d}"
    features = Path(FEATURES_PATH)
    if not features.is_absolute():
        features = PROJECT_ROOT / features
    gate = json.loads(GATE_BASELINE_PATH.read_text(encoding="utf-8"))

    results = validate(
        artifact_dir, features, load_candidate_config(), gate, load_provisional_days()
    )
    for check, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {check:10s} {detail}")
    if not all(ok for _, ok, _ in results):
        print("candidate rejected: no bundle written", file=sys.stderr)
        return 1
    path = build_bundle(artifact_dir, features, args.out / bundle_name(cutoff))
    print(f"tag={release_tag(cutoff)}")
    print(f"bundle={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
