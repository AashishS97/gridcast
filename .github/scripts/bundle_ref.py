"""Which serving bundle is the champion: read (or set) train_cutoff in
configs/serving.toml. Standard library only, so workflow jobs that never
install the project (the CI image job) can run it with the runner's python3.

configs/serving.toml is the single source of truth: CI builds the API image
from the bundle it names and the daily job forecasts with it. Promoting a
challenger is one changed line here, reviewed as a pull request; rolling back
is reverting that commit (every bundle stays an immutable release).

  python3 .github/scripts/bundle_ref.py
      prints tag=, file=, artifact=, cutoff= (append to $GITHUB_OUTPUT)
  python3 .github/scripts/bundle_ref.py --set-cutoff 2026-10-27 --note "..."
      rewrites the train_cutoff line, keeping everything else byte for byte
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

SERVING = Path("configs/serving.toml")
LINE = re.compile(r'^train_cutoff = "[^"]*"[^\n]*$', re.MULTILINE)


def read_cutoff(path: Path = SERVING) -> str:
    raw = path.read_bytes().decode("utf-8-sig")  # tolerate a Windows BOM
    return str(tomllib.loads(raw)["artifact"]["train_cutoff"])[:10]


def refs(cutoff: str) -> dict[str, str]:
    return {
        "tag": f"serving-{cutoff}",
        "file": f"serving_bundle_{cutoff}.tar.gz",
        "artifact": f"lgbm_quantile_{cutoff}",
        "cutoff": cutoff,
    }


def set_cutoff(cutoff: str, note: str, path: Path = SERVING) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        raise SystemExit(f"cutoff must be YYYY-MM-DD, got {cutoff!r}")
    data = path.read_bytes()
    bom = data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    new, n = LINE.subn(f'train_cutoff = "{cutoff}T00:00:00Z"  # {note}', text)
    if n != 1:
        raise SystemExit(f"expected exactly one train_cutoff line in {path}, found {n}")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8"))
    assert read_cutoff(path) == cutoff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-cutoff")
    ap.add_argument("--note", default="")
    ap.add_argument("--path", type=Path, default=SERVING)
    args = ap.parse_args()
    if args.set_cutoff:
        set_cutoff(args.set_cutoff, args.note, args.path)
    for key, value in refs(read_cutoff(args.path)).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
