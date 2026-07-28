"""Inspect raw ENTSO-E day-ahead files: what domain did we actually fetch?

Each ENTSO-E XML response embeds the area code (outBiddingZone_Domain /
inBiddingZone_Domain) that was requested. If the day-ahead fetch used a
different code than the actuals fetch, the two series describe different
bidding zones and no amount of alignment will fix it.
"""

import re
from pathlib import Path

RAW_DIR = Path("data/raw/entsoe")

files = sorted(RAW_DIR.rglob("*"))
files = [p for p in files if p.is_file()]
print(f"{len(files)} files under {RAW_DIR}:")
for p in files[:40]:
    print(f"  {p.relative_to(RAW_DIR)}  ({p.stat().st_size:,} bytes)")
if len(files) > 40:
    print(f"  ... and {len(files) - 40} more")

# Patterns that identify the requested area and document type in ENTSO-E XML
patterns = {
    "domain_codes": re.compile(r"10Y[A-Z0-9\-]{13}"),
    "document_type": re.compile(r"<type>(A\d+)</type>"),
    "process_type": re.compile(r"<process\.processType>(A\d+)</process\.processType>"),
    "business_type": re.compile(r"<businessType>(A\d+)</businessType>"),
}

for p in files:
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"\n{p.name}: could not read as text ({e})")
        continue
    if "<" not in text[:1000]:
        continue  # not XML
    found = {name: sorted(set(rx.findall(text))) for name, rx in patterns.items()}
    print(f"\n{p.relative_to(RAW_DIR)}:")
    for name, values in found.items():
        print(f"  {name:>14}: {values}")
