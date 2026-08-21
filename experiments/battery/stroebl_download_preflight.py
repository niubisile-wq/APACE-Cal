"""Safe preflight for the large Stroebl external archive.

It never downloads or opens labels. It only checks destination capacity and records
the official Figshare endpoint so a later run cannot silently write to the root disk.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

DEST = Path("/autodl-fs/data/battery_external_stroebl")
URL = "https://figshare.com/ndownloader/articles/25975315/versions/1"
EXPECTED_COMPRESSED_BYTES = 9_590_000_000


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DEST).free
    report = {
        "destination": str(DEST),
        "free_bytes": free,
        "required_compressed_bytes": EXPECTED_COMPRESSED_BYTES,
        "safety_margin_bytes": 2_000_000_000,
        "official_endpoint": URL,
        "capacity_ok": free >= EXPECTED_COMPRESSED_BYTES + 2_000_000_000,
        "labels_opened": False,
    }
    print(json.dumps(report, indent=2))
    if not report["capacity_ok"]:
        raise SystemExit("Insufficient destination capacity; no download attempted")


if __name__ == "__main__":
    main()
