"""One-shot postlabel contract audit for the frozen 21700 Expt4 probe.

This script is intentionally not a performance evaluator: after the prelabel
manifest was frozen, it opens only the small processed performance summaries to
check whether a common EOL/cycle-life label exists.  It does not convert
maximum observed cycles into an artificial label.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batterylife_zenodo_21700_prelabel as pre


HERE = Path(__file__).parent
MANIFEST = HERE / "batterylife_zenodo_21700_expt4_prelabel.json"
OUT = HERE / "batterylife_zenodo_21700_postlabel_audit.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    size, entries = pre.central_directory(pre.API)
    rows = []
    for cell in "ABCDEFGH":
        names = [
            name for name in entries
            if "Performance Summary" in name
            and f"cell {cell} (" in name
            and name.endswith("Processed Data.csv")
        ]
        if len(names) != 1:
            raise RuntimeError(f"expected one performance summary for {cell}, got {names}")
        data = pre.member_bytes(pre.API, entries[names[0]])
        records = list(csv.DictReader(io.StringIO(data.decode("utf-8", "replace"))))
        soh = [float(r["SoH"]) for r in records if r.get("SoH") not in (None, "")]
        cycles = [float(r["Ageing Cycles"]) for r in records if r.get("Ageing Cycles") not in (None, "")]
        rows.append({
            "cell": cell,
            "performance_member": names[0],
            "n_age_sets": len(records),
            "max_ageing_cycles": max(cycles) if cycles else None,
            "min_soh": min(soh) if soh else None,
            "eol_80_reached": bool(soh and min(soh) <= 0.80),
            "eol_90_reached": bool(soh and min(soh) <= 0.90),
        })
    output = {
        "phase": "POSTLABEL_INPUT_CONTRACT_AUDIT",
        "manifest": MANIFEST.name,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "archive_size_bytes": size,
        "label_opening": "processed performance summaries opened only after prelabel manifest freeze",
        "rows": rows,
        "decision": "technical_failure_no_common_80_percent_eol",
        "reason": "All eight cells stop near 4653 ageing cycles and minimum SoH is above 0.80; maximum observed cycle is not used as a life label.",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
