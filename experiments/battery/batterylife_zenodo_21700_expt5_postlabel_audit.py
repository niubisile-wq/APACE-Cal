"""Postlabel EOL-contract audit for the frozen Expt5 manifest."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batterylife_zenodo_21700_prelabel as io217

HERE = Path(__file__).parent
MANIFEST = HERE / "batterylife_zenodo_21700_expt5_prelabel.json"
OUT = HERE / "batterylife_zenodo_21700_expt5_postlabel_audit.json"
API = "https://zenodo.org/api/records/10637534/files/Expt%205%20-%20Standard%20Cycle%20Aging%20(Control).zip/content"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    size, entries = io217.central_directory(API)
    rows = []
    for cell in "ABCDEFGH":
        names = [n for n in entries if "Performance Summary" in n and f"cell {cell} (" in n and n.endswith("Processed Data.csv")]
        if len(names) != 1:
            raise RuntimeError(f"performance summary mismatch for {cell}: {names}")
        records = list(csv.DictReader(io.StringIO(io217.member_bytes(API, entries[names[0]]).decode("utf-8", "replace"))))
        soh = [float(r["SoH"]) for r in records if r.get("SoH") not in (None, "")]
        cycles = [float(r["Ageing Cycles"]) for r in records if r.get("Ageing Cycles") not in (None, "")]
        rows.append({"cell": cell, "performance_member": names[0], "n_age_sets": len(records),
                     "max_ageing_cycles": max(cycles) if cycles else None,
                     "min_soh": min(soh) if soh else None,
                     "eol_80_reached": bool(soh and min(soh) <= .80),
                     "eol_90_reached": bool(soh and min(soh) <= .90)})
    output = {"phase":"POSTLABEL_INPUT_CONTRACT_AUDIT", "manifest":MANIFEST.name,
              "manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "archive_size_bytes":size,
              "label_opening":"performance summaries requested only after frozen episode manifest",
              "rows":rows,
              "common_80_percent_eol":all(r["eol_80_reached"] for r in rows),
              "common_90_percent_eol":all(r["eol_90_reached"] for r in rows),
              "decision":"eligible_for_80_percent_eol_evaluation" if all(r["eol_80_reached"] for r in rows) else "technical_failure_no_common_80_percent_eol"}
    OUT.write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps(output,indent=2))


if __name__ == "__main__": main()
