"""Audit whether a common alternate EOL threshold can be reconstructed safely."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import BASE, DATASETS

HERE = Path(__file__).parent
OUT = HERE / "batterylife_eol_definition_audit.json"


def main():
    rows = []
    for dataset in DATASETS:
        labels = json.loads((BASE / "Life labels" / f"{dataset}_labels.json").read_text())
        lengths, differences, missing_capacity = [], [], 0
        for name, label in labels.items():
            path = BASE / dataset / name
            if not path.is_file():
                continue
            data = pickle.load(path.open("rb"))
            cycles = data.get("cycle_data", [])
            lengths.append(len(cycles))
            differences.append(float(label) - len(cycles))
            for c in cycles:
                values = np.asarray(c.get("discharge_capacity_in_Ah", []), dtype=float)
                if values.size == 0 or not np.isfinite(values).any():
                    missing_capacity += 1
        rows.append({"dataset": dataset, "n_labels": len(labels), "n_cycle_files": len(lengths),
                     "cycle_length_min": min(lengths) if lengths else None,
                     "cycle_length_max": max(lengths) if lengths else None,
                     "label_minus_file_length_min": min(differences) if differences else None,
                     "label_minus_file_length_max": max(differences) if differences else None,
                     "cycles_without_finite_discharge_capacity": missing_capacity})
    out = {"conclusion": "No common alternate EOL threshold is reconstructed: the released labels are opaque domain-specific life annotations, and a threshold would require an independently frozen capacity/EOL rule per dataset.", "rows": rows}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
