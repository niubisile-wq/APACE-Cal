"""E0 development-domain data card and leakage/duplicate audit.

This audit is deliberately limited to the six already-opened development
domains. It never reads any external confirmation-domain labels.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import BASE, DATASETS, curve_features, load_cells


HERE = Path(__file__).parent
OUT = HERE / "batterylife_e0_data_audit.json"
HORIZONS = (10, 20, 50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = {
        "scope": "six opened development domains only; no external confirmation labels",
        "datasets": {},
        "checks": {},
    }
    duplicate_names = []
    duplicate_file_hashes = []
    all_feature_checks = []

    for dataset in DATASETS:
        label_path = BASE / "Life labels" / f"{dataset}_labels.json"
        labels = json.loads(label_path.read_text())
        names = sorted(labels)
        file_records = []
        for name in names:
            path = BASE / dataset / name
            if not path.exists():
                continue
            file_records.append(
                {
                    "name": name,
                    "file_sha256": sha256_file(path),
                    "label": float(labels[name]),
                }
            )
        counts = {}
        finite_checks = {}
        for horizon in HORIZONS:
            cells = load_cells(dataset, horizon)
            counts[str(horizon)] = len(cells)
            finite_checks[str(horizon)] = all(
                np.isfinite(np.asarray(cell["protocol"], dtype=float)).any()
                and np.isfinite(np.asarray(cell["curve"], dtype=float)).any()
                for cell in cells
            )
            all_feature_checks.append(finite_checks[str(horizon)])
        labels_by_name = Counter(record["name"] for record in file_records)
        duplicate_names.extend(
            {"dataset": dataset, "name": name, "count": count}
            for name, count in labels_by_name.items()
            if count > 1
        )
        hashes = Counter(record["file_sha256"] for record in file_records)
        duplicate_file_hashes.extend(
            {"dataset": dataset, "sha256": digest, "count": count}
            for digest, count in hashes.items()
            if count > 1
        )
        report["datasets"][dataset] = {
            "label_file_sha256": sha256_file(label_path),
            "label_key_count": len(labels),
            "file_count_with_labels": len(file_records),
            "horizon_cell_counts": counts,
            "all_feature_arrays_have_finite_value": finite_checks,
            "life_min": min((record["label"] for record in file_records), default=None),
            "life_max": max((record["label"] for record in file_records), default=None),
            "file_records": file_records,
        }

    report["checks"] = {
        "duplicate_label_keys": duplicate_names,
        "duplicate_file_hashes_within_domain": duplicate_file_hashes,
        "all_development_domains_present": all(
            report["datasets"][dataset]["file_count_with_labels"] > 0
            for dataset in DATASETS
        ),
        "all_feature_arrays_have_finite_value": all(all_feature_checks),
        "horizon_audit_json_passed": json.loads(
            (HERE / "batterylife_horizon_leakage_audit.json").read_text()
        )["passed"],
    }
    report["passed"] = bool(
        report["checks"]["all_development_domains_present"]
        and report["checks"]["all_feature_arrays_have_finite_value"]
        and not duplicate_names
        and not duplicate_file_hashes
        and report["checks"]["horizon_audit_json_passed"]
    )
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "horizon_counts": {d: report["datasets"][d]["horizon_cell_counts"]
                                         for d in DATASETS}}, indent=2))
    if not report["passed"]:
        raise SystemExit("E0 data audit failed")


if __name__ == "__main__":
    main()

