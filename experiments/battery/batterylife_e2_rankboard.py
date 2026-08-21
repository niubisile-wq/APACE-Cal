"""Build the reproducible E2 traditional/active rank board.

This script only reads already completed, fixed-pool JSON artifacts and never
changes the frozen APACE-Cal method or any external confirmation artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
FILES = {
    "nested_calibrator": HERE / "batterylife_fixed_pool_nested.json",
    "active_selector": HERE / "batterylife_fixed_pool_acquisition.json",
    "apace_v2": HERE / "batterylife_asymmetric_cohort_router_v2.json",
}
OUT = HERE / "batterylife_e2_rankboard.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    loaded = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    rows = []
    for horizon in (10, 20, 50):
        for budget in (1, 3, 5, 10):
            for name, data in loaded.items():
                key = f"macro_h{horizon}_k{budget}"
                metric = data[key]
                rows.append(
                    {
                        "horizon": horizon,
                        "label_budget_k": budget,
                        "method_family": name,
                        "baseline_mape": metric["baseline_mape"],
                        "method_mape": metric["method_mape"],
                        "relative_reduction_percent": 100.0 * (
                            metric["baseline_mape"] - metric["method_mape"]
                        ) / max(metric["baseline_mape"], 1e-12),
                        "baseline_mae": metric.get("baseline_mae"),
                        "method_mae": metric.get("method_mae"),
                        "curve_selected_domains": metric.get("curve_selected_domains"),
                    }
                )
    output = {
        "protocol": (
            "fixed total-K protocol; six opened development domains; 100 episodes; "
            "cell-level aggregation; source artifacts are not overwritten"
        ),
        "source_hashes": {str(path.name): digest(path) for path in FILES.values()},
        "rows": rows,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for h, k in ((10, 3), (20, 3), (50, 3), (50, 5), (50, 10)):
        subset = [r for r in rows if r["horizon"] == h and r["label_budget_k"] == k]
        print(h, k, json.dumps(subset, sort_keys=True))


if __name__ == "__main__":
    main()
