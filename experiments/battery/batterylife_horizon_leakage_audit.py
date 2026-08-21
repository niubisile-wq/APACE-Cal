"""Counterfactual off-by-one audit for fixed early-cycle horizons."""
import json
import pickle
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import BASE, curve_features


OUTPUT = Path(__file__).with_name("batterylife_horizon_leakage_audit.json")
DATASETS = ("CALB", "HNEI", "MICH_EXP", "CALCE", "MICH", "SNL", "NA-ion")


def poison_cycle(cycle):
    changed = dict(cycle)
    for key in (
        "current_in_A",
        "voltage_in_V",
        "charge_capacity_in_Ah",
        "discharge_capacity_in_Ah",
        "time_in_s",
        "temperature_in_C",
    ):
        changed[key] = np.asarray([1e12, -1e12, 5e11], dtype=float)
    changed["internal_resistance_in_ohm"] = np.asarray([1e12], dtype=float)
    return changed


def main():
    rows = []
    for dataset in DATASETS:
        paths = sorted((BASE / dataset).glob("*.pkl"))
        for horizon in (10, 20, 50):
            selected = None
            for path in paths:
                data = pickle.load(open(path, "rb"))
                if len(data.get("cycle_data", [])) >= horizon + 1:
                    selected = (path, data)
                    break
            if selected is None:
                continue
            path, data = selected
            original = curve_features(data, horizon)

            tail_changed = dict(data)
            tail_changed["cycle_data"] = list(data["cycle_data"][:horizon]) + [
                poison_cycle(cycle) for cycle in data["cycle_data"][horizon:]
            ]
            after_tail_poison = curve_features(tail_changed, horizon)

            boundary_changed = dict(data)
            boundary_changed["cycle_data"] = list(data["cycle_data"])
            boundary_changed["cycle_data"][horizon - 1] = poison_cycle(data["cycle_data"][horizon - 1])
            after_boundary_poison = curve_features(boundary_changed, horizon)

            tail_invariant = bool(np.array_equal(original, after_tail_poison, equal_nan=True))
            boundary_sensitive = bool(not np.array_equal(original, after_boundary_poison, equal_nan=True))
            rows.append(
                {
                    "dataset": dataset,
                    "file": path.name,
                    "horizon": horizon,
                    "tail_from_cycle": horizon + 1,
                    "tail_invariant": tail_invariant,
                    "last_visible_cycle_sensitive": boundary_sensitive,
                    "passed": tail_invariant and boundary_sensitive,
                }
            )
    output = {
        "protocol": "poison every cycle after H and require identical features; poison cycle H and require changed features",
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "n_checks": len(rows),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit("Horizon leakage audit failed")


if __name__ == "__main__":
    main()
