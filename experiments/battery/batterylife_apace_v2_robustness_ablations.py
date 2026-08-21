"""E6 fixed-input robustness stress tests for APACE-Cal v2.

Only unlabeled protocol/early-curve features are perturbed. Life labels,
episode splits, outer selection and frozen v2 rules are unchanged. Each
condition is an independent development stress test, not a new method.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2


HERE = Path(__file__).parent
ORIGINAL_LOAD = v2.load_cells


def perturb_cells(dataset, horizon, mode):
    cells = ORIGINAL_LOAD(dataset, horizon)
    output = []
    for cell in cells:
        item = dict(cell)
        token = f"{dataset}|{horizon}|{cell['name']}|{mode}".encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)
        protocol = np.asarray(cell["protocol"], dtype=float).copy()
        curve = np.asarray(cell["curve"], dtype=float).copy()
        if mode.startswith("curve_missing"):
            fraction = float(mode.replace("curve_missing", "")) / 100.0
            mask = rng.random(curve.size) < fraction
            curve[mask] = np.nan
        elif mode.startswith("protocol_missing"):
            fraction = float(mode.replace("protocol_missing", "")) / 100.0
            mask = rng.random(protocol.size) < fraction
            protocol[mask] = np.nan
        elif mode.startswith("curve_noise"):
            fraction = float(mode.replace("curve_noise", "").replace("pct", "")) / 100.0
            finite = np.isfinite(curve)
            curve[finite] *= 1.0 + rng.normal(0.0, fraction, size=int(finite.sum()))
        else:
            raise KeyError(mode)
        item["protocol"] = protocol
        item["curve"] = curve
        output.append(item)
    return output


def run_mode(mode):
    v2.load_cells = lambda dataset, horizon: perturb_cells(dataset, horizon, mode)
    output = HERE / f"batterylife_apace_v2_{mode}.json"
    try:
        v2.run((10, 20, 50), (1, 3, 5, 10), 100, output)
    finally:
        v2.load_cells = ORIGINAL_LOAD
    data = json.loads(output.read_text())
    data["status"] = f"DEVELOPMENT ROBUSTNESS STRESS; {mode}; not frozen"
    data["perturbation"] = {
        "mode": mode,
        "changed": "unlabeled protocol/early-curve features only",
        "unchanged": "life labels, splits, H/K, v2 gates, predictor, outer selection, 100 seeds",
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    for mode in (
        "curve_missing10", "curve_missing20", "curve_missing30",
        "curve_noise0.5pct", "curve_noise1pct", "curve_noise2pct",
        "protocol_missing10", "protocol_missing25", "protocol_missing50",
    ):
        data = run_mode(mode)
        print(mode)
        for h in (10, 20, 50):
            rows = [r for r in data["results"] if r["horizon"] == h and r["label_budget_k"] == 3]
            print(h, sum(r["baseline"]["mape"] for r in rows) / 6,
                  sum(r["method"]["mape"] for r in rows) / 6,
                  [r["method"]["mape"] > r["baseline"]["mape"] + 1e-12 for r in rows].count(True))


if __name__ == "__main__":
    main()
