"""E7 route phase diagram and small-input route-flip audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix

HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_route_stability_full.json"


def dispersion(cells):
    p = np.asarray([c["protocol"] for c in cells], dtype=float)
    dp = distance_matrix(p, robust_scale(p), 1e9)
    upper = dp[np.triu_indices(len(cells), 1)]
    return float(np.median(upper)) if len(upper) else 0.0


def route(spread, k):
    if spread <= 1e-12:
        return "fallback_zero_protocol_dispersion"
    if 0.30 <= spread < 0.60:
        return "fallback_medium_protocol_dispersion"
    if k == 1:
        return "fallback_one_label_unidentifiable"
    if k >= 5 and spread >= 0.60:
        return "fallback_large_budget_high_protocol_dispersion"
    return "active_low_or_high_dispersion"


def noisy(cells, token, level=0.01):
    out = []
    for c in cells:
        h = hashlib.sha256(f"{token}|{c['name']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        x = dict(c)
        p = np.asarray(c["protocol"], dtype=float).copy()
        q = np.asarray(c["curve"], dtype=float).copy()
        pf = np.isfinite(p); qf = np.isfinite(q)
        p[pf] += rng.normal(0, level, int(pf.sum()))
        q[qf] *= 1 + rng.normal(0, level, int(qf.sum()))
        x["protocol"], x["curve"] = p, q
        out.append(x)
    return out


def main():
    output = {"protocol": "unlabeled phase/flip audit; 100 deterministic 1% perturbations", "rows": []}
    for h in (10, 20, 50):
        for d in DATASETS:
            cells = load_cells(d, h)
            spread = dispersion(cells)
            rho = v2.distance_concordance(cells)
            base_routes = {str(k): route(spread, k) for k in (1, 3, 5, 10)}
            flips = {str(k): 0 for k in (1, 3, 5, 10)}
            for seed in range(100):
                perturbed = noisy(cells, f"{d}|{h}|{seed}")
                s = dispersion(perturbed)
                for k in flips:
                    flips[k] += int(route(s, int(k)) != base_routes[k])
            output["rows"].append({"dataset": d, "horizon": h,
                "protocol_dispersion": spread, "rho": rho,
                "base_routes": base_routes,
                "route_flip_rate_percent": {k: 100*v/100 for k,v in flips.items()}})
    # Explicitly retain the largest and smallest per-cell H50/K3 effects.
    main = json.loads((HERE / "batterylife_asymmetric_cohort_router_v2.json").read_text())
    candidates = []
    for r in main["results"]:
        if r["horizon"] == 50 and r["label_budget_k"] == 3:
            for c in r["per_cell"]:
                candidates.append({"domain": r["target"], "cell": c["held_out"],
                    "absolute_mape_change": c["method_ape"] - c["baseline_ape"]})
    output["h50_k3_extremes"] = {"best_5": sorted(candidates, key=lambda x:x["absolute_mape_change"])[:5],
        "worst_5": sorted(candidates, key=lambda x:x["absolute_mape_change"], reverse=True)[:5]}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"rows": len(output["rows"]), "max_flip": max(max(x["route_flip_rate_percent"].values()) for x in output["rows"])}, indent=2))


if __name__ == "__main__":
    main()
