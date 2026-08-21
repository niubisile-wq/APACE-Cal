"""E4 safety ablation: remove every APACE-v2 fallback.

Only the router changes. The v2 predictor, outer baseline selection, domains,
H/K grid, 70/30 split and 100 seeds remain identical. This is expected to
expose the negative-transfer risk that motivated v2.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_asymmetric_cohort_router as v1


HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_v2_always_active_ablation.json"


def always_active(protocol_spread, budget, random_support, acquisition, clients, distances, tie_rank):
    if protocol_spread <= 1e-12:
        return random_support, "fallback_zero_protocol_dispersion"
    weight = 2.0 if protocol_spread < v1.LOW_PROTOCOL_THRESHOLD else 0.5
    support = v1.select_facilities(distances[weight], acquisition, clients, budget, tie_rank)
    return support, f"active_no_fallback_w{v1.key_number(weight)}"


def main():
    original = v2.routed_support
    v2.routed_support = always_active
    try:
        v2.run((10, 20, 50), (1, 3, 5, 10), 100, OUT)
    finally:
        v2.routed_support = original
    data = json.loads(OUT.read_text())
    data["status"] = "DEVELOPMENT SAFETY ABLATION; all fallback gates removed; not frozen"
    data["ablation"] = {
        "changed": "remove K=1, medium-dispersion, high-dispersion-large-K fallbacks",
        "unchanged": "same v2 predictor, outer LODO baseline, domains, H/K, 70/30, 100 seeds",
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    for h in (10, 20, 50):
        row = [r for r in data["results"] if r["horizon"] == h and r["label_budget_k"] == 3]
        b = sum(x["baseline"]["mape"] for x in row) / len(row)
        m = sum(x["method"]["mape"] for x in row) / len(row)
        print(h, "K3", b, m)


if __name__ == "__main__":
    main()

