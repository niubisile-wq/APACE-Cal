"""E4 protocol-only and curve-only acquisition ablations.

The v2 gates and predictor/rho branch remain unchanged; only the distance used
by facility selection is replaced by protocol-only or curve-only. Results are
development ablations and never overwrite frozen JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_asymmetric_cohort_router as v1


HERE = Path(__file__).parent


def make_router(mode):
    weight = 0.0 if mode == "protocol_only" else float("inf")

    def router(protocol_spread, budget, random_support, acquisition, clients, distances, tie_rank):
        if protocol_spread <= 1e-12:
            return random_support, "fallback_zero_protocol_dispersion"
        if v1.LOW_PROTOCOL_THRESHOLD <= protocol_spread < 0.60:
            return random_support, "fallback_medium_protocol_dispersion"
        if budget == 1:
            return random_support, "fallback_one_label_unidentifiable"
        if budget >= 5 and protocol_spread >= 0.60:
            return random_support, "fallback_large_budget_high_protocol_dispersion"
        support = v1.select_facilities(distances[weight], acquisition, clients, budget, tie_rank)
        return support, f"active_{mode}"

    return router


def run_mode(mode):
    output = HERE / f"batterylife_apace_v2_{mode}_ablation.json"
    original = v2.routed_support
    v2.routed_support = make_router(mode)
    try:
        v2.run((10, 20, 50), (1, 3, 5, 10), 100, output)
    finally:
        v2.routed_support = original
    data = json.loads(output.read_text())
    data["status"] = f"DEVELOPMENT MODALITY ABLATION; {mode}; not frozen"
    data["ablation"] = {
        "changed": mode,
        "unchanged": "v2 gates, rho predictor branch, outer LODO, domains, H/K, 70/30, 100 seeds",
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    for mode in ("protocol_only", "curve_only"):
        data = run_mode(mode)
        print(mode)
        for horizon in (10, 20, 50):
            row = [r for r in data["results"] if r["horizon"] == horizon and r["label_budget_k"] == 3]
            print(horizon,
                  sum(r["baseline"]["mape"] for r in row) / len(row),
                  sum(r["method"]["mape"] for r in row) / len(row))


if __name__ == "__main__":
    main()

