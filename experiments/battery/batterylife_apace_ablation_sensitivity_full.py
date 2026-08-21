"""Complete preregistered E4/E5 development-domain ablations.

This file deliberately runs variants outside the frozen v2 source.  It keeps
the same cells, episode seeds, outer LODO predictor selection, and metrics, and
records the exact variant definition so none of these exploratory results can
silently modify APACE-Cal v2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells
from batterylife_transductive_pool_acquisition import WEIGHTS, key_number

HERE = Path(__file__).parent
OUTPUT = HERE / "batterylife_apace_ablation_sensitivity_full.json"
HORIZONS = (10, 20, 50)
BUDGETS = (1, 3, 5, 10)
BASE_PREDICTORS = v1.predictor_names()


def standard_scale(values):
    values = np.asarray(values, dtype=float)
    center = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return scale


def variant_specs():
    specs = [
        {"name": "A3_fixed_concat", "family": "E4", "fixed_weight": 1.0, "disable_fallback": True},
        {"name": "A7_all_median", "family": "E4", "all_median": True},
        {"name": "A8_mismatched_evidence", "family": "E4", "acquisition_weight": 0.0, "predictor_weight": 2.0},
        {"name": "A9_fixed_predictor", "family": "E4", "fixed_predictor": "logmean"},
        {"name": "A10_no_robust_scaling", "family": "E4", "no_robust_scaling": True},
    ]
    for value in (0.20, 0.25, 0.30, 0.35, 0.40):
        specs.append({"name": f"E5_low_{value:g}", "family": "E5", "low": value})
    for value in (0.50, 0.55, 0.60, 0.65, 0.70):
        specs.append({"name": f"E5_high_{value:g}", "family": "E5", "high": value})
    for value in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        specs.append({"name": f"E5_rho_{value:g}", "family": "E5", "rho": value})
    for value in (0.25, 0.5, 1.0, 2.0):
        specs.append({"name": f"E5_bw_{value:g}", "family": "E5", "bandwidth": value})
    for value in (0.0, 0.125, 0.5, 1.0, 2.0, math.inf):
        label = "inf" if math.isinf(value) else f"{value:g}"
        specs.append({"name": f"E5_weight_{label}", "family": "E5", "predictor_weight": value})
    return specs


def baseline_choice(evaluations, target, horizon, budget, fixed_predictor=None):
    if fixed_predictor is not None:
        return fixed_predictor
    candidates = BASE_PREDICTORS + ("support_median",)
    domains = [d for d in DATASETS if d != target]
    ranked = []
    for predictor in candidates:
        values = [evaluations[(horizon, d, budget)]["summaries"]["baseline"][predictor] for d in domains]
        ranked.append((float(np.mean([x["mape"] for x in values])), float(np.mean([x["mae"] for x in values])), predictor))
    return min(ranked)[-1]


def run_variant(spec, loaded, seeds):
    low = float(spec.get("low", 0.30))
    high = float(spec.get("high", 0.60))
    rho_threshold = float(spec.get("rho", 0.35))
    acquisition_weight = spec.get("acquisition_weight")
    fixed_weight = spec.get("fixed_weight")
    predictor_weight = spec.get("predictor_weight", None)
    bandwidth = float(spec.get("bandwidth", 0.5))
    old = (v1.LOW_PROTOCOL_THRESHOLD, v1.routed_support, v1.predictor_names,
           v1.predict, v1.robust_scale, v2.CONCORDANCE_THRESHOLD, v2.robust_scale)
    v1.LOW_PROTOCOL_THRESHOLD = low
    v1.predictor_names = v2.predictor_names
    v1.predict = v2.predict
    if spec.get("no_robust_scaling"):
        v1.robust_scale = standard_scale
        v2.robust_scale = standard_scale

    def route(spread, budget, random_support, acquisition, clients, distances, tie_rank):
        if spec.get("disable_fallback"):
            weight = fixed_weight
            support = v1.select_facilities(distances[weight], acquisition, clients, budget, tie_rank)
            return support, f"active_w{key_number(weight)}"
        if spread <= 1e-12:
            return random_support, "fallback_zero_protocol_dispersion"
        if low <= spread < high:
            return random_support, "fallback_medium_protocol_dispersion"
        if budget == 1:
            return random_support, "fallback_one_label_unidentifiable"
        if budget >= 5 and spread >= high:
            return random_support, "fallback_large_budget_high_protocol_dispersion"
        weight = acquisition_weight if acquisition_weight is not None else (2.0 if spread < low else 0.5)
        support = v1.select_facilities(distances[weight], acquisition, clients, budget, tie_rank)
        return support, f"active_w{key_number(weight)}"

    v1.routed_support = route
    v2.CONCORDANCE_THRESHOLD = rho_threshold
    evaluations = {
        (h, d, k): v1.evaluate(loaded[(h, d)], h, k, seeds, 1e9, False)
        for h in HORIZONS for d in DATASETS for k in BUDGETS
    }
    rows = []
    for h in HORIZONS:
        for target in DATASETS:
            for k in BUDGETS:
                evaluation = evaluations[(h, target, k)]
                baseline_predictor = baseline_choice(evaluations, target, h, k, spec.get("fixed_predictor"))
                active = any(x.startswith("active_") for x in evaluation["route_counts"])
                spread = evaluation["protocol_dispersion"]
                rho = v2.distance_concordance(loaded[(h, target)])
                if not active:
                    method_predictor = baseline_predictor
                elif spec.get("all_median"):
                    method_predictor = "support_median"
                elif predictor_weight is not None:
                    if math.isinf(predictor_weight):
                        method_predictor = f"winf_bw{key_number(bandwidth)}"
                    else:
                        method_predictor = f"w{key_number(predictor_weight)}_bw{key_number(bandwidth)}"
                elif spread >= high and rho < rho_threshold:
                    method_predictor = "support_median"
                elif spread >= high:
                    method_predictor = f"w0.5_bw{key_number(bandwidth)}"
                else:
                    method_predictor = f"w2_bw{key_number(bandwidth)}"
                b = evaluation["summaries"]["baseline"][baseline_predictor]
                m = evaluation["summaries"]["router"][method_predictor]
                rel = 100.0 * (b["mape"] - m["mape"]) / max(b["mape"], 1e-12)
                rows.append({"horizon": h, "target": target, "K": k,
                             "baseline_predictor": baseline_predictor,
                             "method_predictor": method_predictor,
                             "route_counts": evaluation["route_counts"],
                             "baseline_mape": b["mape"], "method_mape": m["mape"],
                             "relative_reduction_percent": rel})
    # Restore imported modules before the next variant.
    (v1.LOW_PROTOCOL_THRESHOLD, v1.routed_support, v1.predictor_names,
     v1.predict, v1.robust_scale, v2.CONCORDANCE_THRESHOLD, v2.robust_scale) = old
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    loaded = {(h, d): load_cells(d, h) for h in HORIZONS for d in DATASETS}
    output = {"protocol": "same six domains, H/K grid, common deterministic seeds; development-only variants", "variants": {}}
    for index, spec in enumerate(variant_specs(), 1):
        print(f"[{index}/{len(variant_specs())}] {spec['name']}", flush=True)
        rows = run_variant(spec, loaded, args.seeds)
        macro = {}
        for h in HORIZONS:
            for k in BUDGETS:
                subset = [r for r in rows if r["horizon"] == h and r["K"] == k]
                rel = [r["relative_reduction_percent"] for r in subset]
                macro[f"h{h}_k{k}"] = {
                    "baseline_mape": float(np.mean([r["baseline_mape"] for r in subset])),
                    "method_mape": float(np.mean([r["method_mape"] for r in subset])),
                    "improved_same_worse_domains": [int(sum(x > 1e-12 for x in rel)), int(sum(abs(x) <= 1e-12 for x in rel)), int(sum(x < -1e-12 for x in rel))],
                    "worst_domain_relative_change_percent": float(-min(rel)),
                }
        output["variants"][spec["name"]] = {"spec": spec, "macro": macro, "rows": rows}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
