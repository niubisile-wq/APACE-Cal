"""Cross-predictor audit for the frozen APACE-Cal v2 routing.

Evaluates the same random/router supports with every predictor exposed by the
frozen local calibration interface on H50/K3.  It is an audit artifact only;
the frozen method and blind manifests are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells


HERE = Path(__file__).parent
OUT = HERE / "batterylife_cross_predictor_audit.json"


def main() -> None:
    # Apply the exact v2 routing/predictor definitions in this process only.
    v1.predictor_names = v2.predictor_names
    v1.predict = v2.predict
    v1.routed_support = v2.routed_support
    horizon, budget, seeds = 50, 3, 100
    loaded = {d: load_cells(d, horizon) for d in DATASETS}
    evaluations = {d: v1.evaluate(loaded[d], horizon, budget, seeds, 1e9, False) for d in DATASETS}
    records = []
    for target in DATASETS:
        target_eval = evaluations[target]
        dev = [evaluations[d] for d in DATASETS if d != target]
        baseline_choice, _ = v2.choose_baseline(dev)
        baseline_predictor = baseline_choice["predictor"]
        active = any(route.startswith("active_") for route in target_eval["route_counts"])
        spread = target_eval["protocol_dispersion"]
        rho = v2.distance_concordance(loaded[target])
        if not active:
            method_predictor = baseline_predictor
        elif spread >= 0.60:
            method_predictor = "support_median" if rho < v2.CONCORDANCE_THRESHOLD else "w0.5_bw0.5"
        else:
            method_predictor = "w2_bw0.5"
        all_rows = []
        for predictor in v2.predictor_names():
            b = target_eval["summaries"]["baseline"][predictor]
            m = target_eval["summaries"]["router"][predictor]
            all_rows.append({
                "predictor": predictor,
                "baseline_mape": b["mape"],
                "router_mape": m["mape"],
                "relative_reduction_percent": 100.0 * (b["mape"] - m["mape"]) / max(b["mape"], 1e-12),
            })
        records.append({
            "target": target,
            "horizon": horizon,
            "label_budget_k": budget,
            "protocol_dispersion": spread,
            "rho": rho,
            "active": active,
            "selected_baseline_predictor": baseline_predictor,
            "selected_method_predictor": method_predictor,
            "all_predictor_rows": all_rows,
        })
    predictors = v2.predictor_names()
    macro = {}
    for predictor in predictors:
        rows = [next(x for x in r["all_predictor_rows"] if x["predictor"] == predictor) for r in records]
        macro[predictor] = {
            "baseline_mape": float(np.mean([r["baseline_mape"] for r in rows])),
            "router_mape": float(np.mean([r["router_mape"] for r in rows])),
            "relative_reduction_percent": float(np.mean([r["relative_reduction_percent"] for r in rows])),
        }
    output = {
        "protocol": "H50/K3; six development domains; 100 fixed episodes; identical random/router supports per episode; all frozen predictor interfaces",
        "records": records,
        "macro_by_predictor": macro,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
