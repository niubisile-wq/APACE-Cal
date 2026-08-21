"""Frozen external NA-ion test for curve-aware PASS-Cal.

All curve-distance weights are recovered solely from the six-domain
development JSON.  NA-ion labels are not consulted until the rule is frozen.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from batterylife_curve_aware_support import evaluate_domain, load_cells, paired_cell_summary, summarize


DEV = Path(__file__).with_name("batterylife_curve_aware_support.json")
OUTPUT = Path(__file__).with_name("batterylife_curve_aware_naion_frozen.json")


def frozen_weight(dev, horizon, k):
    records = [r for r in dev["nested_results"] if r["horizon"] == horizon and r["k"] == k]
    candidates = defaultdict(list)
    for record in records:
        for candidate in record["selection_candidates"]:
            weight = candidate["weight"]
            candidates[str(weight)].append(
                (candidate["development_macro_mape"], candidate["development_macro_mae"])
            )
    ranked = []
    for key, values in candidates.items():
        weight = np.inf if key == "inf" else float(key)
        ranked.append(
            (
                float(np.mean([v[0] for v in values])),
                float(np.mean([v[1] for v in values])),
                weight,
            )
        )
    return min(ranked)[2], ranked


def main():
    dev = json.load(open(DEV))
    output = {
        "development_source": DEV.name,
        "external_dataset": "BatteryLife v12 NA-ion",
        "protocol": "six-domain-frozen curve weight; NA-ion cell-level leave-one-out; support selection uses first-H-cycle curves and metadata but no life labels",
        "results": {},
    }
    for horizon in (10, 20, 50):
        cells = load_cells("NA-ion", horizon)
        for k in (1, 3, 5, 10):
            weight, candidates = frozen_weight(dev, horizon, k)
            baseline_rows = evaluate_domain(cells, k, 0.0, 100)
            method_rows = evaluate_domain(cells, k, weight, 100)
            paired = paired_cell_summary(baseline_rows, method_rows)
            baseline_ape = np.asarray([x["baseline_ape"] for x in paired])
            method_ape = np.asarray([x["method_ape"] for x in paired])
            baseline_ae = np.asarray([x["baseline_ae"] for x in paired])
            method_ae = np.asarray([x["method_ae"] for x in paired])
            output["results"][f"h{horizon}_k{k}"] = {
                "n_cells": len(cells),
                "frozen_weight": "inf" if np.isinf(weight) else weight,
                "development_candidates": [
                    {
                        "weight": "inf" if np.isinf(w) else w,
                        "macro_mape": mape,
                        "macro_mae": mae,
                    }
                    for mape, mae, w in candidates
                ],
                "protocol_only": summarize(baseline_rows),
                "curve_aware": summarize(method_rows),
                "ape_improved_same_worse": [
                    int(np.sum(method_ape < baseline_ape - 1e-9)),
                    int(np.sum(np.abs(method_ape - baseline_ape) <= 1e-9)),
                    int(np.sum(method_ape > baseline_ape + 1e-9)),
                ],
                "ape_wilcoxon_p": float(wilcoxon(baseline_ape, method_ape).pvalue),
                "ae_wilcoxon_p": float(wilcoxon(baseline_ae, method_ae).pvalue),
                "per_cell": paired,
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "weight": value["frozen_weight"],
            "protocol_only": value["protocol_only"],
            "curve_aware": value["curve_aware"],
            "ape_improved_same_worse": value["ape_improved_same_worse"],
            "ape_wilcoxon_p": value["ape_wilcoxon_p"],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
