"""E9 hierarchical statistics for the frozen APACE-Cal v2 result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).parent
SOURCE = HERE / "batterylife_asymmetric_cohort_router_v2.json"
OUT = HERE / "batterylife_apace_v2_stats.json"
BOOTSTRAPS = 10000


def main():
    source = json.loads(SOURCE.read_text())
    output = {
        "source": SOURCE.name,
        "unit": "one prediction per cell; domain macro first; paired hierarchical domain/cell bootstrap",
        "multiple_comparison_plan": "H50/K3 primary; remaining 11 settings secondary and Holm-corrected",
        "results": {},
    }
    for horizon in (10, 20, 50):
        for budget in (1, 3, 5, 10):
            records = [r for r in source["results"]
                       if r["horizon"] == horizon and r["label_budget_k"] == budget]
            arrays = [
                np.asarray([[row["baseline_ape"], row["method_ape"]]
                            for row in record["per_cell"]], dtype=float)
                for record in records
            ]
            paired_a = np.concatenate([a[:, 0] for a in arrays])
            paired_b = np.concatenate([a[:, 1] for a in arrays])
            rng = np.random.default_rng(20260820 + 100 * horizon + budget)
            reductions = np.empty(BOOTSTRAPS, dtype=float)
            for draw in range(BOOTSTRAPS):
                sampled = rng.integers(0, len(arrays), len(arrays))
                domain_means = []
                for index in sampled:
                    rows = arrays[index]
                    cell_indices = rng.integers(0, len(rows), len(rows))
                    domain_means.append(np.mean(rows[cell_indices], axis=0))
                baseline, method = np.mean(domain_means, axis=0)
                reductions[draw] = 100.0 * (baseline - method) / max(baseline, 1e-12)
            domain_relative = [
                100.0 * (np.mean(a[:, 0]) - np.mean(a[:, 1])) / max(np.mean(a[:, 0]), 1e-12)
                for a in arrays
            ]
            if np.all(np.abs(paired_a - paired_b) <= 1e-12):
                p_value = 1.0
            else:
                try:
                    p_value = float(wilcoxon(paired_a, paired_b).pvalue)
                except ValueError:
                    p_value = 1.0
            output["results"][f"h{horizon}_k{budget}"] = {
                "macro_baseline_mape": float(np.mean([np.mean(a[:, 0]) for a in arrays])),
                "macro_method_mape": float(np.mean([np.mean(a[:, 1]) for a in arrays])),
                "relative_reduction_percent": float(np.mean(domain_relative)),
                "hierarchical_ci95_percent": [
                    float(np.percentile(reductions, 2.5)),
                    float(np.percentile(reductions, 97.5)),
                ],
                "bootstrap_probability_nonpositive": float(np.mean(reductions <= 0.0)),
                "pooled_cell_wilcoxon_p": p_value,
                "domain_relative_reductions_percent": domain_relative,
                "improved_same_worse_domains": [
                    int(sum(x > 1e-10 for x in domain_relative)),
                    int(sum(abs(x) <= 1e-10 for x in domain_relative)),
                    int(sum(x < -1e-10 for x in domain_relative)),
                ],
                "n_domains": len(arrays),
                "n_cells": int(len(paired_a)),
            }
    # Holm correction over the 11 non-primary pooled Wilcoxon p-values.
    secondary = [(key, value["pooled_cell_wilcoxon_p"])
                 for key, value in output["results"].items() if key != "h50_k3"]
    ordered = sorted(secondary, key=lambda item: item[1])
    for rank, (key, p_value) in enumerate(ordered):
        output["results"][key]["holm_p_upper_bound"] = float(min(1.0, p_value * (len(ordered) - rank)))
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for key in ("h10_k3", "h20_k3", "h50_k3"):
        print(key, output["results"][key])


if __name__ == "__main__":
    main()
