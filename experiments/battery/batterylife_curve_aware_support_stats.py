"""Paired and hierarchical-bootstrap audit for curve-aware PASS-Cal."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SOURCE = Path(__file__).with_name("batterylife_curve_aware_support.json")
OUTPUT = Path(__file__).with_name("batterylife_curve_aware_support_stats.json")


def hierarchical_bootstrap(records, metric, repeats=20_000, seed=20260819):
    rng = np.random.default_rng(seed)
    datasets = sorted({record["target"] for record in records})
    by_dataset = {dataset: next(record for record in records if record["target"] == dataset)["per_cell"] for dataset in datasets}
    reductions = np.empty(repeats)
    for repeat in range(repeats):
        sampled_datasets = rng.choice(datasets, len(datasets), replace=True)
        baseline_domain, method_domain = [], []
        for dataset in sampled_datasets:
            cells = by_dataset[dataset]
            indices = rng.integers(0, len(cells), len(cells))
            baseline_domain.append(np.mean([cells[i][f"baseline_{metric}"] for i in indices]))
            method_domain.append(np.mean([cells[i][f"method_{metric}"] for i in indices]))
        baseline, method = np.mean(baseline_domain), np.mean(method_domain)
        reductions[repeat] = 100.0 * (baseline - method) / baseline
    return {
        "relative_reduction_percent": float(np.mean(reductions)),
        "ci95": [float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5))],
        "probability_reduction_positive": float(np.mean(reductions > 0)),
        "repeats": repeats,
    }


def main():
    source = json.load(open(SOURCE))
    output = {
        "source": SOURCE.name,
        "protocol": "paired cell-level Wilcoxon plus dataset/cell hierarchical bootstrap",
        "results": {},
    }
    for horizon in (10, 20, 50):
        for k in (1, 3, 5, 10):
            records = [r for r in source["nested_results"] if r["horizon"] == horizon and r["k"] == k]
            entry = {"domains": {}}
            for metric in ("ae", "ape"):
                baseline = np.asarray([c[f"baseline_{metric}"] for r in records for c in r["per_cell"]])
                method = np.asarray([c[f"method_{metric}"] for r in records for c in r["per_cell"]])
                delta = method - baseline
                entry[metric] = {
                    "n_cells": len(delta),
                    "improved": int(np.sum(delta < -1e-9)),
                    "same": int(np.sum(np.abs(delta) <= 1e-9)),
                    "worse": int(np.sum(delta > 1e-9)),
                    "wilcoxon_p": float(wilcoxon(baseline, method).pvalue),
                    "hierarchical_bootstrap": hierarchical_bootstrap(records, metric),
                }
            for record in records:
                before, after = record["protocol_only"], record["nested_curve_aware"]
                entry["domains"][record["target"]] = {
                    "selected_curve_weight": record["selected_curve_weight"],
                    "baseline_mae": before["mae"],
                    "method_mae": after["mae"],
                    "mae_reduction_percent": 100.0 * (before["mae"] - after["mae"]) / before["mae"],
                    "baseline_mape": before["mape"],
                    "method_mape": after["mape"],
                    "mape_reduction_percent": 100.0 * (before["mape"] - after["mape"]) / before["mape"],
                }
            entry["worst_domain"] = {
                "baseline_mae": max(v["baseline_mae"] for v in entry["domains"].values()),
                "method_mae": max(v["method_mae"] for v in entry["domains"].values()),
                "baseline_mape": max(v["baseline_mape"] for v in entry["domains"].values()),
                "method_mape": max(v["method_mape"] for v in entry["domains"].values()),
            }
            output["results"][f"h{horizon}_k{k}"] = entry
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "ae": value["ae"],
            "ape": value["ape"],
            "worst_domain": value["worst_domain"],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
