"""Paired cell and hierarchical-domain statistics for the frozen router."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).parent
SOURCE = HERE / "batterylife_asymmetric_cohort_router.json"
OUTPUT = HERE / "batterylife_asymmetric_cohort_router_stats.json"
BOOTSTRAPS = 20_000


def paired_p(a, b):
    difference = np.asarray(a) - np.asarray(b)
    if np.all(np.abs(difference) <= 1e-12):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def hierarchical_interval(records, seed):
    rng = np.random.default_rng(seed)
    domain_rows = []
    for record in records:
        domain_rows.append(
            np.asarray(
                [
                    [row["baseline_ape"], row["method_ape"]]
                    for row in record["per_cell"]
                ],
                dtype=float,
            )
        )
    reductions = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        sampled_domains = rng.integers(0, len(domain_rows), len(domain_rows))
        domain_means = []
        for domain_index in sampled_domains:
            rows = domain_rows[domain_index]
            sampled_cells = rng.integers(0, len(rows), len(rows))
            domain_means.append(np.mean(rows[sampled_cells], axis=0))
        baseline, method = np.mean(domain_means, axis=0)
        reductions[draw] = (baseline - method) / max(baseline, 1e-12)
    return {
        "relative_mape_reduction_percentile_95": [
            float(100.0 * np.percentile(reductions, 2.5)),
            float(100.0 * np.percentile(reductions, 97.5)),
        ],
        "bootstrap_probability_reduction_nonpositive": float(
            np.mean(reductions <= 0.0)
        ),
        "n_bootstraps": BOOTSTRAPS,
    }


def main():
    source = json.load(open(SOURCE))
    output = {
        "source": SOURCE.name,
        "unit": "cell averages first; domain-macro aggregation; paired cell Wilcoxon; hierarchical domain/cell bootstrap",
        "results": {},
    }
    settings = sorted(
        {(record["horizon"], record["label_budget_k"]) for record in source["results"]}
    )
    for horizon, budget in settings:
        records = [
            record
            for record in source["results"]
            if record["horizon"] == horizon and record["label_budget_k"] == budget
        ]
        baseline, method = [], []
        per_domain = []
        for record in records:
            a = [row["baseline_ape"] for row in record["per_cell"]]
            b = [row["method_ape"] for row in record["per_cell"]]
            baseline.extend(a)
            method.extend(b)
            per_domain.append(
                {
                    "target": record["target"],
                    "baseline_mape": record["baseline"]["mape"],
                    "method_mape": record["method"]["mape"],
                    "relative_change_percent": float(
                        100.0
                        * (record["method"]["mape"] - record["baseline"]["mape"])
                        / max(record["baseline"]["mape"], 1e-12)
                    ),
                    "paired_wilcoxon_p": paired_p(a, b),
                    "improved_same_worse_cells": [
                        int(np.sum(np.asarray(b) < np.asarray(a) - 1e-12)),
                        int(np.sum(np.abs(np.asarray(b) - np.asarray(a)) <= 1e-12)),
                        int(np.sum(np.asarray(b) > np.asarray(a) + 1e-12)),
                    ],
                }
            )
        baseline = np.asarray(baseline)
        method = np.asarray(method)
        macro_baseline = float(np.mean([record["baseline"]["mape"] for record in records]))
        macro_method = float(np.mean([record["method"]["mape"] for record in records]))
        result = {
            "macro_baseline_mape": macro_baseline,
            "macro_method_mape": macro_method,
            "macro_relative_reduction_percent": float(
                100.0 * (macro_baseline - macro_method) / max(macro_baseline, 1e-12)
            ),
            "pooled_cell_wilcoxon_p": paired_p(baseline, method),
            "improved_same_worse_cells": [
                int(np.sum(method < baseline - 1e-12)),
                int(np.sum(np.abs(method - baseline) <= 1e-12)),
                int(np.sum(method > baseline + 1e-12)),
            ],
            "improved_same_worse_domains": [
                int(sum(row["relative_change_percent"] < -1e-10 for row in per_domain)),
                int(sum(abs(row["relative_change_percent"]) <= 1e-10 for row in per_domain)),
                int(sum(row["relative_change_percent"] > 1e-10 for row in per_domain)),
            ],
            "per_domain": per_domain,
        }
        result.update(hierarchical_interval(records, 20260819 + 100 * horizon + budget))
        output["results"][f"h{horizon}_k{budget}"] = result
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "macro": [value["macro_baseline_mape"], value["macro_method_mape"]],
            "relative_reduction_percent": value["macro_relative_reduction_percent"],
            "hierarchical_ci": value["relative_mape_reduction_percentile_95"],
            "domains": value["improved_same_worse_domains"],
            "pooled_p": value["pooled_cell_wilcoxon_p"],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
