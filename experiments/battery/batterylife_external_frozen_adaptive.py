"""Evaluate the frozen adaptive curve-retrieval rule on an untouched domain."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from batterylife_curve_aware_support import evaluate_domain, load_cells, paired_cell_summary, summarize


ADAPTIVE = Path(__file__).with_name("batterylife_adaptive_effective_support.json")
BUDGETS = (1, 3, 5, 10)


def decode_weight(value):
    return math.inf if value == "inf" else float(value)


def frozen_rule(source, horizon, budget):
    records = [
        record
        for record in source["results"]
        if record["horizon"] == horizon and record["budget_k"] == budget
    ]
    scores = defaultdict(list)
    for record in records:
        for candidate in record["selection_candidates"]:
            key = (int(candidate["effective_k"]), str(candidate["weight"]))
            scores[key].append(
                (candidate["development_macro_mape"], candidate["development_macro_mae"])
            )
    ranked = []
    for (effective_k, weight), values in scores.items():
        ranked.append(
            {
                "effective_k": effective_k,
                "weight": weight,
                "macro_mape": float(np.mean([value[0] for value in values])),
                "macro_mae": float(np.mean([value[1] for value in values])),
            }
        )
    chosen = min(ranked, key=lambda item: (item["macro_mape"], item["macro_mae"]))
    return chosen, ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RWTH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_path = args.output or Path(__file__).with_name(
        f"batterylife_external_{args.dataset.lower().replace('-', '_')}_frozen_adaptive.json"
    )
    source = json.load(open(ADAPTIVE))
    output = {
        "development_source": ADAPTIVE.name,
        "external_dataset": args.dataset,
        "protocol": (
            "effective m<=K and curve weight frozen from six development domains; "
            "same-m protocol-only comparator; external query label held out; "
            "one prediction per cell averaged over 100 deterministic tie seeds"
        ),
        "results": {},
    }
    for horizon in (10, 20, 50):
        cells = load_cells(args.dataset, horizon)
        if len(cells) < 2:
            raise RuntimeError(f"Only {len(cells)} eligible {args.dataset} cells at H={horizon}")
        for budget in BUDGETS:
            chosen, candidates = frozen_rule(source, horizon, budget)
            effective_k = min(chosen["effective_k"], len(cells) - 1)
            weight = decode_weight(chosen["weight"])
            baseline_rows = evaluate_domain(cells, effective_k, 0.0, 100)
            method_rows = evaluate_domain(cells, effective_k, weight, 100)
            paired = paired_cell_summary(baseline_rows, method_rows)
            baseline_ape = np.asarray([row["baseline_ape"] for row in paired])
            method_ape = np.asarray([row["method_ape"] for row in paired])
            baseline_ae = np.asarray([row["baseline_ae"] for row in paired])
            method_ae = np.asarray([row["method_ae"] for row in paired])
            output["results"][f"h{horizon}_k{budget}"] = {
                "n_cells": len(cells),
                "label_budget_k": budget,
                "effective_k": effective_k,
                "frozen_weight": chosen["weight"],
                "development_candidates": candidates,
                "protocol_only_same_m": summarize(baseline_rows),
                "adaptive_curve_aware": summarize(method_rows),
                "ape_improved_same_worse": [
                    int(np.sum(method_ape < baseline_ape - 1e-9)),
                    int(np.sum(np.abs(method_ape - baseline_ape) <= 1e-9)),
                    int(np.sum(method_ape > baseline_ape + 1e-9)),
                ],
                "ape_wilcoxon_p": float(wilcoxon(baseline_ape, method_ape).pvalue),
                "ae_wilcoxon_p": float(wilcoxon(baseline_ae, method_ae).pvalue),
                "per_cell": paired,
            }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "n": value["n_cells"],
            "m": value["effective_k"],
            "weight": value["frozen_weight"],
            "protocol": value["protocol_only_same_m"],
            "method": value["adaptive_curve_aware"],
            "improved_same_worse": value["ape_improved_same_worse"],
            "p": value["ape_wilcoxon_p"],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
