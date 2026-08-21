"""Post-blind paired cell bootstrap for the frozen SNU Dataset 1 result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=HERE / "snu_dynamic_dataset1_blind_eval.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "snu_dynamic_dataset1_blind_stats.json")
    parser.add_argument("--repetitions", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    row = next(x for x in data["results"]
               if x["horizon"] == 50 and x["label_budget_k"] == 3)
    baseline = np.asarray([x["baseline_ape"] for x in row["per_cell"]])
    method = np.asarray([x["method_ape"] for x in row["per_cell"]])
    difference = baseline - method
    # Recover life only for distribution auditing. For nonzero APE,
    # truth = 100*AE/APE by definition; exact-error rows are omitted.
    life = np.asarray([100.0 * x["baseline_ae"] / x["baseline_ape"]
                       for x in row["per_cell"] if x["baseline_ape"] > 1e-12])
    rng = np.random.default_rng(args.seed)
    n = len(difference)
    draws = rng.integers(0, n, size=(args.repetitions, n))
    delta = difference[draws].mean(axis=1)
    relative = 100.0 * delta / np.maximum(baseline[draws].mean(axis=1), 1e-12)
    output = {
        "status": "POST-BLIND INFERENCE; METHOD AND MANIFEST ALREADY FROZEN",
        "setting": "SNU Dataset 1 H50/K3",
        "n_cells": n,
        "paired_mape_reduction_percentage_points": float(difference.mean()),
        "paired_cell_bootstrap_mape_reduction_95_ci":
            np.percentile(delta, [2.5, 97.5]).tolist(),
        "relative_mape_reduction_percent": float(
            100.0 * difference.mean() / baseline.mean()),
        "paired_cell_bootstrap_relative_reduction_95_ci":
            np.percentile(relative, [2.5, 97.5]).tolist(),
        "life_distribution_recovered_from_error_identity": {
            "n_recoverable": len(life), "min": float(np.min(life)),
            "median": float(np.median(life)), "max": float(np.max(life)),
            "mean": float(np.mean(life)), "std": float(np.std(life)),
            "coefficient_of_variation": float(np.std(life) / np.mean(life)),
        },
        "repetitions": args.repetitions, "seed": args.seed,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
