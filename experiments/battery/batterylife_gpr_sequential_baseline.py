"""Sequential GPR active-learning baseline for the APACE-Cal protocol.

This is a comparison method, not part of the frozen APACE-Cal method.  It uses
the identical 70/30 episode splits and random support draw.  The active arm
keeps the first random support cell, then repeatedly fits a constant-times-RBF
Gaussian process on observed log life and acquires the candidate with maximum
posterior standard deviation.  Final log-life predictions are conservatively
bounded by the observed support-label range; the original unbounded failure is
retained separately.  Outer leave-one-domain-out evaluation chooses the
protocol/curve feature weight without using target-domain labels.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_gpr_sequential_baseline.json"
FEATURE_WEIGHTS = (0.0, 0.5, 2.0, math.inf)


def key_number(value):
    return "inf" if math.isinf(value) else f"{value:g}"


def standardized_block(matrix):
    matrix = np.asarray(matrix, dtype=float)
    center = np.zeros(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        if finite.size:
            center[column] = np.median(finite)
    scale = robust_scale(matrix)
    filled = np.where(np.isfinite(matrix), matrix, center)
    return (filled - center) / scale


def feature_matrix(protocol, curve, weight):
    """Create a target-cohort-only embedding; no life labels are accessed."""
    p = standardized_block(protocol) / math.sqrt(protocol.shape[1])
    c = standardized_block(curve) / math.sqrt(curve.shape[1])
    if math.isinf(weight):
        return c
    if weight <= 0:
        return p
    return np.c_[p, math.sqrt(weight) * c]


def fit_gp(x, y, optimize=True):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(0.05, 20.0)
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        n_restarts_optimizer=0,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, np.log(y))
    return model


def sequential_support(x, truth, acquisition, k, first, tie_rank):
    selected = [int(first)]
    while len(selected) < k:
        model = fit_gp(
            x[selected], truth[selected], optimize=len(selected) >= 2
        )
        remaining = np.asarray(
            [index for index in acquisition if index not in selected], dtype=int
        )
        _, std = model.predict(x[remaining], return_std=True)
        # Highest uncertainty wins; the episode-specific rank resolves exact ties.
        choice = min(
            range(len(remaining)),
            key=lambda pos: (-float(std[pos]), int(tie_rank[remaining[pos]])),
        )
        selected.append(int(remaining[choice]))
    return np.asarray(selected, dtype=int)


def gp_prediction(x, truth, support, test):
    model = fit_gp(x[support], truth[support], optimize=len(support) >= 2)
    support_log_life = np.log(truth[support])
    prediction = model.predict(x[test])
    prediction = np.clip(
        prediction, np.min(support_log_life), np.max(support_log_life)
    )
    return np.exp(prediction)


def evaluate(cells, horizon, budget, seeds):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    features = {
        weight: feature_matrix(protocol, curve, weight)
        for weight in FEATURE_WEIGHTS
    }
    storage = {
        (arm, weight): defaultdict(
            lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0}
        )
        for arm in ("random_gpr", "sequential_gpr")
        for weight in FEATURE_WEIGHTS
    }
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(
            len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget)
        )
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        for weight, x in features.items():
            active_support = sequential_support(
                x, truth, acquisition, k, random_support[0], tie_rank
            )
            for arm, support in (
                ("random_gpr", random_support),
                ("sequential_gpr", active_support),
            ):
                prediction = gp_prediction(x, truth, support, test)
                ae = np.abs(prediction - truth[test])
                ape = 100.0 * ae / np.maximum(truth[test], 1.0)
                for position, held in enumerate(test):
                    row = storage[(arm, weight)][names[held]]
                    row["ae_sum"] += float(ae[position])
                    row["ape_sum"] += float(ape[position])
                    row["count"] += 1
    summaries, per_cell = {}, {}
    for arm in ("random_gpr", "sequential_gpr"):
        summaries[arm], per_cell[arm] = {}, {}
        for weight in FEATURE_WEIGHTS:
            key = key_number(weight)
            rows = []
            for name in names:
                value = storage[(arm, weight)][name]
                if value["count"]:
                    rows.append(
                        {
                            "held_out": name,
                            "ae": value["ae_sum"] / value["count"],
                            "ape": value["ape_sum"] / value["count"],
                            "test_episodes": value["count"],
                        }
                    )
            per_cell[arm][key] = rows
            summaries[arm][key] = {
                "mae": float(np.mean([row["ae"] for row in rows])),
                "mape": float(np.mean([row["ape"] for row in rows])),
                "n_cells": len(rows),
            }
    return {"summaries": summaries, "per_cell": per_cell}


def choose(development, arm):
    ranking = []
    for weight in FEATURE_WEIGHTS:
        key = key_number(weight)
        values = [record["summaries"][arm][key] for record in development]
        ranking.append(
            {
                "feature_weight": key,
                "development_macro_mape": float(
                    np.mean([value["mape"] for value in values])
                ),
                "development_macro_mae": float(
                    np.mean([value["mae"] for value in values])
                ),
            }
        )
    return min(
        ranking,
        key=lambda row: (
            row["development_macro_mape"], row["development_macro_mae"]
        ),
    ), ranking


def run(horizons, budgets, seeds, output_path):
    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    cache = {
        (h, d, k): evaluate(loaded[(h, d)], h, k, seeds)
        for h in horizons
        for d in DATASETS
        for k in budgets
    }
    results = []
    for horizon in horizons:
        for target in DATASETS:
            development_domains = [domain for domain in DATASETS if domain != target]
            for budget in budgets:
                development = [
                    cache[(horizon, domain, budget)]
                    for domain in development_domains
                ]
                target_eval = cache[(horizon, target, budget)]
                selected = {}
                rankings = {}
                rows_by_arm = {}
                for arm in ("random_gpr", "sequential_gpr"):
                    selected[arm], rankings[arm] = choose(development, arm)
                    weight = selected[arm]["feature_weight"]
                    rows_by_arm[arm] = {
                        row["held_out"]: row
                        for row in target_eval["per_cell"][arm][weight]
                    }
                common = sorted(set(rows_by_arm["random_gpr"]) & set(
                    rows_by_arm["sequential_gpr"]
                ))
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "label_budget_k": budget,
                        "selected": selected,
                        "rankings": rankings,
                        "random_gpr": target_eval["summaries"]["random_gpr"][
                            selected["random_gpr"]["feature_weight"]
                        ],
                        "sequential_gpr": target_eval["summaries"]["sequential_gpr"][
                            selected["sequential_gpr"]["feature_weight"]
                        ],
                        "per_cell": [
                            {
                                "held_out": name,
                                "random_gpr_ape": rows_by_arm["random_gpr"][name]["ape"],
                                "sequential_gpr_ape": rows_by_arm["sequential_gpr"][name]["ape"],
                            }
                            for name in common
                        ],
                    }
                )
    output = {
        "dataset_version": "BatteryLife v12 structured-protocol-v2",
        "protocol": (
            f"{seeds} common APACE 70/30 episodes; total K labels; first support "
            "shared with passive random draw; sequential maximum GPR posterior "
            "standard deviation; constant-times-isotropic-RBF kernel; predicted "
            "log life bounded by observed support range; target labels "
            "excluded from outer LODO feature-weight selection"
        ),
        "results": results,
    }
    for horizon in horizons:
        for budget in budgets:
            records = [
                record for record in results
                if record["horizon"] == horizon
                and record["label_budget_k"] == budget
            ]
            output[f"macro_h{horizon}_k{budget}"] = {
                "random_gpr_mape": float(np.mean([
                    record["random_gpr"]["mape"] for record in records
                ])),
                "sequential_gpr_mape": float(np.mean([
                    record["sequential_gpr"]["mape"] for record in records
                ])),
            }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(
        {key: value for key, value in output.items() if key.startswith("macro_")},
        indent=2,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(tuple(args.horizons), tuple(args.budgets), args.seeds, args.output)


if __name__ == "__main__":
    main()
