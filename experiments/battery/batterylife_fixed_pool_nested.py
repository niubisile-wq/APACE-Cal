"""Strict domain-level K-shot audit with one fixed calibration pool per episode.

Each episode labels K target-domain cells once.  Every remaining cell shares
that same labeled pool, preventing the union of query-specific supports from
silently exceeding the K-label budget.  Hyperparameters/method family are
selected by outer leave-one-dataset-out macro error on the other five domains.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from batterylife_curve_aware_support import (
    DATASETS,
    combined_distance,
    load_cells,
    nan_rms,
    robust_scale,
)


OUTPUT = Path(__file__).with_suffix(".json")
HORIZONS = (10, 20, 50)
KS = (1, 3, 5, 10)
SEEDS = 100
ALPHAS = (0.1, 1.0, 10.0, 100.0)
TAUS = (0.5, 1.0, 2.0)
CURVE_WEIGHTS = (0.125, 0.25, 0.5, 1.0, 2.0, math.inf)


def design(matrix, scale):
    matrix = np.asarray(matrix, dtype=float)
    normalized = matrix / scale
    return np.c_[np.nan_to_num(normalized, nan=0.0), (~np.isfinite(matrix)).astype(float)]


def predict_ridge(train_x, train_y, test_x, alpha):
    if len(train_y) == 1:
        return np.full(len(test_x), float(train_y[0]))
    model = Ridge(alpha=alpha).fit(train_x, np.log(train_y))
    # With at most ten labels, unconstrained high-dimensional extrapolation can
    # overflow despite ridge shrinkage.  Calibration is therefore restricted
    # to interpolation within the observed support-life range.
    log_prediction = np.clip(model.predict(test_x), np.log(train_y).min(), np.log(train_y).max())
    return np.exp(log_prediction)


def distance_matrix(query, support, scale):
    return np.asarray([[nan_rms(q, s, scale) for s in support] for q in query])


def all_pair_distances(matrix, scale):
    output = np.zeros((len(matrix), len(matrix)), dtype=float)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            output[i, j] = output[j, i] = nan_rms(matrix[i], matrix[j], scale)
    return output


def kernel_prediction(distance, support_y, tau):
    weights = np.exp(-0.5 * np.square(distance / tau))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.exp(weights @ np.log(support_y))


def candidate_names():
    baseline = ["logmean"]
    baseline += [f"protocol_ridge_a{alpha:g}" for alpha in ALPHAS]
    baseline += [f"protocol_kernel_t{tau:g}" for tau in TAUS]
    curve = []
    for family in ("curve", "hybrid"):
        curve += [f"{family}_ridge_a{alpha:g}" for alpha in ALPHAS]
    curve += [f"curve_kernel_t{tau:g}" for tau in TAUS]
    curve += [
        f"hybrid_kernel_w{'inf' if math.isinf(weight) else f'{weight:g}'}"
        for weight in CURVE_WEIGHTS
    ]
    return baseline, baseline + curve


def evaluate_domain(cells, horizon, k, seeds=SEEDS):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    p_scale, c_scale = robust_scale(protocol), robust_scale(curve)
    p_design, c_design = design(protocol, p_scale), design(curve, c_scale)
    hybrid_design = np.c_[p_design, c_design]
    all_dp = all_pair_distances(protocol, p_scale)
    all_dc = all_pair_distances(curve, c_scale)
    baseline_names, all_names = candidate_names()
    cell_errors = {
        candidate: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for candidate in all_names
    }

    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(10_000_000 * horizon + 10_000 * k + seed)
        support = np.sort(rng.choice(len(cells), size=min(k, len(cells) - 1), replace=False))
        test = np.asarray([index for index in range(len(cells)) if index not in set(support)])
        support_y = truth[support]
        predictions = {"logmean": np.full(len(test), np.exp(np.mean(np.log(support_y))))}
        for alpha in ALPHAS:
            predictions[f"protocol_ridge_a{alpha:g}"] = predict_ridge(
                p_design[support], support_y, p_design[test], alpha
            )
            predictions[f"curve_ridge_a{alpha:g}"] = predict_ridge(
                c_design[support], support_y, c_design[test], alpha
            )
            predictions[f"hybrid_ridge_a{alpha:g}"] = predict_ridge(
                hybrid_design[support], support_y, hybrid_design[test], alpha
            )
        dp = all_dp[np.ix_(test, support)]
        dc = all_dc[np.ix_(test, support)]
        for tau in TAUS:
            predictions[f"protocol_kernel_t{tau:g}"] = kernel_prediction(dp, support_y, tau)
            predictions[f"curve_kernel_t{tau:g}"] = kernel_prediction(dc, support_y, tau)
        for weight in CURVE_WEIGHTS:
            if math.isinf(weight):
                distance = dc
            else:
                distance = np.sqrt(np.square(dp) + weight * np.square(dc))
            key = f"hybrid_kernel_w{'inf' if math.isinf(weight) else f'{weight:g}'}"
            predictions[key] = kernel_prediction(distance, support_y, 1.0)

        for candidate, prediction in predictions.items():
            ae = np.abs(prediction - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for position, index in enumerate(test):
                record = cell_errors[candidate][names[index]]
                record["ae_sum"] += float(ae[position])
                record["ape_sum"] += float(ape[position])
                record["count"] += 1

    summaries, per_cell = {}, {}
    for candidate in all_names:
        rows = []
        for name in names:
            record = cell_errors[candidate][name]
            if record["count"]:
                rows.append(
                    {
                        "held_out": name,
                        "ae": record["ae_sum"] / record["count"],
                        "ape": record["ape_sum"] / record["count"],
                        "test_episodes": record["count"],
                    }
                )
        per_cell[candidate] = rows
        summaries[candidate] = {
            "mae": float(np.mean([row["ae"] for row in rows])),
            "mape": float(np.mean([row["ape"] for row in rows])),
            "n_cells": len(rows),
            "mean_test_episodes": float(np.mean([row["test_episodes"] for row in rows])),
        }
    return summaries, per_cell, baseline_names, all_names


def choose_candidate(scores, horizon, target, k, candidates):
    development = [dataset for dataset in DATASETS if dataset != target]
    ranked = []
    for candidate in candidates:
        domain = [scores[(horizon, dataset, k)][candidate] for dataset in development]
        ranked.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([value["mape"] for value in domain])),
                "development_macro_mae": float(np.mean([value["mae"] for value in domain])),
            }
        )
    return min(ranked, key=lambda row: (row["development_macro_mape"], row["development_macro_mae"])), ranked


def main():
    loaded = {(h, d): load_cells(d, h) for h in HORIZONS for d in DATASETS}
    scores, cell_cache = {}, {}
    baseline_names = all_names = None
    for horizon in HORIZONS:
        for dataset in DATASETS:
            for k in KS:
                summary, per_cell, baseline_names, all_names = evaluate_domain(
                    loaded[(horizon, dataset)], horizon, k
                )
                scores[(horizon, dataset, k)] = summary
                cell_cache[(horizon, dataset, k)] = per_cell

    results = []
    for horizon in HORIZONS:
        for target in DATASETS:
            for k in KS:
                baseline_choice, baseline_ranking = choose_candidate(
                    scores, horizon, target, k, baseline_names
                )
                method_choice, method_ranking = choose_candidate(scores, horizon, target, k, all_names)
                baseline = baseline_choice["candidate"]
                method = method_choice["candidate"]
                by_candidate = cell_cache[(horizon, target, k)]
                baseline_cell = {row["held_out"]: row for row in by_candidate[baseline]}
                method_cell = {row["held_out"]: row for row in by_candidate[method]}
                paired = [
                    {
                        "held_out": name,
                        "baseline_ae": baseline_cell[name]["ae"],
                        "baseline_ape": baseline_cell[name]["ape"],
                        "method_ae": method_cell[name]["ae"],
                        "method_ape": method_cell[name]["ape"],
                    }
                    for name in sorted(baseline_cell)
                ]
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "k": k,
                        "selected_baseline": baseline_choice,
                        "selected_method": method_choice,
                        "baseline": scores[(horizon, target, k)][baseline],
                        "method": scores[(horizon, target, k)][method],
                        "method_uses_curve": "curve" in method or "hybrid" in method,
                        "per_cell": paired,
                        "baseline_ranking": baseline_ranking,
                        "method_ranking": method_ranking,
                    }
                )
    output = {
        "dataset_version": "BatteryLife v12",
        "protocol": (
            "100 fixed-pool episodes per domain/H/K; K cells labeled once per episode; "
            "all other cells share that pool; cell-aggregated metrics; outer LODO "
            "candidate selection on other five domains; ridge calibration predictions "
            "bounded to the observed support-life range"
        ),
        "results": results,
    }
    for horizon in HORIZONS:
        for k in KS:
            records = [r for r in results if r["horizon"] == horizon and r["k"] == k]
            output[f"macro_h{horizon}_k{k}"] = {
                "baseline_mae": float(np.mean([r["baseline"]["mae"] for r in records])),
                "method_mae": float(np.mean([r["method"]["mae"] for r in records])),
                "baseline_mape": float(np.mean([r["baseline"]["mape"] for r in records])),
                "method_mape": float(np.mean([r["method"]["mape"] for r in records])),
                "curve_selected_domains": int(sum(r["method_uses_curve"] for r in records)),
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key.startswith("macro_")}, indent=2))
    for record in results:
        print(
            record["horizon"],
            record["target"],
            record["k"],
            record["selected_baseline"]["candidate"],
            record["selected_method"]["candidate"],
        )


if __name__ == "__main__":
    main()
