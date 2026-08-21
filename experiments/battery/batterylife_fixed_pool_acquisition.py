"""Active calibration-cell acquisition under a strict total-K label budget.

For every episode the target domain is split, before labels are revealed, into
an acquisition pool and a common test pool.  A selector chooses K cells from
the acquisition pool using metadata/first-H curves only; all methods are then
evaluated on the identical test cells.  Candidate rules are selected by outer
leave-one-dataset-out performance on the other five domains.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import DATASETS, load_cells, nan_rms, robust_scale


OUTPUT = Path(__file__).with_suffix(".json")
HORIZONS = (10, 20, 50)
BUDGETS = (1, 3, 5, 10)
SEEDS = 100
WEIGHTS = (0.125, 0.25, 0.5, 1.0, 2.0, math.inf)


def distance_matrix(matrix, scale):
    output = np.zeros((len(matrix), len(matrix)), dtype=float)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            output[i, j] = output[j, i] = nan_rms(matrix[i], matrix[j], scale)
    return output


def facility_medoids(distance, pool, k, tie_rank):
    """Greedy k-median facility selection with randomized deterministic ties."""
    selected = []
    current = np.full(len(pool), np.inf)
    for _ in range(min(k, len(pool))):
        candidates = []
        for candidate in pool:
            if candidate in selected:
                continue
            candidate_cost = float(np.sum(np.minimum(current, distance[pool, candidate])))
            candidates.append((candidate_cost, tie_rank[candidate], candidate))
        _, _, chosen = min(candidates)
        selected.append(chosen)
        current = np.minimum(current, distance[pool, chosen])
    return np.asarray(selected, dtype=int)


def predict_logmean(support_y, n_test):
    return np.full(n_test, np.exp(np.mean(np.log(support_y))))


def predict_kernel(distance, test, support, support_y):
    local = distance[np.ix_(test, support)]
    weights = np.exp(-0.5 * np.square(local))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.exp(weights @ np.log(support_y))


def key_weight(weight):
    return "inf" if math.isinf(weight) else f"{weight:g}"


def candidate_names():
    baseline = (
        "random_logmean",
        "random_protocol_kernel",
        "protocol_medoid_logmean",
        "protocol_medoid_kernel",
    )
    curve = tuple(
        f"curve_medoid_w{key_weight(weight)}_{predictor}"
        for weight in WEIGHTS
        for predictor in ("logmean", "kernel")
    )
    return baseline, baseline + curve


def evaluate_domain(cells, horizon, budget):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    baseline_names, all_names = candidate_names()
    aggregate = {
        candidate: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for candidate in all_names
    }
    actual_support_sizes = []

    for seed in range(1, SEEDS + 1):
        rng = np.random.default_rng(20_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        support_k = min(budget, len(acquisition))
        actual_support_sizes.append(support_k)

        # Scale estimation and acquisition use acquisition-pool features only;
        # test curves do not influence which cells receive labels.
        p_scale = robust_scale(protocol[acquisition])
        c_scale = robust_scale(curve[acquisition])
        dp = distance_matrix(protocol, p_scale)
        dc = distance_matrix(curve, c_scale)
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))

        random_support = np.sort(rng.choice(acquisition, size=support_k, replace=False))
        protocol_support = facility_medoids(dp, acquisition, support_k, tie_rank)
        predictions = {
            "random_logmean": predict_logmean(truth[random_support], len(test)),
            "random_protocol_kernel": predict_kernel(
                dp, test, random_support, truth[random_support]
            ),
            "protocol_medoid_logmean": predict_logmean(truth[protocol_support], len(test)),
            "protocol_medoid_kernel": predict_kernel(
                dp, test, protocol_support, truth[protocol_support]
            ),
        }
        for weight in WEIGHTS:
            distance = dc if math.isinf(weight) else np.sqrt(np.square(dp) + weight * np.square(dc))
            support = facility_medoids(distance, acquisition, support_k, tie_rank)
            prefix = f"curve_medoid_w{key_weight(weight)}"
            predictions[f"{prefix}_logmean"] = predict_logmean(truth[support], len(test))
            predictions[f"{prefix}_kernel"] = predict_kernel(distance, test, support, truth[support])

        for candidate, prediction in predictions.items():
            ae = np.abs(prediction - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for position, index in enumerate(test):
                record = aggregate[candidate][names[index]]
                record["ae_sum"] += float(ae[position])
                record["ape_sum"] += float(ape[position])
                record["count"] += 1

    summaries, per_cell = {}, {}
    for candidate in all_names:
        rows = []
        for name in names:
            record = aggregate[candidate][name]
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
        }
    return summaries, per_cell, baseline_names, all_names, int(np.median(actual_support_sizes))


def choose(scores, horizon, target, budget, candidates):
    development = [dataset for dataset in DATASETS if dataset != target]
    ranked = []
    for candidate in candidates:
        values = [scores[(horizon, dataset, budget)][candidate] for dataset in development]
        ranked.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([value["mape"] for value in values])),
                "development_macro_mae": float(np.mean([value["mae"] for value in values])),
            }
        )
    return min(ranked, key=lambda row: (row["development_macro_mape"], row["development_macro_mae"])), ranked


def main():
    loaded = {(h, d): load_cells(d, h) for h in HORIZONS for d in DATASETS}
    scores, cells, effective = {}, {}, {}
    baseline_names = all_names = None
    for horizon in HORIZONS:
        for dataset in DATASETS:
            for budget in BUDGETS:
                summary, per_cell, baseline_names, all_names, effective_k = evaluate_domain(
                    loaded[(horizon, dataset)], horizon, budget
                )
                scores[(horizon, dataset, budget)] = summary
                cells[(horizon, dataset, budget)] = per_cell
                effective[(horizon, dataset, budget)] = effective_k

    results = []
    for horizon in HORIZONS:
        for target in DATASETS:
            for budget in BUDGETS:
                baseline_choice, baseline_ranking = choose(
                    scores, horizon, target, budget, baseline_names
                )
                method_choice, method_ranking = choose(scores, horizon, target, budget, all_names)
                baseline, method = baseline_choice["candidate"], method_choice["candidate"]
                baseline_rows = {row["held_out"]: row for row in cells[(horizon, target, budget)][baseline]}
                method_rows = {row["held_out"]: row for row in cells[(horizon, target, budget)][method]}
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "label_budget_k": budget,
                        "effective_k": effective[(horizon, target, budget)],
                        "selected_baseline": baseline_choice,
                        "selected_method": method_choice,
                        "baseline": scores[(horizon, target, budget)][baseline],
                        "method": scores[(horizon, target, budget)][method],
                        "method_uses_curve": method.startswith("curve_"),
                        "per_cell": [
                            {
                                "held_out": name,
                                "baseline_ae": baseline_rows[name]["ae"],
                                "baseline_ape": baseline_rows[name]["ape"],
                                "method_ae": method_rows[name]["ae"],
                                "method_ape": method_rows[name]["ape"],
                            }
                            for name in sorted(baseline_rows)
                        ],
                        "baseline_ranking": baseline_ranking,
                        "method_ranking": method_ranking,
                    }
                )
    output = {
        "dataset_version": "BatteryLife v12",
        "protocol": (
            "100 episodes; 70% unlabeled acquisition pool and common 30% test pool; "
            "K selected once without life labels; test features excluded from selection; "
            "cell aggregation then outer LODO candidate selection"
        ),
        "results": results,
    }
    for horizon in HORIZONS:
        for budget in BUDGETS:
            records = [
                record
                for record in results
                if record["horizon"] == horizon and record["label_budget_k"] == budget
            ]
            output[f"macro_h{horizon}_k{budget}"] = {
                "baseline_mae": float(np.mean([record["baseline"]["mae"] for record in records])),
                "method_mae": float(np.mean([record["method"]["mae"] for record in records])),
                "baseline_mape": float(np.mean([record["baseline"]["mape"] for record in records])),
                "method_mape": float(np.mean([record["method"]["mape"] for record in records])),
                "curve_selected_domains": int(sum(record["method_uses_curve"] for record in records)),
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key.startswith("macro_")}, indent=2))


if __name__ == "__main__":
    main()
