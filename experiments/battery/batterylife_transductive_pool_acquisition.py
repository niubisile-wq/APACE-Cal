"""Transductive cohort design under a strict total-K life-label budget.

All target cells expose protocol and first-H-cycle measurements.  Only cells
in a predeclared acquisition subset can be run to end of life; a common test
subset is never labeled.  Selection minimizes coverage distance over the full
unlabeled cohort, which mirrors choosing long-running prototypes after short
screening of an incoming batch.  Outer LODO selection excludes target labels.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import DATASETS, load_cells, nan_rms, robust_scale


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_transductive_pool_acquisition.json"
WEIGHTS = (0.0, 0.125, 0.5, 1.0, 2.0, math.inf)
BANDWIDTHS = (0.5, 1.0, 2.0)


def key_number(value):
    return "inf" if math.isinf(value) else f"{value:g}"


def distance_matrix(matrix, scale, influence_cap=3.0):
    output = np.zeros((len(matrix), len(matrix)), dtype=float)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            a, b = matrix[i], matrix[j]
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(scale) & (scale > 0)
            if ok.any():
                standardized = (a[ok] - b[ok]) / scale[ok]
                # A nearly constant sensor channel must not dominate an entire
                # cohort because of one instrument spike.  This is a fixed
                # bounded-influence metric, not a target-label-tuned threshold.
                standardized = np.clip(standardized, -influence_cap, influence_cap)
                value = float(np.sqrt(np.mean(np.square(standardized))))
            else:
                value = 0.0
            output[i, j] = output[j, i] = value
    return output


def select_facilities(distance, acquisition, clients, k, tie_rank):
    selected = []
    current = np.full(len(clients), np.inf)
    for _ in range(min(k, len(acquisition))):
        candidates = []
        for candidate in acquisition:
            if candidate in selected:
                continue
            cost = float(np.sum(np.minimum(current, distance[clients, candidate])))
            candidates.append((cost, tie_rank[candidate], int(candidate)))
        _, _, chosen = min(candidates)
        selected.append(chosen)
        current = np.minimum(current, distance[clients, chosen])
    return np.asarray(selected, dtype=int)


def kernel_prediction(distance, test, support, support_y, bandwidth):
    local = distance[np.ix_(test, support)]
    weights = np.exp(-0.5 * np.square(local / bandwidth))
    row_sum = weights.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-12
    weights /= np.maximum(row_sum, 1e-12)
    if np.any(empty):
        nearest = np.argmin(local[empty], axis=1)
        weights[empty] = 0.0
        weights[np.where(empty)[0], nearest] = 1.0
    return np.exp(weights @ np.log(support_y))


def candidate_names(prefix):
    names = [f"{prefix}_logmean"]
    for weight in WEIGHTS:
        for bandwidth in BANDWIDTHS:
            names.append(
                f"{prefix}_w{key_number(weight)}_bw{key_number(bandwidth)}"
            )
    return tuple(names)


def parse_candidate(name):
    prefix, weight_text, bandwidth_text = name.rsplit("_", 2)
    weight = math.inf if weight_text == "winf" else float(weight_text[1:])
    bandwidth = float(bandwidth_text[2:])
    return prefix, weight, bandwidth


def evaluate(cells, horizon, budget, seeds, influence_cap=3.0):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    baseline_names = candidate_names("random")
    method_names = candidate_names("medoid")
    all_names = baseline_names + method_names
    aggregate = {
        candidate: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for candidate in all_names
    }
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(50_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        clients = np.arange(len(cells), dtype=int)
        k = min(budget, len(acquisition))
        p_scale = robust_scale(protocol[clients])
        c_scale = robust_scale(curve[clients])
        dp = distance_matrix(protocol, p_scale, influence_cap)
        dc = distance_matrix(curve, c_scale, influence_cap)
        distances = {
            weight: dc
            if math.isinf(weight)
            else np.sqrt(np.square(dp) + weight * np.square(dc))
            for weight in WEIGHTS
        }
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        medoid_support = {
            weight: select_facilities(
                distance, acquisition, clients, k, tie_rank
            )
            for weight, distance in distances.items()
        }
        predictions = {
            "random_logmean": np.full(
                len(test), np.exp(np.mean(np.log(truth[random_support])))
            ),
            "medoid_logmean": None,
        }
        # A method's log-mean uses the protocol+curve configuration w=1.
        central_support = medoid_support[1.0]
        predictions["medoid_logmean"] = np.full(
            len(test), np.exp(np.mean(np.log(truth[central_support])))
        )
        for prefix, supports in (
            ("random", {weight: random_support for weight in WEIGHTS}),
            ("medoid", medoid_support),
        ):
            for weight in WEIGHTS:
                for bandwidth in BANDWIDTHS:
                    candidate = (
                        f"{prefix}_w{key_number(weight)}_bw{key_number(bandwidth)}"
                    )
                    predictions[candidate] = kernel_prediction(
                        distances[weight],
                        test,
                        supports[weight],
                        truth[supports[weight]],
                        bandwidth,
                    )
        for candidate, prediction in predictions.items():
            ae = np.abs(prediction - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for position, index in enumerate(test):
                row = aggregate[candidate][names[index]]
                row["ae_sum"] += float(ae[position])
                row["ape_sum"] += float(ape[position])
                row["count"] += 1
    summaries, per_cell = {}, {}
    for candidate in all_names:
        rows = []
        for name in names:
            value = aggregate[candidate][name]
            if value["count"]:
                rows.append(
                    {
                        "held_out": name,
                        "ae": value["ae_sum"] / value["count"],
                        "ape": value["ape_sum"] / value["count"],
                        "test_episodes": value["count"],
                    }
                )
        per_cell[candidate] = rows
        summaries[candidate] = {
            "mae": float(np.mean([row["ae"] for row in rows])),
            "mape": float(np.mean([row["ape"] for row in rows])),
            "n_cells": len(rows),
        }
    return summaries, per_cell, baseline_names, method_names


def choose(scores, candidates):
    ranking = []
    for candidate in candidates:
        values = [domain[candidate] for domain in scores]
        ranking.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([v["mape"] for v in values])),
                "development_macro_mae": float(np.mean([v["mae"] for v in values])),
            }
        )
    return min(
        ranking,
        key=lambda row: (row["development_macro_mape"], row["development_macro_mae"]),
    ), ranking


def run(horizons, budgets, seeds, influence_cap, output_path):
    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    cache = {}
    for horizon in horizons:
        for domain in DATASETS:
            for budget in budgets:
                cache[(horizon, domain, budget)] = evaluate(
                    loaded[(horizon, domain)], horizon, budget, seeds, influence_cap
                )
    results = []
    for horizon in horizons:
        for target in DATASETS:
            development = [domain for domain in DATASETS if domain != target]
            for budget in budgets:
                development_scores = [
                    cache[(horizon, domain, budget)][0] for domain in development
                ]
                target_summary, target_cells, baseline_names, method_names = cache[
                    (horizon, target, budget)
                ]
                baseline_choice, baseline_ranking = choose(
                    development_scores, baseline_names
                )
                method_choice, method_ranking = choose(development_scores, method_names)
                baseline = baseline_choice["candidate"]
                method = method_choice["candidate"]
                baseline_rows = {r["held_out"]: r for r in target_cells[baseline]}
                method_rows = {r["held_out"]: r for r in target_cells[method]}
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "label_budget_k": budget,
                        "selected_baseline": baseline_choice,
                        "selected_method": method_choice,
                        "baseline": target_summary[baseline],
                        "method": target_summary[method],
                        "target_candidate_summaries": target_summary,
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
        "dataset_version": "BatteryLife v12 structured-protocol-v2",
        "protocol": (
            f"{seeds} 70/30 episodes; common test cells; all first-H target features "
            "available transductively; only K acquisition cells labeled once; fixed "
            f"{influence_cap:g}-IQR bounded-influence feature distance; target labels excluded from "
            "outer LODO candidate selection"
        ),
        "results": results,
    }
    for horizon in horizons:
        for budget in budgets:
            records = [
                r
                for r in results
                if r["horizon"] == horizon and r["label_budget_k"] == budget
            ]
            output[f"macro_h{horizon}_k{budget}"] = {
                "baseline_mae": float(np.mean([r["baseline"]["mae"] for r in records])),
                "method_mae": float(np.mean([r["method"]["mae"] for r in records])),
                "baseline_mape": float(np.mean([r["baseline"]["mape"] for r in records])),
                "method_mape": float(np.mean([r["method"]["mape"] for r in records])),
            }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--influence-cap", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        tuple(args.horizons),
        tuple(args.budgets),
        args.seeds,
        args.influence_cap,
        args.output,
    )


if __name__ == "__main__":
    main()
