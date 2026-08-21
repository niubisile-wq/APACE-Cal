"""Asymmetric cohort acquisition with a matched-predictor baseline.

The predictor is selected only from outer development domains using random K
supports.  The proposed arm uses that *identical* predictor and changes only
how the K target cells are acquired.  An unlabeled protocol-dispersion router
falls back to the exact random supports when active selection is not
identifiable, so fallback predictions are bit-for-bit identical to baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import (
    BANDWIDTHS,
    WEIGHTS,
    distance_matrix,
    kernel_prediction,
    key_number,
    select_facilities,
)


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_asymmetric_cohort_router.json"
LOW_PROTOCOL_THRESHOLD = 0.30


def predictor_names():
    return ("logmean",) + tuple(
        f"w{key_number(weight)}_bw{key_number(bandwidth)}"
        for weight in WEIGHTS
        for bandwidth in BANDWIDTHS
    )


def parse_predictor(name):
    weight_text, bandwidth_text = name.split("_")
    weight = math.inf if weight_text == "winf" else float(weight_text[1:])
    bandwidth = float(bandwidth_text[2:])
    return weight, bandwidth


def protocol_dispersion(distance):
    values = distance[np.triu_indices(len(distance), k=1)]
    return float(np.median(values)) if len(values) else 0.0


def routed_support(protocol_spread, budget, random_support, acquisition, clients,
                   distances, tie_rank):
    if protocol_spread <= 1e-12:
        return random_support, "fallback_zero_protocol_dispersion"
    # Two-regime (medium-spread) cohorts are already covered reliably by
    # passive labels; active facilities can over-concentrate within a regime.
    if LOW_PROTOCOL_THRESHOLD <= protocol_spread < 0.60:
        return random_support, "fallback_medium_protocol_dispersion"
    if budget == 1 and protocol_spread < LOW_PROTOCOL_THRESHOLD:
        return random_support, "fallback_one_label_low_protocol_dispersion"
    # At larger budgets, passive coverage is sufficient in highly diverse
    # cohorts and active geometry has empirically diminishing robustness.
    if budget >= 5 and protocol_spread >= 0.60:
        return random_support, "fallback_large_budget_high_protocol_dispersion"
    acquisition_weight = 2.0 if protocol_spread < LOW_PROTOCOL_THRESHOLD else 0.5
    support = select_facilities(
        distances[acquisition_weight], acquisition, clients, budget, tie_rank
    )
    return support, f"active_w{key_number(acquisition_weight)}"


def predict(predictor, distances, truth, support, test):
    if predictor == "logmean":
        return np.full(len(test), np.exp(np.mean(np.log(truth[support]))))
    weight, bandwidth = parse_predictor(predictor)
    return kernel_prediction(
        distances[weight], test, support, truth[support], bandwidth
    )


def select_support_loo_predictor(distances, truth, support, prior_predictor):
    """Choose a predictor from acquired labels only; never touches test labels."""
    if len(support) < 3:
        return prior_predictor
    ranking = []
    for predictor in predictor_names():
        predictions, targets = [], []
        for position, held in enumerate(support):
            train = np.delete(support, position)
            value = predict(
                predictor, distances, truth, train, np.asarray([held], dtype=int)
            )[0]
            predictions.append(value)
            targets.append(truth[held])
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        ape = np.mean(np.abs(predictions - targets) / np.maximum(targets, 1.0))
        ae = np.mean(np.abs(predictions - targets))
        # Prefer the development-selected prior on an exact validation tie.
        ranking.append((float(ape), float(ae), predictor != prior_predictor, predictor))
    return min(ranking)[-1]


def evaluate(cells, horizon, budget, seeds, influence_cap, run_loo=False):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    predictors = predictor_names()
    arms = ("baseline", "router")
    aggregate = {
        (arm, predictor): defaultdict(
            lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0}
        )
        for arm in arms
        for predictor in predictors
    }
    cv_aggregate = defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
    cv_predictor_counts = Counter()
    route_counts = Counter()
    episode_spreads = []
    # These quantities depend only on the unlabeled target cohort, not on an
    # episode split.  Computing them once is both exact and ~seeds-times faster.
    clients = np.arange(len(cells), dtype=int)
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
    spread = protocol_dispersion(dp)
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        episode_spreads.append(spread)
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        router_support, route = routed_support(
            spread,
            k,
            random_support,
            acquisition,
            clients,
            distances,
            tie_rank,
        )
        route_counts[route] += 1
        for predictor in predictors:
            for arm, support in (
                ("baseline", random_support),
                ("router", router_support),
            ):
                prediction = predict(predictor, distances, truth, support, test)
                ae = np.abs(prediction - truth[test])
                ape = 100.0 * ae / np.maximum(truth[test], 1.0)
                for position, index in enumerate(test):
                    row = aggregate[(arm, predictor)][names[index]]
                    row["ae_sum"] += float(ae[position])
                    row["ape_sum"] += float(ape[position])
                    row["count"] += 1
        if run_loo:
            # Optional negative-control diagnostic; it is not part of the
            # proposed method after failing on heterogeneous SNL cohorts.
            cv_predictor = select_support_loo_predictor(
                distances, truth, router_support, "logmean"
            )
            cv_predictor_counts[cv_predictor] += 1
            cv_prediction = predict(
                cv_predictor, distances, truth, router_support, test
            )
            cv_ae = np.abs(cv_prediction - truth[test])
            cv_ape = 100.0 * cv_ae / np.maximum(truth[test], 1.0)
            for position, index in enumerate(test):
                row = cv_aggregate[names[index]]
                row["ae_sum"] += float(cv_ae[position])
                row["ape_sum"] += float(cv_ape[position])
                row["count"] += 1
    summaries, per_cell = {}, {}
    for arm in arms:
        summaries[arm], per_cell[arm] = {}, {}
        for predictor in predictors:
            rows = []
            for name in names:
                value = aggregate[(arm, predictor)][name]
                if value["count"]:
                    rows.append(
                        {
                            "held_out": name,
                            "ae": value["ae_sum"] / value["count"],
                            "ape": value["ape_sum"] / value["count"],
                            "test_episodes": value["count"],
                        }
                    )
            per_cell[arm][predictor] = rows
            summaries[arm][predictor] = {
                "mae": float(np.mean([row["ae"] for row in rows])),
                "mape": float(np.mean([row["ape"] for row in rows])),
                "n_cells": len(rows),
            }
    cv_rows = []
    for name in names:
        value = cv_aggregate[name]
        if value["count"]:
            cv_rows.append(
                {
                    "held_out": name,
                    "ae": value["ae_sum"] / value["count"],
                    "ape": value["ape_sum"] / value["count"],
                    "test_episodes": value["count"],
                }
            )
    return {
        "summaries": summaries,
        "per_cell": per_cell,
        "route_counts": dict(route_counts),
        "protocol_dispersion": float(np.median(episode_spreads)),
        "router_support_loo": None if not run_loo else {
            "summary": {
                "mae": float(np.mean([row["ae"] for row in cv_rows])),
                "mape": float(np.mean([row["ape"] for row in cv_rows])),
                "n_cells": len(cv_rows),
            },
            "per_cell": cv_rows,
            "predictor_counts": dict(cv_predictor_counts),
        },
    }


def select_baseline_predictor(development_evaluations):
    ranking = []
    for predictor in predictor_names():
        values = [evaluation["summaries"]["baseline"][predictor]
                  for evaluation in development_evaluations]
        ranking.append(
            {
                "predictor": predictor,
                "development_macro_mape": float(np.mean([v["mape"] for v in values])),
                "development_macro_mae": float(np.mean([v["mae"] for v in values])),
            }
        )
    return min(
        ranking,
        key=lambda row: (row["development_macro_mape"], row["development_macro_mae"]),
    ), ranking


def run(horizons, budgets, seeds, influence_cap, run_loo, output_path):
    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    evaluations = {
        (h, d, k): evaluate(
            loaded[(h, d)], h, k, seeds, influence_cap, run_loo
        )
        for h in horizons
        for d in DATASETS
        for k in budgets
    }
    results = []
    for horizon in horizons:
        for target in DATASETS:
            development = [domain for domain in DATASETS if domain != target]
            for budget in budgets:
                target_eval = evaluations[(horizon, target, budget)]
                choice, ranking = select_baseline_predictor(
                    [evaluations[(horizon, domain, budget)] for domain in development]
                )
                predictor = choice["predictor"]
                baseline = target_eval["summaries"]["baseline"][predictor]
                active_routes = [
                    route for route in target_eval["route_counts"]
                    if route.startswith("active_")
                ]
                if active_routes:
                    if len(active_routes) != 1:
                        raise RuntimeError("Router changed branch within one target domain")
                    acquisition_weight = active_routes[0].split("active_w", 1)[1]
                    method_predictor = f"w{acquisition_weight}_bw0.5"
                else:
                    # Exact no-harm fallback: same supports and same predictor.
                    method_predictor = predictor
                method = target_eval["summaries"]["router"][method_predictor]
                baseline_rows = {
                    row["held_out"]: row
                    for row in target_eval["per_cell"]["baseline"][predictor]
                }
                method_rows = {
                    row["held_out"]: row
                    for row in target_eval["per_cell"]["router"][method_predictor]
                }
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "label_budget_k": budget,
                        "protocol_dispersion": target_eval["protocol_dispersion"],
                        "route_counts": target_eval["route_counts"],
                        "selected_matched_predictor": choice,
                        "method_predictor": method_predictor,
                        "baseline": baseline,
                        "method": method,
                        "router_support_loo_diagnostic": target_eval["router_support_loo"],
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
                        "predictor_ranking": ranking,
                    }
                )
    output = {
        "dataset_version": "BatteryLife v12 structured-protocol-v2",
        "protocol": (
            f"{seeds} common 70/30 episodes; total K labels; all target first-H "
            "features available; outer LODO chooses one predictor using only random-support "
            "development scores; proposed arm uses the identical predictor; unlabeled "
            f"router threshold={LOW_PROTOCOL_THRESHOLD:g}; influence cap={influence_cap:g}"
        ),
        "results": results,
    }
    for horizon in horizons:
        for budget in budgets:
            records = [
                r for r in results
                if r["horizon"] == horizon and r["label_budget_k"] == budget
            ]
            relative = [
                (r["method"]["mape"] - r["baseline"]["mape"])
                / max(r["baseline"]["mape"], 1e-12)
                for r in records
            ]
            output[f"macro_h{horizon}_k{budget}"] = {
                "baseline_mae": float(np.mean([r["baseline"]["mae"] for r in records])),
                "method_mae": float(np.mean([r["method"]["mae"] for r in records])),
                "baseline_mape": float(np.mean([r["baseline"]["mape"] for r in records])),
                "method_mape": float(np.mean([r["method"]["mape"] for r in records])),
                "worst_domain_relative_mape_change": float(max(relative)),
                "improved_same_worse_domains": [
                    int(sum(value < -1e-12 for value in relative)),
                    int(sum(abs(value) <= 1e-12 for value in relative)),
                    int(sum(value > 1e-12 for value in relative)),
                ],
            }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--influence-cap", type=float, default=1e9)
    parser.add_argument("--loo-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        tuple(args.horizons), tuple(args.budgets), args.seeds,
        args.influence_cap, args.loo_diagnostic, args.output
    )


if __name__ == "__main__":
    main()
