"""Rank-transfer and scale-calibration under a strict total-K label budget.

The source learner never tries to transfer absolute cycle life.  Instead, it
learns the within-domain normal-score rank from protocol and early-cycle curve
features.  In each target episode, every query's first-H-cycle features may be
observed (the declared transductive design setting), while only K acquisition
cells receive end-of-life labels.  Those labels restore the target-specific
location and scale of log life.  All hyperparameter choices use nested source
domains and exclude the outer target domain.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from batterylife_curve_aware_support import DATASETS, load_cells


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_rank_scale_acquisition.json"
SELECTORS = ("rank_quantile", "rank_uncertainty_medoid")
PRIOR_MULTIPLIERS = (0.5, 1.0)
RIDGE_STRENGTHS = (1.0, 3.0, 10.0)
RANK_BANDWIDTHS = (0.5, 1.0, 2.0)


def raw_features(cells):
    return np.asarray(
        [np.r_[cell["protocol"], cell["curve"]] for cell in cells], dtype=float
    )


def normal_scores(values):
    ranks = rankdata(np.asarray(values), method="average")
    probabilities = (ranks - 0.5) / len(ranks)
    return norm.ppf(np.clip(probabilities, 1e-4, 1.0 - 1e-4))


def fit_rank_source(loaded, horizon, train_domains):
    features, targets, weights, source_slopes = [], [], [], []
    for domain in train_domains:
        cells = loaded[(horizon, domain)]
        log_life = np.log([cell["life"] for cell in cells])
        score = normal_scores(log_life)
        features.extend(raw_features(cells))
        targets.extend(score)
        weights.extend([1.0 / len(cells)] * len(cells))
        centered_score = score - np.mean(score)
        centered_life = log_life - np.mean(log_life)
        denominator = float(np.sum(np.square(centered_score)))
        if denominator > 1e-12:
            source_slopes.append(
                max(0.0, float(np.sum(centered_score * centered_life) / denominator))
            )
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x = imputer.fit_transform(np.asarray(features, dtype=float))
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=20260819,
        n_jobs=-1,
    )
    model.fit(x, np.asarray(targets), sample_weight=np.asarray(weights))
    prior_slope = float(np.median(source_slopes)) if source_slopes else 0.5
    return imputer, model, prior_slope


def rank_predictions(bundle, cells):
    imputer, model, prior_slope = bundle
    x = imputer.transform(raw_features(cells))
    tree_predictions = np.asarray([tree.predict(x) for tree in model.estimators_])
    return (
        np.mean(tree_predictions, axis=0),
        np.std(tree_predictions, axis=0),
        prior_slope,
    )


def robust_columns(matrix, reference):
    center = np.median(matrix[reference], axis=0)
    scale = np.percentile(matrix[reference], 75, axis=0) - np.percentile(
        matrix[reference], 25, axis=0
    )
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return (matrix - center) / scale


def transductive_medoids(matrix, acquisition, clients, k, tie_rank):
    normalized = robust_columns(matrix, clients)
    distance = np.sqrt(
        np.sum(np.square(normalized[:, None, :] - normalized[None, :, :]), axis=2)
    )
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


def quantile_support(score, acquisition, clients, k, tie_rank):
    quantiles = (np.arange(k, dtype=float) + 0.5) / k
    targets = np.quantile(score[clients], quantiles)
    remaining = set(int(index) for index in acquisition)
    selected = []
    for target in targets:
        chosen = min(
            remaining,
            key=lambda index: (abs(float(score[index] - target)), tie_rank[index]),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(selected, dtype=int)


def select_support(selector, score, uncertainty, acquisition, clients, k, rng, tie_rank):
    if selector == "random":
        return np.sort(rng.choice(acquisition, size=k, replace=False))
    if selector == "rank_quantile":
        return quantile_support(score, acquisition, clients, k, tie_rank)
    if selector == "rank_uncertainty_medoid":
        return transductive_medoids(
            np.c_[score, uncertainty], acquisition, clients, k, tie_rank
        )
    raise KeyError(selector)


def calibrate(score, truth, support, test, prior_slope, multiplier, ridge):
    support_x = score[support]
    support_y = np.log(truth[support])
    slope_prior = max(0.0, multiplier * prior_slope)
    if len(support) == 1:
        slope = slope_prior
    else:
        x_center = support_x - np.mean(support_x)
        y_center = support_y - np.mean(support_y)
        slope = float(
            (np.sum(x_center * y_center) + ridge * slope_prior)
            / (np.sum(np.square(x_center)) + ridge)
        )
        slope = float(np.clip(slope, 0.0, max(3.0 * prior_slope, 0.25)))
    intercept = float(np.median(support_y - slope * support_x))
    prediction = np.exp(intercept + slope * score[test])
    return np.clip(prediction, 100.0, 5000.0)


def rank_kernel(score, truth, support, test, bandwidth):
    distance = np.abs(score[test, None] - score[support][None, :])
    weights = np.exp(-0.5 * np.square(distance / bandwidth))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.exp(weights @ np.log(truth[support]))


def candidate_names():
    calibrated = tuple(
        f"{selector}_rankscale_p{int(100 * multiplier)}_r{int(ridge)}"
        for selector in ("random",) + SELECTORS
        for multiplier in PRIOR_MULTIPLIERS
        for ridge in RIDGE_STRENGTHS
    )
    simple = tuple(
        f"{selector}_logmean"
        for selector in ("random",) + SELECTORS
    ) + tuple(
        f"{selector}_rankkernel_bw{int(10 * bandwidth)}"
        for selector in ("random",) + SELECTORS
        for bandwidth in RANK_BANDWIDTHS
    )
    baseline = tuple(name for name in simple if name.startswith("random_")) + tuple(
        name for name in calibrated if name.startswith("random_")
    )
    method = tuple(name for name in simple if not name.startswith("random_")) + tuple(
        name for name in calibrated if not name.startswith("random_")
    )
    return baseline, method


def parse_candidate(name):
    selector, _, prior_text, ridge_text = name.rsplit("_", 3)
    return selector, int(prior_text[1:]) / 100.0, float(ridge_text[1:])


def evaluate(cells, horizon, budget, score, uncertainty, prior_slope, seeds):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    baseline_names, method_names = candidate_names()
    all_names = baseline_names + method_names
    aggregate = {
        candidate: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for candidate in all_names
    }
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(40_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        clients = np.arange(len(cells), dtype=int)
        k = min(budget, len(acquisition))
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        supports = {
            selector: select_support(
                selector, score, uncertainty, acquisition, clients, k, rng, tie_rank
            )
            for selector in ("random",) + SELECTORS
        }
        predictions = {
            "random_logmean": np.full(
                len(test), np.exp(np.mean(np.log(truth[supports["random"]])))
            )
        }
        for candidate in all_names:
            if candidate == "random_logmean":
                continue
            if candidate.endswith("_logmean"):
                selector = candidate[: -len("_logmean")]
                predictions[candidate] = np.full(
                    len(test),
                    np.exp(np.mean(np.log(truth[supports[selector]]))),
                )
                continue
            if "_rankkernel_bw" in candidate:
                selector, bandwidth_text = candidate.split("_rankkernel_bw")
                predictions[candidate] = rank_kernel(
                    score,
                    truth,
                    supports[selector],
                    test,
                    int(bandwidth_text) / 10.0,
                )
                continue
            selector, multiplier, ridge = parse_candidate(candidate)
            predictions[candidate] = calibrate(
                score,
                truth,
                supports[selector],
                test,
                prior_slope,
                multiplier,
                ridge,
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


def choose(development_scores, candidates):
    ranked = []
    for candidate in candidates:
        values = [scores[candidate] for scores in development_scores]
        ranked.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([v["mape"] for v in values])),
                "development_macro_mae": float(np.mean([v["mae"] for v in values])),
                "development_domain_mapes": [float(v["mape"]) for v in values],
            }
        )
    return min(
        ranked,
        key=lambda row: (row["development_macro_mape"], row["development_macro_mae"]),
    ), ranked


def run(horizons, budgets, seeds, output_path):
    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    model_cache, prediction_cache, evaluation_cache = {}, {}, {}

    def predictions(horizon, train_domains, eval_domain):
        model_key = (horizon, tuple(sorted(train_domains)))
        if model_key not in model_cache:
            model_cache[model_key] = fit_rank_source(loaded, horizon, model_key[1])
        key = (model_key, eval_domain)
        if key not in prediction_cache:
            prediction_cache[key] = rank_predictions(
                model_cache[model_key], loaded[(horizon, eval_domain)]
            )
        return prediction_cache[key]

    def evaluation(horizon, budget, train_domains, eval_domain):
        key = (horizon, budget, tuple(sorted(train_domains)), eval_domain)
        if key not in evaluation_cache:
            score, uncertainty, prior = predictions(horizon, train_domains, eval_domain)
            evaluation_cache[key] = evaluate(
                loaded[(horizon, eval_domain)],
                horizon,
                budget,
                score,
                uncertainty,
                prior,
                seeds,
            )
        return evaluation_cache[key]

    results = []
    for horizon in horizons:
        for target in DATASETS:
            development = [domain for domain in DATASETS if domain != target]
            for budget in budgets:
                development_scores = []
                for eval_domain in development:
                    inner_train = [
                        domain for domain in DATASETS if domain not in (target, eval_domain)
                    ]
                    summary, _, _, _ = evaluation(
                        horizon, budget, inner_train, eval_domain
                    )
                    development_scores.append(summary)
                _, _, baseline_names, method_names = evaluation(
                    horizon, budget, development, target
                )
                baseline_choice, baseline_ranking = choose(
                    development_scores, baseline_names
                )
                method_choice, method_ranking = choose(development_scores, method_names)
                target_summary, target_cells, _, _ = evaluation(
                    horizon, budget, development, target
                )
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
            f"{seeds} 70/30 episodes; strict outer target and inner validation exclusion; "
            "source learns only within-domain normal-score life rank; all target first-H "
            "features may guide transductive acquisition but test life labels are never used; "
            "K acquisition labels used once to restore target log-life location/scale"
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(tuple(args.horizons), tuple(args.budgets), args.seeds, args.output)


if __name__ == "__main__":
    main()
