"""Strict total-K calibration using source-guided identifiability acquisition.

Unlike geometric k-median selection, this experiment trains an early-curve
source ensemble on allowed development domains.  Its target pseudolife rank
and ensemble disagreement guide which acquisition-pool cells receive full-life
labels.  Outer target labels never train the source model or choose the rule;
inner development models exclude both the outer target and evaluated domain.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from batterylife_curve_aware_support import DATASETS, load_cells
from batterylife_fixed_pool_acquisition import facility_medoids


OUTPUT = Path(__file__).with_suffix(".json")
HORIZONS = (10, 20, 50)
BUDGETS = (1, 3, 5, 10)
SEEDS = 100
SELECTORS = ("random", "pred_quantile", "uncertainty_top", "risk_medoid")
CALIBRATORS = ("bias", "affine")
BLENDS = (0.25, 0.5, 0.75, 1.0)


def raw_features(cells):
    return np.asarray([np.r_[cell["protocol"], cell["curve"]] for cell in cells], dtype=float)


def fit_source(loaded, horizon, train_domains):
    features, labels, weights = [], [], []
    for domain in train_domains:
        cells = loaded[(horizon, domain)]
        features.extend(raw_features(cells))
        labels.extend(np.log([cell["life"] for cell in cells]))
        weights.extend([1.0 / len(cells)] * len(cells))
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x = imputer.fit_transform(np.asarray(features))
    model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=20260819,
        n_jobs=-1,
    )
    model.fit(x, np.asarray(labels), sample_weight=np.asarray(weights))
    return imputer, model


def source_predictions(model_bundle, cells):
    imputer, model = model_bundle
    x = imputer.transform(raw_features(cells))
    tree_predictions = np.asarray([tree.predict(x) for tree in model.estimators_])
    return np.mean(tree_predictions, axis=0), np.std(tree_predictions, axis=0)


def normalized_risk_space(prediction, uncertainty, acquisition):
    matrix = np.c_[prediction, uncertainty]
    center = np.median(matrix[acquisition], axis=0)
    scale = np.percentile(matrix[acquisition], 75, axis=0) - np.percentile(
        matrix[acquisition], 25, axis=0
    )
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return (matrix - center) / scale


def select_support(selector, acquisition, k, prediction, uncertainty, rng, tie_rank):
    if selector == "random":
        return np.sort(rng.choice(acquisition, size=k, replace=False))
    if selector == "pred_quantile":
        ordered = sorted(acquisition, key=lambda i: (prediction[i], tie_rank[i]))
        positions = np.rint(np.linspace(0, len(ordered) - 1, k)).astype(int)
        return np.asarray([ordered[position] for position in positions], dtype=int)
    if selector == "uncertainty_top":
        ordered = sorted(acquisition, key=lambda i: (-uncertainty[i], tie_rank[i]))
        return np.asarray(ordered[:k], dtype=int)
    if selector == "risk_medoid":
        risk = normalized_risk_space(prediction, uncertainty, acquisition)
        distance = np.sqrt(np.sum(np.square(risk[:, None, :] - risk[None, :, :]), axis=2))
        return facility_medoids(distance, acquisition, k, tie_rank)
    raise KeyError(selector)


def calibrated_prediction(calibrator, source_log, truth, support, test):
    support_x = source_log[support]
    support_y = np.log(truth[support])
    if calibrator == "bias" or len(support) == 1:
        output = source_log[test] + float(np.median(support_y - support_x))
    elif calibrator == "affine":
        model = Ridge(alpha=10.0).fit(support_x.reshape(-1, 1), support_y)
        output = model.predict(source_log[test].reshape(-1, 1))
    else:
        raise KeyError(calibrator)
    # Fixed dataset eligibility/plausibility bounds, independent of target
    # labels and wide enough not to manufacture narrow support interpolation.
    return np.exp(np.clip(output, np.log(100.0), np.log(5000.0)))


def candidate_names():
    baseline = ("random_logmean", "random_bias", "random_affine")
    method = baseline + tuple(f"{selector}_logmean" for selector in SELECTORS if selector != "random")
    method += tuple(
        f"{selector}_{calibrator}_b{int(round(100 * blend))}"
        for selector in SELECTORS
        if selector != "random"
        for calibrator in CALIBRATORS
        for blend in BLENDS
    )
    return baseline, method


def evaluate(cells, horizon, budget, source_log, uncertainty):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    baseline_names, all_names = candidate_names()
    aggregate = {
        candidate: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for candidate in all_names
    }
    effective_sizes = []
    for seed in range(1, SEEDS + 1):
        rng = np.random.default_rng(30_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        effective_sizes.append(k)
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        supports = {
            selector: select_support(
                selector, acquisition, k, source_log, uncertainty, rng, tie_rank
            )
            for selector in SELECTORS
        }
        predictions = {
            "random_logmean": np.full(
                len(test), np.exp(np.mean(np.log(truth[supports["random"]])))
            )
        }
        for calibrator in CALIBRATORS:
            predictions[f"random_{calibrator}"] = calibrated_prediction(
                calibrator, source_log, truth, supports["random"], test
            )
        for selector in SELECTORS:
            if selector == "random":
                continue
            conservative_log = float(np.mean(np.log(truth[supports[selector]])))
            predictions[f"{selector}_logmean"] = np.full(
                len(test), np.exp(conservative_log)
            )
            for calibrator in CALIBRATORS:
                calibrated_log = np.log(
                    calibrated_prediction(
                        calibrator, source_log, truth, supports[selector], test
                    )
                )
                for blend in BLENDS:
                    key = f"{selector}_{calibrator}_b{int(round(100 * blend))}"
                    predictions[key] = np.exp(
                        (1.0 - blend) * conservative_log + blend * calibrated_log
                    )
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
    return summaries, per_cell, baseline_names, all_names, int(np.median(effective_sizes))


def choose(development_scores, candidates):
    ranked = []
    for candidate in candidates:
        values = [scores[candidate] for scores in development_scores]
        ranked.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([value["mape"] for value in values])),
                "development_macro_mae": float(np.mean([value["mae"] for value in values])),
            }
        )
    return min(ranked, key=lambda row: (row["development_macro_mape"], row["development_macro_mae"])), ranked


def choose_with_directional_gate(development_scores, candidates, baseline_candidate):
    """Enable transfer only when no development domain loses on MAPE."""
    ranked = []
    for candidate in candidates:
        values = [scores[candidate] for scores in development_scores]
        baselines = [scores[baseline_candidate] for scores in development_scores]
        relative = [
            (value["mape"] - baseline["mape"]) / max(baseline["mape"], 1e-12)
            for value, baseline in zip(values, baselines)
        ]
        ranked.append(
            {
                "candidate": candidate,
                "development_macro_mape": float(np.mean([value["mape"] for value in values])),
                "development_macro_mae": float(np.mean([value["mae"] for value in values])),
                "worst_development_mape_degradation_fraction": float(max(relative)),
                "passes_directional_gate": bool(max(relative) <= 1e-12),
            }
        )
    eligible = [row for row in ranked if row["passes_directional_gate"]]
    if not eligible:
        raise RuntimeError("Baseline candidate must always pass its own directional gate")
    return min(
        eligible,
        key=lambda row: (row["development_macro_mape"], row["development_macro_mae"]),
    ), ranked


def main():
    loaded = {(h, d): load_cells(d, h) for h in HORIZONS for d in DATASETS}
    model_cache, prediction_cache = {}, {}

    def predictions(horizon, train_domains, eval_domain):
        model_key = (horizon, tuple(sorted(train_domains)))
        if model_key not in model_cache:
            model_cache[model_key] = fit_source(loaded, horizon, model_key[1])
        key = (model_key, eval_domain)
        if key not in prediction_cache:
            prediction_cache[key] = source_predictions(
                model_cache[model_key], loaded[(horizon, eval_domain)]
            )
        return prediction_cache[key]

    results = []
    for horizon in HORIZONS:
        for target in DATASETS:
            development = [domain for domain in DATASETS if domain != target]
            final_train = development
            final_prediction, final_uncertainty = predictions(horizon, final_train, target)
            for budget in BUDGETS:
                development_evaluations = []
                baseline_names = all_names = None
                for eval_domain in development:
                    inner_train = [
                        domain for domain in DATASETS if domain not in (target, eval_domain)
                    ]
                    pred, unc = predictions(horizon, inner_train, eval_domain)
                    summary, _, baseline_names, all_names, _ = evaluate(
                        loaded[(horizon, eval_domain)], horizon, budget, pred, unc
                    )
                    development_evaluations.append(summary)
                baseline_choice, baseline_ranking = choose(
                    development_evaluations, baseline_names
                )
                method_choice, method_ranking = choose_with_directional_gate(
                    development_evaluations,
                    all_names,
                    baseline_choice["candidate"],
                )
                target_summary, target_cells, _, _, effective_k = evaluate(
                    loaded[(horizon, target)],
                    horizon,
                    budget,
                    final_prediction,
                    final_uncertainty,
                )
                baseline, method = baseline_choice["candidate"], method_choice["candidate"]
                baseline_rows = {row["held_out"]: row for row in target_cells[baseline]}
                method_rows = {row["held_out"]: row for row in target_cells[method]}
                results.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "label_budget_k": budget,
                        "effective_k": effective_k,
                        "selected_baseline": baseline_choice,
                        "selected_method": method_choice,
                        "baseline": target_summary[baseline],
                        "method": target_summary[method],
                        "method_is_source_guided": not method.startswith("random_"),
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
            "strict outer target exclusion; inner source model excludes outer target and "
            "evaluated development domain; 70/30 acquisition/test episodes; K labels once; "
            "source ExtraTrees pseudolife/disagreement selection; transfer enabled only if "
            "no development domain worsens in MAPE versus the selected random baseline; "
            "cell then domain aggregation"
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
                "source_guided_selected_domains": int(
                    sum(record["method_is_source_guided"] for record in records)
                ),
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key.startswith("macro_")}, indent=2))


if __name__ == "__main__":
    main()
