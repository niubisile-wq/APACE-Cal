"""Post-MATR development search for high-dispersion K=3 acquisition.

Tests a small, predeclared 3x3 factorial: keep 0/1/2 random anchors from the
matched random K=3 draw, fill the remaining slots by conditional facility
coverage, then predict by the fixed w=0.5 kernel, support log mean, or support
median.  The search is restricted to the three seen high-dispersion domains
MICH_EXP, SNL, and MATR and is not blind evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from batterylife_asymmetric_cohort_router import predict
from batterylife_blind_prelabel_manifest import (
    load_unlabeled_archive,
    top_level_json_keys,
)
from batterylife_curve_aware_support import load_cells, robust_scale
from batterylife_transductive_pool_acquisition import (
    WEIGHTS,
    distance_matrix,
    kernel_prediction,
)


HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "batterylife_high_dispersion_hybrid_search.json"
DOMAINS = ("MICH_EXP", "SNL", "MATR")
ANCHORS = (0, 1, 2)
PREDICTORS = (
    "kernel",
    "logmean",
    "median",
    "blend0.25",
    "blend0.5",
    "blend0.75",
)


def load_matr_all(horizons):
    archive = ROOT / "data" / "batterylife_zenodo" / "MATR.zip"
    label_file = ROOT / "data" / "batterylife_processed" / "Life labels" / "MATR_labels.json"
    allowed = top_level_json_keys(label_file)
    unlabeled = load_unlabeled_archive(archive, horizons, allowed)
    labels = json.load(open(label_file))
    return {
        horizon: [dict(cell, life=float(labels[cell["name"]])) for cell in cells]
        for horizon, cells in unlabeled.items()
    }


def conditional_facilities(distance, acquisition, clients, initial, k, tie_rank):
    selected = [int(index) for index in initial]
    if selected:
        current = np.min(distance[np.ix_(clients, selected)], axis=1)
    else:
        current = np.full(len(clients), np.inf)
    while len(selected) < min(k, len(acquisition)):
        candidates = []
        for candidate in acquisition:
            if candidate in selected:
                continue
            cost = float(np.sum(np.minimum(current, distance[clients, candidate])))
            candidates.append((cost, int(tie_rank[candidate]), int(candidate)))
        _, _, chosen = min(candidates)
        selected.append(chosen)
        current = np.minimum(current, distance[clients, chosen])
    return np.asarray(selected, dtype=int)


def prediction(kind, distance, truth, support, test):
    if kind == "kernel":
        return kernel_prediction(distance, test, support, truth[support], 0.5)
    if kind == "logmean":
        return np.full(len(test), np.exp(np.mean(np.log(truth[support]))))
    if kind == "median":
        return np.full(len(test), np.median(truth[support]))
    if kind.startswith("blend"):
        kernel_weight = float(kind.split("blend", 1)[1])
        kernel = kernel_prediction(distance, test, support, truth[support], 0.5)
        median = np.full(len(test), np.median(truth[support]))
        return np.exp(
            kernel_weight * np.log(kernel)
            + (1.0 - kernel_weight) * np.log(median)
        )
    raise ValueError(kind)


def evaluate(cells, horizon, seeds):
    names = [cell["name"] for cell in cells]
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    clients = np.arange(len(cells), dtype=int)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distance = np.sqrt(np.square(dp) + 0.5 * np.square(dc))
    stores = {
        (anchors, predictor): defaultdict(
            lambda: {"ape_sum": 0.0, "ae_sum": 0.0, "count": 0}
        )
        for anchors in ANCHORS
        for predictor in PREDICTORS
    }
    random_stores = {
        predictor: defaultdict(
            lambda: {"ape_sum": 0.0, "ae_sum": 0.0, "count": 0}
        )
        for predictor in PREDICTORS
    }
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 30_000 + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), 3))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_draw = rng.choice(acquisition, size=3, replace=False)
        random_support = np.sort(random_draw)
        for predictor in PREDICTORS:
            value = prediction(predictor, distance, truth, random_support, test)
            ae = np.abs(value - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for position, held in enumerate(test):
                row = random_stores[predictor][names[held]]
                row["ape_sum"] += float(ape[position])
                row["ae_sum"] += float(ae[position])
                row["count"] += 1
        for anchors in ANCHORS:
            initial = random_draw[:anchors]
            support = conditional_facilities(
                distance, acquisition, clients, initial, 3, tie_rank
            )
            for predictor in PREDICTORS:
                value = prediction(predictor, distance, truth, support, test)
                ae = np.abs(value - truth[test])
                ape = 100.0 * ae / np.maximum(truth[test], 1.0)
                for position, held in enumerate(test):
                    row = stores[(anchors, predictor)][names[held]]
                    row["ape_sum"] += float(ape[position])
                    row["ae_sum"] += float(ae[position])
                    row["count"] += 1

    def summarize(store):
        rows = []
        for name in names:
            value = store[name]
            if value["count"]:
                rows.append(
                    {
                        "held_out": name,
                        "ape": value["ape_sum"] / value["count"],
                        "ae": value["ae_sum"] / value["count"],
                    }
                )
        return {
            "mape": float(np.mean([row["ape"] for row in rows])),
            "mae": float(np.mean([row["ae"] for row in rows])),
            "per_cell": rows,
        }

    return {
        "random": {predictor: summarize(random_stores[predictor]) for predictor in PREDICTORS},
        "hybrid": {
            f"a{anchors}_{predictor}": summarize(stores[(anchors, predictor)])
            for anchors in ANCHORS
            for predictor in PREDICTORS
        },
    }


def run(horizons, seeds, output_path):
    matr = load_matr_all(horizons)
    loaded = {
        (domain, horizon): (
            matr[horizon] if domain == "MATR" else load_cells(domain, horizon)
        )
        for domain in DOMAINS
        for horizon in horizons
    }
    results = []
    for domain in DOMAINS:
        for horizon in horizons:
            result = evaluate(loaded[(domain, horizon)], horizon, seeds)
            results.append({"domain": domain, "horizon": horizon, **result})
    candidate_summary = {}
    for anchors in ANCHORS:
        for predictor in PREDICTORS:
            key = f"a{anchors}_{predictor}"
            changes = []
            for record in results:
                baseline = record["random"][predictor]["mape"]
                method = record["hybrid"][key]["mape"]
                changes.append((method - baseline) / max(baseline, 1e-12))
            candidate_summary[key] = {
                "mean_relative_change": float(np.mean(changes)),
                "worst_relative_change": float(np.max(changes)),
                "improved_same_worse_domain_horizons": [
                    int(sum(x < -1e-12 for x in changes)),
                    int(sum(abs(x) <= 1e-12 for x in changes)),
                    int(sum(x > 1e-12 for x in changes)),
                ],
            }
    v1_result = json.load(open(HERE / "batterylife_asymmetric_cohort_router.json"))
    matr_result = json.load(open(HERE / "batterylife_matr_blind_eval.json"))
    strong_baseline = {}
    for record in v1_result["results"]:
        if record["target"] in ("MICH_EXP", "SNL") and record["label_budget_k"] == 3:
            strong_baseline[(record["target"], record["horizon"])] = record["baseline"]["mape"]
    for record in matr_result["results"]:
        if record["label_budget_k"] == 3:
            strong_baseline[("MATR", record["horizon"])] = record["baseline"]["mape"]
    versus_strong_baseline = {}
    for predictor in PREDICTORS:
        key = f"a0_{predictor}"
        changes = []
        details = []
        for record in results:
            baseline = strong_baseline[(record["domain"], record["horizon"])]
            method = record["hybrid"][key]["mape"]
            change = (method - baseline) / max(baseline, 1e-12)
            changes.append(change)
            details.append(
                {
                    "domain": record["domain"],
                    "horizon": record["horizon"],
                    "strong_baseline_mape": baseline,
                    "method_mape": method,
                    "relative_change": change,
                }
            )
        versus_strong_baseline[key] = {
            "mean_relative_change": float(np.mean(changes)),
            "worst_relative_change": float(np.max(changes)),
            "improved_same_worse_domain_horizons": [
                int(sum(x < -1e-12 for x in changes)),
                int(sum(abs(x) <= 1e-12 for x in changes)),
                int(sum(x > 1e-12 for x in changes)),
            ],
            "details": details,
        }
    output = {
        "status": "POST-BLIND DEVELOPMENT SEARCH; NOT CONFIRMATORY",
        "protocol": f"{seeds} matched episodes; high-dispersion K3 only",
        "results": results,
        "candidate_summary": candidate_summary,
        "full_facility_versus_existing_strong_baseline": versus_strong_baseline,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(versus_strong_baseline, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(tuple(args.horizons), args.seeds, args.output)


if __name__ == "__main__":
    main()
