"""APACE-Cal v2 development candidate after the frozen v1 MATR failure.

The scientific problem and total-K protocol stay unchanged.  This candidate
tests two mechanism-level changes motivated by the audited failure:

1. K=1 always falls back to passive random acquisition because an unlabeled
   geometric centre is not identifiable as a life-representative cell.
2. In a high-protocol-dispersion cohort (D_p >= 0.60), the only active budget
   is K=3.  Unlabeled protocol/curve distance concordance routes prediction:
   Spearman rho < 0.35 uses the support-life median, otherwise the original
   evidence-coupled kernel.  Low-dispersion K>=3 keeps the original branch.

MATR is now a seen failure-development domain and cannot be reused as blind
confirmation.  This script first evaluates the candidate on the original six
development domains with the original episode seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import batterylife_asymmetric_cohort_router as v1
from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_asymmetric_cohort_router_v2.json"
BASE_PREDICTORS = v1.predictor_names()
BASE_PREDICT = v1.predict
CONCORDANCE_THRESHOLD = 0.35


def predictor_names():
    return BASE_PREDICTORS + (
        "support_median",
        "binary_loo_w0.5",
        "binary_loo_w2",
    )


def predict(predictor, distances, truth, support, test):
    if predictor == "support_median":
        return np.full(len(test), float(np.median(truth[support])))
    if predictor.startswith("binary_loo_w"):
        weight = float(predictor.split("binary_loo_w", 1)[1])
        kernel_predictor = f"w{v1.key_number(weight)}_bw0.5"
        if len(support) < 3:
            return BASE_PREDICT(kernel_predictor, distances, truth, support, test)
        scores = {}
        for candidate in (kernel_predictor, "support_median"):
            errors = []
            for position, held in enumerate(support):
                train = np.delete(support, position)
                if candidate == "support_median":
                    value = float(np.median(truth[train]))
                else:
                    value = BASE_PREDICT(
                        candidate,
                        distances,
                        truth,
                        train,
                        np.asarray([held], dtype=int),
                    )[0]
                errors.append(abs(value - truth[held]) / max(truth[held], 1.0))
            scores[candidate] = float(np.mean(errors))
        # Exact ties preserve the original evidence-coupled kernel prior.
        selected = min(
            (kernel_predictor, "support_median"),
            key=lambda candidate: (
                scores[candidate], candidate == "support_median"
            ),
        )
        if selected == "support_median":
            return np.full(len(test), float(np.median(truth[support])))
        return BASE_PREDICT(selected, distances, truth, support, test)
    return BASE_PREDICT(predictor, distances, truth, support, test)


def routed_support(protocol_spread, budget, random_support, acquisition, clients,
                   distances, tie_rank):
    if protocol_spread <= 1e-12:
        return random_support, "fallback_zero_protocol_dispersion"
    if v1.LOW_PROTOCOL_THRESHOLD <= protocol_spread < 0.60:
        return random_support, "fallback_medium_protocol_dispersion"
    # v2 safety correction: a single unlabeled geometric centre can be a life
    # extreme, as observed prospectively on MATR H20.
    if budget == 1:
        return random_support, "fallback_one_label_unidentifiable"
    if budget >= 5 and protocol_spread >= 0.60:
        return random_support, "fallback_large_budget_high_protocol_dispersion"
    acquisition_weight = 2.0 if protocol_spread < v1.LOW_PROTOCOL_THRESHOLD else 0.5
    support = v1.select_facilities(
        distances[acquisition_weight], acquisition, clients, budget, tie_rank
    )
    return support, f"active_w{v1.key_number(acquisition_weight)}"


def choose_baseline(development_evaluations):
    ranking = []
    for predictor in BASE_PREDICTORS + ("support_median",):
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


def distance_concordance(cells):
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    upper = np.triu_indices(len(cells), k=1)
    if len(upper[0]) < 3:
        return 0.0
    # scipy warns for a constant distance vector; here that condition has the
    # intended interpretation of no usable unlabeled concordance evidence.
    if np.ptp(dp[upper]) <= 1e-12 or np.ptp(dc[upper]) <= 1e-12:
        return 0.0
    value = spearmanr(dp[upper], dc[upper]).statistic
    return float(value) if np.isfinite(value) else 0.0


def run(horizons, budgets, seeds, output_path):
    # Patch only this process's imported v1 module.  The frozen v1 source file
    # and its hash remain unchanged.
    v1.predictor_names = predictor_names
    v1.predict = predict
    v1.routed_support = routed_support

    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    concordance = {
        (h, d): distance_concordance(loaded[(h, d)])
        for h in horizons
        for d in DATASETS
    }
    evaluations = {
        (h, d, k): v1.evaluate(loaded[(h, d)], h, k, seeds, 1e9, False)
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
                choice, ranking = choose_baseline(
                    [evaluations[(horizon, domain, budget)] for domain in development]
                )
                baseline_predictor = choice["predictor"]
                active = any(
                    route.startswith("active_")
                    for route in target_eval["route_counts"]
                )
                spread = target_eval["protocol_dispersion"]
                rho = concordance[(horizon, target)]
                if not active:
                    method_predictor = baseline_predictor
                elif spread >= 0.60:
                    method_predictor = (
                        "support_median"
                        if rho < CONCORDANCE_THRESHOLD
                        else "w0.5_bw0.5"
                    )
                else:
                    method_predictor = "w2_bw0.5"
                baseline = target_eval["summaries"]["baseline"][baseline_predictor]
                method = target_eval["summaries"]["router"][method_predictor]
                baseline_rows = {
                    row["held_out"]: row
                    for row in target_eval["per_cell"]["baseline"][baseline_predictor]
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
                        "protocol_dispersion": spread,
                        "distance_concordance_spearman": rho,
                        "route_counts": target_eval["route_counts"],
                        "selected_matched_predictor": choice,
                        "method_predictor": method_predictor,
                        "baseline": baseline,
                        "method": method,
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
        "status": "POST-MATR DEVELOPMENT CANDIDATE; NOT BLIND CONFIRMED",
        "protocol": (
            f"{seeds} common 70/30 episodes; total K labels; original six-domain "
            "LODO baseline selection; K1 always passive; high-dispersion K3 uses "
            f"facility supports and rho<{CONCORDANCE_THRESHOLD:g} routes to support median "
            "otherwise w0.5/bw0.5 kernel; low-dispersion K>=3 keeps "
            "evidence-coupled w2/bw0.5 kernel; medium and high K>=5 fallback"
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
