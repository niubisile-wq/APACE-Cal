"""Development-only APACE candidate with an unlabeled route-stability gate.

The frozen v2 source/results are not modified.  For each target cohort the
candidate recomputes the v2 route under 100 deterministic 1% perturbations of
the unlabeled protocol/curve features.  An active branch is permitted only if
its route changes in at most ``MAX_FLIP_RATE`` of perturbations; otherwise the
exact matched random-support fallback is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_asymmetric_cohort_router as v1
from batterylife_curve_aware_support import DATASETS, load_cells
from batterylife_transductive_pool_acquisition import distance_matrix
from batterylife_curve_aware_support import robust_scale


HERE = Path(__file__).parent
MAX_FLIP_RATE = 0.05
# A 10% random metadata loss should remain eligible, whereas the 25% stress
# condition should be treated as insufficient evidence and fall back.
MIN_PROTOCOL_COVERAGE = 0.90


def dispersion(cells):
    p = np.asarray([c["protocol"] for c in cells], dtype=float)
    dp = distance_matrix(p, robust_scale(p), 1e9)
    upper = dp[np.triu_indices(len(cells), 1)]
    return float(np.median(upper)) if len(upper) else 0.0


def route_name(spread, budget):
    if spread <= 1e-12:
        return "fallback_zero_protocol_dispersion"
    if v1.LOW_PROTOCOL_THRESHOLD <= spread < 0.60:
        return "fallback_medium_protocol_dispersion"
    if budget == 1:
        return "fallback_one_label_unidentifiable"
    if budget >= 5 and spread >= 0.60:
        return "fallback_large_budget_high_protocol_dispersion"
    weight = 2.0 if spread < v1.LOW_PROTOCOL_THRESHOLD else 0.5
    return f"active_w{v1.key_number(weight)}"


def perturb(cells, token, level=0.01):
    out = []
    protocol_matrix = np.asarray([c["protocol"] for c in cells], dtype=float)
    protocol_scale = robust_scale(protocol_matrix)
    protocol_variable = np.asarray([
        np.isfinite(column).sum() >= 2
        and float(np.nanmax(column) - np.nanmin(column)) > 1e-12
        for column in protocol_matrix.T
    ], dtype=bool)
    for cell in cells:
        digest = hashlib.sha256(f"{token}|{cell['name']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        clone = dict(cell)
        p = np.asarray(cell["protocol"], dtype=float).copy()
        q = np.asarray(cell["curve"], dtype=float).copy()
        p_mask = np.isfinite(p) & protocol_variable
        q_mask = np.isfinite(q)
        # Do not inject continuous noise into constant/categorical metadata.
        # Noise is one percent of each observed dimension's robust scale.
        for column in range(p.shape[0]):
            if p_mask[column]:
                p[column] += rng.normal(
                    0.0, level * max(float(protocol_scale[column]), 1e-12)
                )
        q[q_mask] *= 1.0 + rng.normal(0.0, level, int(q_mask.sum()))
        clone["protocol"], clone["curve"] = p, q
        out.append(clone)
    return out


def stability(cells, horizon, domain, budgets, repetitions=100):
    spread = dispersion(cells)
    matrix = np.asarray([c["protocol"] for c in cells], dtype=float)
    expected_columns = np.isfinite(matrix).mean(axis=0) >= 0.50
    if expected_columns.any():
        coverage = float(np.isfinite(matrix[:, expected_columns]).mean())
    else:
        coverage = 0.0
    base = {k: route_name(spread, k) for k in budgets}
    flips = {k: 0 for k in budgets}
    for seed in range(repetitions):
        noisy = perturb(cells, f"{domain}|{horizon}|{seed}")
        s = dispersion(noisy)
        for k in budgets:
            flips[k] += int(route_name(s, k) != base[k])
    rows = {}
    for k in budgets:
        rate = flips[k] / float(repetitions)
        rows[k] = {
            "base_route": base[k],
            "flip_rate": rate,
            "protocol_coverage": coverage,
            "active_allowed": (
                base[k].startswith("active_")
                and rate <= MAX_FLIP_RATE
                and coverage >= MIN_PROTOCOL_COVERAGE
            ),
        }
    return spread, rows


def run(horizons, budgets, seeds, output_path):
    # Use the frozen v2 predictor/routing implementation only as a component;
    # this script adds a precomputed unlabeled abstention decision.
    v1.predictor_names = v2.predictor_names
    v1.predict = v2.predict
    loaded = {(h, d): load_cells(d, h) for h in horizons for d in DATASETS}
    concordance = {
        (h, d): v2.distance_concordance(loaded[(h, d)])
        for h in horizons for d in DATASETS
    }
    stability_rows = {}
    for h in horizons:
        for d in DATASETS:
            spread, rows = stability(loaded[(h, d)], h, d, budgets)
            stability_rows[(h, d)] = {"protocol_dispersion": spread, "budgets": rows}

    evaluations = {}
    for h in horizons:
        for d in DATASETS:
            for k in budgets:
                allow = stability_rows[(h, d)]["budgets"][k]["active_allowed"]

                def gated_router(protocol_spread, budget, random_support, acquisition,
                                 clients, distances, tie_rank, _allow=allow):
                    if not _allow:
                        return random_support, "fallback_stability_gate"
                    return v2.routed_support(
                        protocol_spread, budget, random_support, acquisition,
                        clients, distances, tie_rank
                    )

                v1.routed_support = gated_router
                evaluations[(h, d, k)] = v1.evaluate(
                    loaded[(h, d)], h, k, seeds, 1e9, False
                )

    results = []
    for h in horizons:
        for target in DATASETS:
            development = [d for d in DATASETS if d != target]
            for k in budgets:
                target_eval = evaluations[(h, target, k)]
                choice, ranking = v2.choose_baseline(
                    [evaluations[(h, d, k)] for d in development]
                )
                baseline_predictor = choice["predictor"]
                active_routes = [
                    route for route in target_eval["route_counts"]
                    if route.startswith("active_")
                ]
                if active_routes:
                    if len(active_routes) != 1:
                        raise RuntimeError("more than one active route in a cohort")
                    method_predictor = (
                        "support_median" if (
                            target_eval["protocol_dispersion"] >= 0.60
                            and concordance[(h, target)] < v2.CONCORDANCE_THRESHOLD
                        ) else (
                            "w0.5_bw0.5" if target_eval["protocol_dispersion"] >= 0.60
                            else "w2_bw0.5"
                        )
                    )
                else:
                    method_predictor = baseline_predictor
                baseline = target_eval["summaries"]["baseline"][baseline_predictor]
                method = target_eval["summaries"]["router"][method_predictor]
                rel = (method["mape"] - baseline["mape"]) / max(baseline["mape"], 1e-12)
                results.append({
                    "horizon": h, "target": target, "label_budget_k": k,
                    "baseline": baseline, "method": method,
                    "relative_change": rel,
                    "route_counts": target_eval["route_counts"],
                    "stability": stability_rows[(h, target)]["budgets"][k],
                    "selected_matched_predictor": choice,
                    "method_predictor": method_predictor,
                    "predictor_ranking": ranking,
                })

    output = {
        "status": "DEVELOPMENT CANDIDATE; v2 UNCHANGED",
        "protocol": f"100 common episodes; 100 unlabeled 1% perturbations; max flip rate {MAX_FLIP_RATE}",
        "results": results,
        "stability": [
            {"horizon": h, "dataset": d, **stability_rows[(h, d)]}
            for h in horizons for d in DATASETS
        ],
    }
    for h in horizons:
        for k in budgets:
            rows = [r for r in results if r["horizon"] == h and r["label_budget_k"] == k]
            rel = np.asarray([r["relative_change"] for r in rows])
            output[f"macro_h{h}_k{k}"] = {
                "baseline_mape": float(np.mean([r["baseline"]["mape"] for r in rows])),
                "method_mape": float(np.mean([r["method"]["mape"] for r in rows])),
                "relative_reduction_percent": float(-100.0 * np.mean(rel)),
                "improved_same_worse_domains": [
                    int(np.sum(rel < -1e-12)), int(np.sum(np.abs(rel) <= 1e-12)),
                    int(np.sum(rel > 1e-12))
                ],
                "max_domain_relative_degradation_percent": float(100.0 * np.max(rel)),
            }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=HERE / "batterylife_apace_stability_gate_candidate.json")
    args = parser.parse_args()
    run(args.horizons, args.budgets, args.seeds, args.output)


if __name__ == "__main__":
    main()
