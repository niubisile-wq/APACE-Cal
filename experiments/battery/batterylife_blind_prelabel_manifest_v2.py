"""Label-blind external-domain manifest builder for frozen APACE-Cal v2."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_blind_prelabel_manifest as common
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_asymmetric_cohort_router import protocol_dispersion
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import WEIGHTS, distance_matrix


HERE = Path(__file__).parent
DEV_RESULT = HERE / "batterylife_asymmetric_cohort_router_v2.json"
FREEZE_RECORD = HERE.parents[1] / "METHOD_FREEZE_V2.md"


def setting_manifest(cells, horizon, budget, seeds, baseline_predictor):
    names = [cell["name"] for cell in cells]
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    clients = np.arange(len(cells), dtype=int)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distances = {w: dc if math.isinf(w) else np.sqrt(dp ** 2 + w * dc ** 2)
                 for w in WEIGHTS}
    spread = protocol_dispersion(dp)
    rho = v2.distance_concordance(cells)
    episodes, route_counts = [], defaultdict(int)
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2,
                            max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        method_support, route = v2.routed_support(
            spread, k, random_support, acquisition, clients, distances, tie_rank)
        route_counts[route] += 1
        method_predictor = baseline_predictor
        if route.startswith("active_"):
            if spread >= 0.60:
                method_predictor = ("support_median" if rho < v2.CONCORDANCE_THRESHOLD
                                    else "w0.5_bw0.5")
            else:
                method_predictor = "w2_bw0.5"
        episodes.append({
            "seed": seed,
            "acquisition": [names[i] for i in acquisition],
            "test": [names[i] for i in test],
            "baseline_support": [names[i] for i in random_support],
            "method_support": [names[i] for i in method_support],
            "route": route,
            "baseline_predictor": baseline_predictor,
            "method_predictor": method_predictor,
        })
    return {
        "horizon": horizon, "label_budget_k": budget,
        "n_unlabeled_cells": len(cells), "protocol_dispersion": spread,
        "distance_concordance_spearman": rho,
        "route_counts": dict(route_counts), "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--label-membership-only", action="store_true")
    args = parser.parse_args()

    archive_md5 = common.md5(args.archive)
    if archive_md5 != args.expected_md5:
        raise RuntimeError(f"Archive MD5 mismatch: {archive_md5}")
    dev = json.loads(DEV_RESULT.read_text())
    horizons, budgets = (10, 20, 50), (1, 3, 5, 10)
    allowed = (common.top_level_json_keys(args.label_file)
               if args.label_membership_only else None)
    if args.dataset_dir is None:
        loaded = common.load_unlabeled_archive(args.archive, horizons, allowed)
        feature_source = "verified_zip_stream"
    else:
        loaded = common.load_unlabeled(args.dataset_dir, horizons, allowed)
        feature_source = "extracted_directory"
    settings, predictor_freeze = [], {}
    for horizon in horizons:
        if len(loaded[horizon]) < 3:
            raise RuntimeError(f"Too few unlabeled cells at H={horizon}")
        for budget in budgets:
            choice, ranking = common.external_predictor(dev, horizon, budget)
            predictor_freeze[f"h{horizon}_k{budget}"] = {
                "selected": choice, "ranking": ranking}
            settings.append(setting_manifest(loaded[horizon], horizon, budget,
                                             args.seeds, choice["predictor"]))
    active = [f"h{x['horizon']}_k{x['label_budget_k']}" for x in settings
              if any(route.startswith("active_") for route in x["route_counts"])]
    output = {
        "phase": "PRELABEL_FROZEN_MANIFEST_V2", "dataset": args.dataset,
        "archive": str(args.archive), "archive_size": args.archive.stat().st_size,
        "archive_md5": archive_md5, "feature_source": feature_source,
        "label_file_opaque_sha256": common.sha256(args.label_file),
        "explicit_non_access_statement": (
            "This script lexically read top-level label keys only and skipped every "
            "value without decoding, storing, or summarizing it."
            if args.label_membership_only else
            "This script hashed the label file as opaque bytes and never parsed it."),
        "label_membership_only": args.label_membership_only,
        "label_membership_count": len(allowed) if allowed is not None else None,
        "method_freeze_sha256": common.sha256(FREEZE_RECORD),
        "prelabel_script_sha256": common.sha256(Path(__file__)),
        "evaluator_script_sha256": common.sha256(
            HERE / "batterylife_blind_manifest_eval_v2.py"),
        "common_prelabel_dependency_sha256": common.sha256(
            HERE / "batterylife_blind_prelabel_manifest.py"),
        "method_script_sha256": common.sha256(
            HERE / "batterylife_asymmetric_cohort_router_v2.py"),
        "v1_dependency_sha256": common.sha256(
            HERE / "batterylife_asymmetric_cohort_router.py"),
        "development_result_sha256": common.sha256(DEV_RESULT),
        "router_constants": {"low_protocol_threshold": 0.30,
                             "high_protocol_threshold": 0.60,
                             "concordance_threshold": v2.CONCORDANCE_THRESHOLD,
                             "active_weights": [2.0, 0.5], "rbf_bandwidth": 0.5},
        "predictor_freeze": predictor_freeze,
        "active_settings": active, "settings": settings,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "active_settings": active,
                      "cells_by_horizon": {h: len(loaded[h]) for h in horizons},
                      "unlabeled_diagnostics": {
                          f"h{x['horizon']}": {
                              "protocol_dispersion": x["protocol_dispersion"],
                              "distance_concordance_spearman":
                                  x["distance_concordance_spearman"]}
                          for x in settings if x["label_budget_k"] == 1}}, indent=2))


if __name__ == "__main__":
    main()
