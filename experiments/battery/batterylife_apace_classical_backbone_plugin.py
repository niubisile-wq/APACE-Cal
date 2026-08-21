"""Cross-backbone APACE plugin audit on four local source-trained regressors.

Each held-out target domain is predicted by a model trained only on the other
five development domains.  Random-K and APACE-K share the same test pool,
support budget and calibrator.  All model/calibrator combinations are
reported; no target test labels select a combination.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells


HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_classical_backbone_plugin.json"
MODELS = ("ridge", "rf", "extra", "gbr")
CALIBRATORS = ("raw", "logbias_median", "residual_kernel_w2_bw0.5")


def feature_matrix(cells):
    return np.nan_to_num(
        np.asarray([np.r_[c["protocol"], c["curve"]] for c in cells], dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def make_model(name):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     max_features=0.8, random_state=819, n_jobs=-1)
    if name == "extra":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                   max_features=0.8, random_state=819, n_jobs=-1)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                         learning_rate=0.03, loss="huber", random_state=819)
    raise KeyError(name)


def predict_calibrator(calibrator, query, support_pred, support_truth, distance):
    if calibrator == "raw":
        return float(query)
    residual = np.log(np.maximum(support_truth, 1.0)) - np.log(np.maximum(support_pred, 1.0))
    qlog = math.log(max(float(query), 1.0))
    if calibrator == "logbias_median":
        return float(math.exp(qlog + np.median(residual)))
    weights = np.exp(-0.5 * np.square(distance / 0.5))
    weights /= max(float(weights.sum()), 1e-12)
    return float(math.exp(qlog + weights @ residual))


def main():
    output = {
        "protocol": (
            "six development domains; held-out target source-only model trained on other five; "
            "fixed total K; 100 common episodes; random/APACE supports; all model/calibrator "
            "combinations reported without target-test selection"
        ),
        "results": {},
    }
    for horizon in (10, 20, 50):
        loaded = {d: load_cells(d, horizon) for d in DATASETS}
        for target in DATASETS:
            target_cells = loaded[target]
            target_names = [c["name"] for c in target_cells]
            target_life = np.asarray([c["life"] for c in target_cells], dtype=float)
            protocol = np.asarray([c["protocol"] for c in target_cells], dtype=float)
            curve = np.asarray([c["curve"] for c in target_cells], dtype=float)
            p_scale, c_scale = v1.robust_scale(protocol), v1.robust_scale(curve)
            dp = v1.distance_matrix(protocol, p_scale, 1e9)
            dc = v1.distance_matrix(curve, c_scale, 1e9)
            distances = {w: dc if math.isinf(w) else np.sqrt(dp * dp + w * dc * dc)
                         for w in v1.WEIGHTS}
            spread = v1.protocol_dispersion(dp)
            source_cells = [c for d in DATASETS if d != target for c in loaded[d]]
            source_x, source_y = feature_matrix(source_cells), np.asarray([c["life"] for c in source_cells])
            # Fixed numerical/physical guard: source-only log models cannot
            # extrapolate beyond the largest observed source life.  The cap is
            # determined before any target label is read and is shared by all
            # arms; without it a cross-domain model can overflow exp() and make
            # an otherwise invalid experiment appear to run.
            source_log_cap = float(np.log(max(np.max(source_y), 1.0)))
            model_predictions = {}
            for model_name in MODELS:
                model = make_model(model_name)
                model.fit(source_x, np.log(np.maximum(source_y, 1.0)))
                raw_log = np.asarray(model.predict(feature_matrix(target_cells)), dtype=float)
                raw_log = np.clip(raw_log, 0.0, source_log_cap)
                model_predictions[model_name] = np.exp(raw_log)
            for budget in (1, 3, 5, 10):
                rows = {(m, c, a): defaultdict(list) for m in MODELS for c in CALIBRATORS
                        for a in ("random", "apace")}
                for seed in range(1, 101):
                    rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
                    permutation = rng.permutation(len(target_cells))
                    acquisition_n = min(len(target_cells) - 2,
                                        max(int(math.ceil(0.7 * len(target_cells))), budget))
                    acquisition = np.sort(permutation[:acquisition_n])
                    test = np.sort(permutation[acquisition_n:])
                    k = min(budget, len(acquisition))
                    shuffled = rng.permutation(len(target_cells))
                    tie_rank = np.empty(len(target_cells), dtype=int)
                    tie_rank[shuffled] = np.arange(len(target_cells))
                    random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
                    apace_support, _ = v2.routed_support(
                        spread, k, random_support, np.arange(len(target_cells)),
                        np.arange(len(target_cells)), distances, tie_rank
                    )
                    for arm, support in (("random", random_support), ("apace", apace_support)):
                        support_truth = target_life[support]
                        local_distance = distances[2.0]
                        for model_name in MODELS:
                            pred = model_predictions[model_name]
                            for index in test:
                                for calibrator in CALIBRATORS:
                                    estimate = predict_calibrator(
                                        calibrator,
                                        pred[index],
                                        pred[support],
                                        support_truth,
                                        local_distance[index, support],
                                    )
                                    ae = abs(estimate - target_life[index])
                                    rows[(model_name, calibrator, arm)][target_names[index]].append(
                                        100.0 * ae / max(target_life[index], 1.0)
                                    )
                for model_name in MODELS:
                    for calibrator in CALIBRATORS:
                        for arm in ("random", "apace"):
                            cell_values = rows[(model_name, calibrator, arm)]
                            output["results"].setdefault(
                                f"{target}_h{horizon}_k{budget}",
                                {"protocol_dispersion": spread, "rows": []},
                            )["rows"].append(
                                {
                                    "model": model_name,
                                    "calibrator": calibrator,
                                    "arm": arm,
                                    "mape": float(np.mean([np.mean(v) for v in cell_values.values()])),
                                    "n_cells": len(cell_values),
                                }
                            )
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for key, value in output["results"].items():
        if key.endswith("_k3") and key.startswith(("CALB_", "MICH_EXP_", "SNL_")):
            print(key)
            for row in value["rows"]:
                if row["calibrator"] in ("raw", "residual_kernel_w2_bw0.5"):
                    print(row)


if __name__ == "__main__":
    main()
