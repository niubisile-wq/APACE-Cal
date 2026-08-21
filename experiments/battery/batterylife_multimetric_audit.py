"""Re-run frozen APACE-Cal main windows while retaining predictions.

The original frozen JSON intentionally stores absolute error and APE only. This
separate audit reproduces the same H/K/seed protocol and stores enough summary
statistics for MAE, RMSE, MAPE and sMAPE. It never overwrites frozen artifacts.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


HERE = Path(__file__).parent
OUT = HERE / "batterylife_multimetric_audit.json"
HORIZONS = (10, 20, 50)
BUDGETS = (3,)
SEEDS = 100


def setup_v2() -> None:
    v1.predictor_names = v2.predictor_names
    v1.predict = v2.predict
    v1.routed_support = v2.routed_support


def raw_selected(cells, horizon, budget, seeds, baseline_predictor, method_predictor):
    names = [c["name"] for c in cells]
    truth = np.asarray([c["life"] for c in cells], float)
    protocol = np.asarray([c["protocol"] for c in cells], float)
    curve = np.asarray([c["curve"] for c in cells], float)
    clients = np.arange(len(cells), dtype=int)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distances = {
        w: dc if math.isinf(w) else np.sqrt(np.square(dp) + w * np.square(dc))
        for w in v1.WEIGHTS
    }
    spread = v1.protocol_dispersion(dp)
    by_arm = {"baseline": defaultdict(list), "method": defaultdict(list)}
    route_counts = defaultdict(int)
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        router_support, route = v1.routed_support(
            spread, k, random_support, acquisition, clients, distances, tie_rank
        )
        route_counts[route] += 1
        pred_b = v1.predict(baseline_predictor, distances, truth, random_support, test)
        pred_m = v1.predict(method_predictor, distances, truth, router_support, test)
        for arm, pred in (("baseline", pred_b), ("method", pred_m)):
            for pos, idx in enumerate(test):
                y = float(truth[idx]); p = float(pred[pos]); err = p - y
                ape = 100.0 * abs(err) / max(y, 1.0)
                smape = 200.0 * abs(err) / max(abs(p) + abs(y), 1e-12)
                by_arm[arm][names[idx]].append((err, abs(err), err * err, ape, smape))
    summary = {}
    for arm in ("baseline", "method"):
        rows = []
        for name in names:
            values = np.asarray(by_arm[arm][name], float)
            if len(values) == 0:
                continue
            rows.append({
                "held_out": name,
                "mae": float(np.mean(values[:, 1])),
                "rmse": float(np.sqrt(np.mean(values[:, 2]))),
                "mape": float(np.mean(values[:, 3])),
                "smape": float(np.mean(values[:, 4])),
                "n_predictions": int(len(values)),
            })
        summary[arm] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("mae", "rmse", "mape", "smape")
        }
        summary[arm]["n_cells"] = len(rows)
        summary[arm]["per_cell"] = rows
    return {"route_counts": dict(route_counts), "protocol_dispersion": spread, "summary": summary}


def main() -> None:
    setup_v2()
    output = {
        "protocol": "exact v2 H10/H20/H50 K3 100-seed fixed-pool protocol; raw predictions retained only in this audit",
        "results": [],
    }
    for horizon in HORIZONS:
        loaded = {d: load_cells(d, horizon) for d in DATASETS}
        evaluations = {
            d: v1.evaluate(loaded[d], horizon, 3, SEEDS, 1e9, False)
            for d in DATASETS
        }
        for target in DATASETS:
            development = [evaluations[d] for d in DATASETS if d != target]
            choice, _ = v2.choose_baseline(development)
            baseline_predictor = choice["predictor"]
            target_eval = evaluations[target]
            active_routes = [r for r in target_eval["route_counts"] if r.startswith("active_")]
            if active_routes:
                spread = target_eval["protocol_dispersion"]
                rho = v2.distance_concordance(loaded[target])
                if spread >= 0.60:
                    method_predictor = (
                        "support_median"
                        if rho < v2.CONCORDANCE_THRESHOLD
                        else "w0.5_bw0.5"
                    )
                else:
                    weight = active_routes[0].split("active_w", 1)[1]
                    method_predictor = f"w{weight}_bw0.5"
            else:
                method_predictor = baseline_predictor
            raw = raw_selected(
                loaded[target], horizon, 3, SEEDS, baseline_predictor, method_predictor
            )
            output["results"].append({
                "horizon": horizon,
                "target": target,
                "baseline_predictor": baseline_predictor,
                "method_predictor": method_predictor,
                **raw,
            })
    for h in HORIZONS:
        records = [r for r in output["results"] if r["horizon"] == h]
        output[f"macro_h{h}_k3"] = {
            arm: {
                metric: float(np.mean([r["summary"][arm][metric] for r in records]))
                for metric in ("mae", "rmse", "mape", "smape")
            }
            for arm in ("baseline", "method")
        }
        output[f"relative_reduction_h{h}_k3_percent"] = {
            metric: 100.0 * (
                output[f"macro_h{h}_k3"]["baseline"][metric]
                - output[f"macro_h{h}_k3"]["method"][metric]
            ) / max(output[f"macro_h{h}_k3"]["baseline"][metric], 1e-12)
            for metric in ("mae", "rmse", "mape", "smape")
        }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for h in HORIZONS:
        print(f"H{h}/K3", json.dumps({
            "baseline": output[f"macro_h{h}_k3"]["baseline"],
            "method": output[f"macro_h{h}_k3"]["method"],
            "relative_reduction_percent": output[f"relative_reduction_h{h}_k3_percent"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
