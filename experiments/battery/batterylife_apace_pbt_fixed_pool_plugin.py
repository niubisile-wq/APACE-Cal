"""Fixed-total-K APACE support plugin for the official PBT ensemble.

The PBT checkpoints are frozen.  This experiment changes only the K target
support identities: random-K versus APACE-v2 routed-K.  PBT predictions are
never retrained and target test EOL labels are read only after support
selection.  It is a plugin/generalization experiment, not a modification of
the frozen native APACE-Cal predictor.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_asymmetric_cohort_router as v1
from batterylife_curve_aware_support import load_cells


HERE = Path(__file__).parent
PBT_JSON = HERE / "pbt_official_unseen_eval.json"
OUTPUT = HERE / "batterylife_apace_pbt_fixed_pool_plugin.json"
DATASET_MAP = {"CALB": "CALB", "HNEI": "HNEI"}
HORIZONS = (10, 20, 50)
BUDGETS = (1, 3, 5, 10)
SEEDS = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pbt_ensemble(source, dataset, horizon):
    runs = [r for r in source["runs"] if r["dataset"] == dataset and r["horizon"] == horizon]
    if len(runs) != 3:
        raise RuntimeError(f"PBT coverage is not three checkpoints for {dataset} H={horizon}")
    maps = [{row["file"]: float(row["prediction"]) for row in run["rows"]} for run in runs]
    names = set(maps[0])
    if any(set(mapping) != names for mapping in maps[1:]):
        raise RuntimeError("PBT checkpoint cell coverage differs")
    return {name: float(np.mean([mapping[name] for mapping in maps])) for name in sorted(names)}


def log_bias_prediction(query_prediction, support_names, pbt, life):
    residuals = [
        math.log(max(life[name], 1.0)) - math.log(max(pbt[name], 1.0))
        for name in support_names
    ]
    return float(math.exp(math.log(max(query_prediction, 1.0)) + np.median(residuals)))


def summarize(rows):
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["held_out"]].append(row)
    cell_rows = []
    for name, values in sorted(by_cell.items()):
        cell_rows.append(
            {
                "held_out": name,
                "ae": float(np.mean([v["ae"] for v in values])),
                "ape": float(np.mean([v["ape"] for v in values])),
            }
        )
    return {
        "mae": float(np.mean([row["ae"] for row in cell_rows])),
        "mape": float(np.mean([row["ape"] for row in cell_rows])),
        "n_cells": len(cell_rows),
        "cell_rows": cell_rows,
    }


def main():
    pbt = json.loads(PBT_JSON.read_text())
    output = {
        "protocol": (
            "official frozen three-checkpoint PBT ensemble; fixed total-K target pool; "
            "100 common episodes; random-K versus APACE-v2 routed-K; log-bias calibrator; "
            "support selection uses only protocol/early-curve features"
        ),
        "pbt_source_sha256": sha256(PBT_JSON),
        "results": {},
    }
    for pbt_dataset, local_dataset in DATASET_MAP.items():
        for horizon in HORIZONS:
            ensemble = pbt_ensemble(pbt, pbt_dataset, horizon)
            cells = [c for c in load_cells(local_dataset, horizon) if c["name"] in ensemble]
            if {c["name"] for c in cells} != set(ensemble):
                raise RuntimeError(f"PBT/local cell mismatch for {pbt_dataset} H={horizon}")
            names = [c["name"] for c in cells]
            life = {c["name"]: float(c["life"]) for c in cells}
            protocol = np.asarray([c["protocol"] for c in cells], dtype=float)
            curve = np.asarray([c["curve"] for c in cells], dtype=float)
            clients = np.arange(len(cells), dtype=int)
            p_scale = v1.robust_scale(protocol)
            c_scale = v1.robust_scale(curve)
            dp = v1.distance_matrix(protocol, p_scale, 1e9)
            dc = v1.distance_matrix(curve, c_scale, 1e9)
            distances = {
                weight: dc if math.isinf(weight) else np.sqrt(np.square(dp) + weight * np.square(dc))
                for weight in v1.WEIGHTS
            }
            spread = v1.protocol_dispersion(dp)
            rho = v2.distance_concordance(cells)
            for budget in BUDGETS:
                rows = {"random": [], "apace": []}
                route_counts = defaultdict(int)
                for seed in range(1, SEEDS + 1):
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
                    apace_support, route = v2.routed_support(
                        spread, k, random_support, acquisition, clients, distances, tie_rank
                    )
                    route_counts[route] += 1
                    support_map = {
                        "random": random_support,
                        "apace": apace_support,
                    }
                    for arm, support in support_map.items():
                        support_names = [names[int(i)] for i in support]
                        for index in test:
                            name = names[int(index)]
                            prediction = log_bias_prediction(
                                ensemble[name], support_names, ensemble, life
                            )
                            ae = abs(prediction - life[name])
                            rows[arm].append(
                                {
                                    "held_out": name,
                                    "seed": seed,
                                    "support": support_names,
                                    "prediction": prediction,
                                    "truth": life[name],
                                    "ae": ae,
                                    "ape": 100.0 * ae / max(life[name], 1.0),
                                }
                            )
                key = f"{pbt_dataset}_h{horizon}_k{budget}"
                random_summary = summarize(rows["random"])
                apace_summary = summarize(rows["apace"])
                random_map = {r["held_out"]: r for r in random_summary.pop("cell_rows")}
                apace_map = {r["held_out"]: r for r in apace_summary.pop("cell_rows")}
                diff = [random_map[n]["ape"] - apace_map[n]["ape"] for n in sorted(random_map)]
                output["results"][key] = {
                    "n_cells": len(cells),
                    "protocol_dispersion": spread,
                    "distance_concordance_rho": rho,
                    "route_counts": dict(route_counts),
                    "random_pbt_logbias": random_summary,
                    "apace_pbt_logbias": apace_summary,
                    "paired_cell": {
                        "mean_mape_reduction_pp": float(np.mean(diff)),
                        "improved_same_worse": [
                            int(np.sum(np.asarray(diff) > 1e-9)),
                            int(np.sum(np.abs(diff) <= 1e-9)),
                            int(np.sum(np.asarray(diff) < -1e-9)),
                        ],
                    },
                }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    for key, value in output["results"].items():
        print(key, value["random_pbt_logbias"]["mape"], value["apace_pbt_logbias"]["mape"],
              value["paired_cell"])


if __name__ == "__main__":
    main()

