"""Strong label-free selector baselines under the frozen fixed-pool protocol.

This is a diagnostic comparison, not a change to APACE-Cal.  Every selector
uses only protocol/early-curve features from the acquisition pool.  All
methods share the same K labels, test cells, log-mean predictor, and episode
seeds.  The output is intentionally separate from the frozen APACE artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import DATASETS, load_cells, nan_rms, robust_scale


HERE = Path(__file__).parent
OUT = HERE / "batterylife_strong_selector_baselines.json"
HORIZONS = (10, 20, 50)
BUDGETS = (1, 3, 5, 10)
SEEDS = 100


def pairwise(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = nan_rms(x[i], x[j], scale)
    return out


def greedy_kcenter(distance: np.ndarray, pool: np.ndarray, k: int, rank: np.ndarray) -> np.ndarray:
    """Greedy farthest-first traversal, a standard coverage baseline."""
    if k <= 1:
        return np.asarray([min(pool, key=lambda i: (rank[i], i))], dtype=int)
    first = min(pool, key=lambda i: (rank[i], i))
    selected = [int(first)]
    nearest = distance[pool, first].copy()
    for _ in range(1, min(k, len(pool))):
        candidates = [i for i in pool if i not in selected]
        chosen = max(candidates, key=lambda i: (float(nearest[list(pool).index(i)]), -rank[i], -i))
        selected.append(int(chosen))
        nearest = np.minimum(nearest, distance[pool, chosen])
    return np.asarray(selected, dtype=int)


def facility_medoids(distance: np.ndarray, pool: np.ndarray, k: int, rank: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    current = np.full(len(pool), np.inf)
    for _ in range(min(k, len(pool))):
        candidates = []
        for candidate in pool:
            if int(candidate) in selected:
                continue
            cost = float(np.sum(np.minimum(current, distance[pool, candidate])))
            candidates.append((cost, int(rank[candidate]), int(candidate)))
        _, _, chosen = min(candidates)
        selected.append(chosen)
        current = np.minimum(current, distance[pool, chosen])
    return np.asarray(selected, dtype=int)


def logmean(y: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, np.exp(np.mean(np.log(np.maximum(y, 1e-9)))))


def evaluate(cells: list[dict], horizon: int, budget: int) -> dict:
    names = [c["name"] for c in cells]
    truth = np.asarray([c["life"] for c in cells], float)
    protocol = np.asarray([c["protocol"] for c in cells], float)
    curve = np.asarray([c["curve"] for c in cells], float)
    dp = pairwise(protocol, robust_scale(protocol))
    dc = pairwise(curve, robust_scale(curve))
    distances = {
        "protocol_medoid": dp,
        "curve_medoid": dc,
        "hybrid_medoid": np.sqrt(dp**2 + dc**2),
        "protocol_kcenter": dp,
        "curve_kcenter": dc,
        "hybrid_kcenter": np.sqrt(dp**2 + dc**2),
    }
    methods = tuple(distances) + ("random",)
    agg = {m: defaultdict(lambda: [0.0, 0.0, 0]) for m in methods}
    for seed in range(1, SEEDS + 1):
        rng = np.random.default_rng(20_000_000 * horizon + 10_000 * budget + seed)
        perm = rng.permutation(len(cells))
        pool_n = min(len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget))
        pool = np.sort(perm[:pool_n])
        test = np.sort(perm[pool_n:])
        k = min(budget, len(pool))
        rank_order = rng.permutation(len(cells))
        rank = np.empty(len(cells), dtype=int)
        rank[rank_order] = np.arange(len(cells))
        supports = {"random": np.sort(rng.choice(pool, size=k, replace=False))}
        for method, distance in distances.items():
            if "kcenter" in method:
                supports[method] = greedy_kcenter(distance, pool, k, rank)
            else:
                supports[method] = facility_medoids(distance, pool, k, rank)
        for method, support in supports.items():
            pred = logmean(truth[support], len(test))
            ae = np.abs(pred - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for pos, idx in enumerate(test):
                rec = agg[method][names[idx]]
                rec[0] += float(ae[pos]); rec[1] += float(ape[pos]); rec[2] += 1
    summary = {}
    for method in methods:
        rows = []
        for name in names:
            ae, ape, count = agg[method][name]
            if count:
                rows.append({"held_out": name, "mae": ae / count, "mape": ape / count})
        summary[method] = {
            "mae": float(np.mean([r["mae"] for r in rows])),
            "mape": float(np.mean([r["mape"] for r in rows])),
            "n_cells": len(rows),
            "per_cell": rows,
        }
    return summary


def main() -> None:
    output = {
        "protocol": "100 fixed-pool episodes; 70% unlabeled acquisition pool; common test pool; label-free selector; log-mean predictor",
        "source": "batterylife_curve_aware_support.load_cells",
        "source_hash": hashlib.sha256((HERE / "batterylife_strong_selector_baselines.py").read_bytes()).hexdigest(),
        "results": {},
    }
    for horizon in HORIZONS:
        for dataset in DATASETS:
            cells = load_cells(dataset, horizon)
            for budget in BUDGETS:
                key = f"h{horizon}_{dataset}_k{budget}"
                output["results"][key] = evaluate(cells, horizon, budget)
                print(key, json.dumps({m: round(v["mape"], 4) for m, v in output["results"][key].items()}, sort_keys=True))
    for horizon in HORIZONS:
        for budget in BUDGETS:
            rows = [output["results"][f"h{horizon}_{d}_k{budget}"] for d in DATASETS]
            output[f"macro_h{horizon}_k{budget}"] = {
                m: {metric: float(np.mean([row[m][metric] for row in rows])) for metric in ("mae", "mape")}
                for m in ("random", "protocol_medoid", "curve_medoid", "hybrid_medoid", "protocol_kcenter", "curve_kcenter", "hybrid_kcenter")
            }
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
