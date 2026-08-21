"""E8 target-EOL experimental cost audit for random-K versus APACE-K.

Selection is reproduced before reading support/test life values. Life values
are used only after the support identities are fixed to estimate actual
serial-cycle and parallel-wall-clock proxies.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells


HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_cost_audit.json"


def main():
    rows = []
    for horizon in (10, 20, 50):
        for dataset in DATASETS:
            cells = load_cells(dataset, horizon)
            names = [c["name"] for c in cells]
            life = np.asarray([c["life"] for c in cells], dtype=float)
            protocol = np.asarray([c["protocol"] for c in cells], dtype=float)
            curve = np.asarray([c["curve"] for c in cells], dtype=float)
            p_scale, c_scale = v1.robust_scale(protocol), v1.robust_scale(curve)
            dp = v1.distance_matrix(protocol, p_scale, 1e9)
            dc = v1.distance_matrix(curve, c_scale, 1e9)
            distances = {w: dc if math.isinf(w) else np.sqrt(dp * dp + w * dc * dc)
                         for w in v1.WEIGHTS}
            spread = v1.protocol_dispersion(dp)
            for budget in (1, 3, 5, 10):
                for seed in range(1, 101):
                    rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
                    permutation = rng.permutation(len(cells))
                    acquisition_n = min(len(cells) - 2,
                                        max(int(math.ceil(0.7 * len(cells))), budget))
                    acquisition = np.sort(permutation[:acquisition_n])
                    shuffled = rng.permutation(len(cells))
                    tie_rank = np.empty(len(cells), dtype=int)
                    tie_rank[shuffled] = np.arange(len(cells))
                    k = min(budget, len(acquisition))
                    random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
                    apace_support, route = v2.routed_support(
                        spread, k, random_support, acquisition,
                        np.arange(len(cells)), distances, tie_rank
                    )
                    for arm, support in (("random", random_support), ("apace", apace_support)):
                        selected_life = life[support]
                        rows.append({
                            "horizon": horizon,
                            "dataset": dataset,
                            "label_budget_k": budget,
                            "seed": seed,
                            "arm": arm,
                            "route": route if arm == "apace" else "random",
                            "support_names": [names[int(i)] for i in support],
                            "serial_cycles": float(np.sum(selected_life)),
                            "parallel_cycles": float(np.max(selected_life)),
                            "mean_life": float(np.mean(selected_life)),
                        })
    output = {
        "protocol": (
            "same v2 fixed-pool support identities; life used only after selection; "
            "serial sum and parallel max cycle life are cost proxies; no energy claim "
            "without current/voltage integration"
        ),
        "rows": rows,
        "summary": {},
    }
    for horizon in (10, 20, 50):
        for budget in (1, 3, 5, 10):
            for metric in ("serial_cycles", "parallel_cycles", "mean_life"):
                values = {}
                for arm in ("random", "apace"):
                    z = [r[metric] for r in rows if r["horizon"] == horizon
                         and r["label_budget_k"] == budget and r["arm"] == arm]
                    values[arm] = float(np.mean(z))
                output["summary"][f"h{horizon}_k{budget}_{metric}"] = values
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for h, k in ((10, 3), (20, 3), (50, 3), (50, 5), (50, 10)):
        print(h, k, {m: output["summary"][f"h{h}_k{k}_{m}"]
                      for m in ("serial_cycles", "parallel_cycles", "mean_life")})


if __name__ == "__main__":
    main()

