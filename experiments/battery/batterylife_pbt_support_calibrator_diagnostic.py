"""Development-only diagnostic of PBT residual calibrators on APACE supports.

All listed calibrators are evaluated together. No candidate is selected from
these target labels and no result modifies the frozen APACE-Cal method.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import load_cells
from batterylife_apace_pbt_fixed_pool_plugin import pbt_ensemble


HERE = Path(__file__).parent
PBT_JSON = HERE / "pbt_official_unseen_eval.json"
OUT = HERE / "batterylife_pbt_support_calibrator_diagnostic.json"


def calibrate(name, q, support, names, pbt, life, distances, query_index):
    qlog = math.log(max(q, 1.0))
    residual = np.asarray(
        [math.log(max(life[n], 1.0)) - math.log(max(pbt[n], 1.0)) for n in support],
        dtype=float,
    )
    if name == "raw":
        return q
    if name == "logbias_median":
        return float(math.exp(qlog + np.median(residual)))
    if name == "support_median":
        return float(np.median([life[n] for n in support]))
    if name == "residual_nearest":
        support_indices = [names.index(item) for item in support]
        dist = distances[2.0][query_index, support_indices]
        return float(math.exp(qlog + residual[int(np.argmin(dist))]))
    if name.startswith("residual_kernel_"):
        _, _, wtxt, btxt = name.split("_")
        bandwidth = float(btxt)
        weight = math.inf if wtxt == "inf" else float(wtxt)
        support_indices = [names.index(item) for item in support]
        local = distances[weight][query_index, support_indices]
        weights = np.exp(-0.5 * np.square(local / bandwidth))
        weights /= max(float(weights.sum()), 1e-12)
        return float(math.exp(qlog + weights @ residual))
    raise KeyError(name)


def main():
    pbt = json.loads(PBT_JSON.read_text())
    candidates = ["raw", "logbias_median", "support_median", "residual_nearest"]
    candidates += [f"residual_kernel_{w}_{b}" for w in ("0", "2", "inf") for b in ("0.5", "1.0", "2.0")]
    output = {"protocol": "development diagnostic; all calibrators reported; no post-hoc selection", "results": {}}
    for pbt_dataset, local_dataset in (("CALB", "CALB"), ("HNEI", "HNEI")):
        for horizon in (10, 20, 50):
            ensemble = pbt_ensemble(pbt, pbt_dataset, horizon)
            cells = [c for c in load_cells(local_dataset, horizon) if c["name"] in ensemble]
            names = [c["name"] for c in cells]
            life = {c["name"]: float(c["life"]) for c in cells}
            protocol = np.asarray([c["protocol"] for c in cells], dtype=float)
            curve = np.asarray([c["curve"] for c in cells], dtype=float)
            p_scale, c_scale = v1.robust_scale(protocol), v1.robust_scale(curve)
            dp = v1.distance_matrix(protocol, p_scale, 1e9)
            dc = v1.distance_matrix(curve, c_scale, 1e9)
            distances = {w: dc if math.isinf(w) else np.sqrt(dp * dp + w * dc * dc) for w in v1.WEIGHTS}
            spread = v1.protocol_dispersion(dp)
            rows = {name: [] for name in candidates}
            for budget in (3, 5, 10):
                by_candidate = {name: [] for name in candidates}
                for seed in range(1, 101):
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
                    support, _ = v2.routed_support(spread, k, random_support, acquisition,
                                                   np.arange(len(cells)), distances, tie_rank)
                    support_names = [names[int(i)] for i in support]
                    for index in test:
                        qname = names[int(index)]
                        local_dist = distances[2.0][index, support]
                        for arm, selected in (("random", random_support), ("apace", support)):
                            selected_names = [names[int(i)] for i in selected]
                            for candidate in candidates:
                                pred = calibrate(candidate, ensemble[qname], selected_names,
                                                 names, ensemble, life, distances, int(index))
                                ae = abs(pred - life[qname])
                                by_candidate[candidate].append({"arm": arm, "held_out": qname,
                                                                 "ae": ae, "ape": 100 * ae / max(life[qname], 1.0)})
                for candidate, values in by_candidate.items():
                    for arm in ("random", "apace"):
                        cell = defaultdict(list)
                        for row in values:
                            if row["arm"] == arm:
                                cell[row["held_out"]].append(row["ape"])
                        mape = float(np.mean([np.mean(v) for v in cell.values()]))
                        rows[candidate].append({"budget": budget, "arm": arm, "mape": mape})
            output["results"][f"{pbt_dataset}_h{horizon}"] = {
                "protocol_dispersion": spread,
                "rows": rows,
            }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for key, value in output["results"].items():
        print(key)
        for candidate in ("raw", "logbias_median", "support_median", "residual_kernel_2_0.5"):
            z = value["rows"][candidate]
            print(candidate, [(x["budget"], x["arm"], round(x["mape"], 3)) for x in z])


if __name__ == "__main__":
    main()
