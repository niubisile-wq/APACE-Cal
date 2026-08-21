"""Build the complete label-blind v3 episode manifest for UConn-ISU-ILCC."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_apace_stability_gate_candidate as gate
import batterylife_blind_prelabel_manifest as common
import batterylife_uconn_isu_prelabel as pre


HERE = Path(__file__).parent
DEV = HERE / "batterylife_asymmetric_cohort_router_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setting(cells, horizon, budget, seeds, baseline_predictor, stability_row):
    names = [c["name"] for c in cells]
    protocol = np.asarray([c["protocol"] for c in cells], dtype=float)
    curve = np.asarray([c["curve"] for c in cells], dtype=float)
    clients = np.arange(len(cells), dtype=int)
    p_scale, c_scale = pre.robust_scale(protocol), pre.robust_scale(curve)
    dp = pre.distance_matrix(protocol, p_scale, 1e9)
    dc = pre.distance_matrix(curve, c_scale, 1e9)
    weights = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, math.inf)
    distances = {w: dc if math.isinf(w) else np.sqrt(dp * dp + w * dc * dc) for w in weights}
    spread = float(np.median(dp[np.triu_indices(len(cells), 1)]))
    rho = v2.distance_concordance(cells)
    episodes, routes = [], defaultdict(int)
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
        if stability_row["active_allowed"]:
            method_support, route = v2.routed_support(
                spread, k, random_support, acquisition, clients, distances, tie_rank
            )
        else:
            method_support, route = random_support, "fallback_stability_or_coverage"
        if route.startswith("active_"):
            if spread >= 0.60:
                method_predictor = "support_median" if rho < v2.CONCORDANCE_THRESHOLD else "w0.5_bw0.5"
            else:
                method_predictor = "w2_bw0.5"
        else:
            method_predictor = baseline_predictor
        routes[route] += 1
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
        "horizon": horizon, "label_budget_k": budget, "n_unlabeled_cells": len(cells),
        "protocol_dispersion": spread, "distance_concordance_spearman": rho,
        "route_counts": dict(routes), "stability": stability_row, "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_isu"))
    ap.add_argument("--output", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_isu/uconn_isu_manifest_v3.json"))
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    root = args.root
    par = pre.params(root / "cycling_parameters.csv")
    members = pre.member_map([root / "cycling_part1.zip", root / "cycling_part2.zip"])
    cache = {cid: pre.read_cell(members[cid], 50) for cid in sorted(par)}
    loaded = {}
    for h in (10, 20, 50):
        loaded[h] = [
            {"name": f"UConn_ISU_cell_{cid}", "protocol": par[cid]["protocol"],
             "curve": pre.curve_features(cache[cid], h)}
            for cid in sorted(par) if pre.curve_features(cache[cid], h) is not None
        ]
    dev = json.loads(DEV.read_text())
    settings, predictor_freeze, active = [], {}, []
    for h in (10, 20, 50):
        stability = gate.stability(loaded[h], h, "UConn_ISU", (1, 3, 5, 10))[1]
        for k in (1, 3, 5, 10):
            choice, ranking = common.external_predictor(dev, h, k)
            predictor_freeze[f"h{h}_k{k}"] = {"selected": choice, "ranking": ranking}
            row = setting(loaded[h], h, k, args.seeds, choice["predictor"], stability[k])
            settings.append(row)
            if any(route.startswith("active_") for route in row["route_counts"]):
                active.append(f"h{h}_k{k}")
    output = {
        "phase": "UCONN_ISU_V3_PRELABEL_FROZEN", "dataset": "UConn-ISU-ILCC LFP/Gr",
        "label_read": False, "archive_root": str(root), "active_settings": active,
        "predictor_freeze": predictor_freeze, "settings": settings,
        "method_candidate_sha256": sha256(HERE / "batterylife_apace_stability_gate_candidate.py"),
        "manifest_builder_sha256": sha256(Path(__file__)),
        "prelabel_reader_sha256": sha256(HERE / "batterylife_uconn_isu_prelabel.py"),
        "development_result_sha256": sha256(DEV),
        "explicit_non_access_statement": "Only cycling data and protocol parameters were read; RPT archive was not opened.",
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cells": {h: len(loaded[h]) for h in loaded}, "active_settings": active,
                      "routes": {f"h{x['horizon']}_k{x['label_budget_k']}": x["route_counts"] for x in settings}}, indent=2))


if __name__ == "__main__":
    main()
