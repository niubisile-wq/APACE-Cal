"""Strict label-blind manifest builder for Zenodo 21700 Expt5.

Only ZIP central-directory metadata and the first 50 rows of each cycle-summary
member are read. Performance-summary/EOL members are never requested here.
The large archive is accessed solely through HTTP Range requests.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_zenodo_21700_prelabel as io217
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


HERE = Path(__file__).parent
OUT = HERE / "batterylife_zenodo_21700_expt5_prelabel.json"
API = "https://zenodo.org/api/records/10637534/files/Expt%205%20-%20Standard%20Cycle%20Aging%20(Control).zip/content"
MAX_H = 50
SEEDS = 100


def main() -> None:
    size, entries = io217.central_directory(API)
    cells = []
    members = {}
    temps = {c: (10.0 if c in "ABC" else 25.0 if c in "DE" else 40.0) for c in "ABCDEFGH"}
    for cell in "ABCDEFGH":
        suffix = f"Summary per Cycle/expt 5 - cell {cell} - cycle_data.csv"
        names = [n for n in entries if n.endswith(suffix)]
        if len(names) != 1:
            raise RuntimeError(f"cycle summary mismatch for {cell}: {names}")
        name = names[0]; members[cell] = name
        raw = io217.member_bytes(API, entries[name], limit=8 * 1024 * 1024)
        header, rows = io217.first_rows(raw)
        lookup = {x: i for i, x in enumerate(header)}
        values = []
        for row in rows[:MAX_H]:
            text = row[lookup["Discharge Capacity [A h]"]].strip()
            values.append(float(text) if text else np.nan)
        values = np.asarray(values, float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise RuntimeError(f"insufficient early capacity values for {cell}")
        if not valid.all():
            values = np.interp(np.arange(MAX_H), np.flatnonzero(valid), values[valid])
        cells.append({
            "name": f"21700_expt5_cell_{cell}",
            "cell": cell,
            "temperature_C": temps[cell],
            "protocol": [temps[cell], 0.0, 100.0, 5.0, 0.0],
            "curve_h50_capacity_Ah": values.tolist(),
            "cycle_summary_member": name,
        })

    protocol = np.asarray([c["protocol"] for c in cells], float)
    curve = np.asarray([c["curve_h50_capacity_Ah"] for c in cells], float)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distances = {w: dc if math.isinf(w) else np.sqrt(dp**2 + w * dc**2) for w in v1.WEIGHTS}
    spread = v1.protocol_dispersion(dp)
    clients = np.arange(len(cells), dtype=int)
    names = [c["name"] for c in cells]
    settings = []
    for horizon in (10, 20, 50):
        curve_h = curve[:, :horizon]
        # Rebuild H-specific distances exactly from first-H curve features.
        dc_h = distance_matrix(curve_h, robust_scale(curve_h), 1e9)
        dist_h = {w: dc_h if math.isinf(w) else np.sqrt(dp**2 + w * dc_h**2) for w in v1.WEIGHTS}
        for budget in (1, 3, 5, 10):
            rng_probe = np.random.default_rng(900_000 + horizon * 100 + budget)
            route_counts = {}
            episodes = []
            for seed in range(1, SEEDS + 1):
                rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
                perm = rng.permutation(len(cells))
                pool_n = min(len(cells) - 2, max(int(math.ceil(.7 * len(cells))), budget))
                acquisition = np.sort(perm[:pool_n]); test = np.sort(perm[pool_n:]); k = min(budget, len(acquisition))
                tie_order = rng.permutation(len(cells)); tie_rank = np.empty(len(cells), int); tie_rank[tie_order] = np.arange(len(cells))
                random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
                router_support, route = v2.routed_support(spread, k, random_support, acquisition, clients, dist_h, tie_rank)
                route_counts[route] = route_counts.get(route, 0) + 1
                if route.startswith("active_"):
                    weight = route.split("active_w", 1)[1]
                    method_predictor = f"w{weight}_bw0.5"
                else:
                    method_predictor = "logmean" if k == 1 else "w2_bw0.5"
                episodes.append({
                    "seed": seed,
                    "acquisition": [names[i] for i in acquisition],
                    "test": [names[i] for i in test],
                    "baseline_support": [names[i] for i in random_support],
                    "method_support": [names[i] for i in router_support],
                    "route": route,
                    "baseline_predictor": "logmean" if k == 1 else "w2_bw0.5",
                    "method_predictor": method_predictor,
                })
            settings.append({"horizon": horizon, "label_budget_k": budget,
                             "protocol_dispersion": spread, "route_counts": route_counts,
                             "episodes": episodes})
    output = {
        "phase": "PRELABEL_FROZEN_MANIFEST_V2",
        "dataset": "Zenodo-21700-Expt5",
        "archive_url": API,
        "archive_size_bytes": size,
        "archive_member_count": len(entries),
        "information_barrier": "Only first 50 rows of cycle-summary members were parsed; performance-summary/EOL members were never requested.",
        "cells": cells,
        "protocol_dispersion": spread,
        "settings": settings,
        "active_settings": [s for s in settings if any(r.startswith("active_") for r in s["route_counts"])],
        "source_entry_hash": hashlib.sha256("\n".join(sorted(members.values())).encode()).hexdigest(),
        "method_script_sha256": hashlib.sha256((HERE / "batterylife_asymmetric_cohort_router_v2.py").read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"protocol_dispersion": spread,
                      "active_settings": len(output["active_settings"]),
                      "routes": {r: sum(s["route_counts"].get(r, 0) for s in settings) for r in sorted({r for s in settings for r in s["route_counts"]})}}, indent=2))


if __name__ == "__main__":
    main()
