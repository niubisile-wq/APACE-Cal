"""Post-hoc risk/return and non-inferiority audit for frozen APACE-Cal results.

This analysis does not tune thresholds or rewrite source results.  It reports
active-route coverage, paired per-cell gain, worst degradation, and the fraction
of comparisons satisfying predeclared non-inferiority margins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
SOURCE = HERE / "batterylife_asymmetric_cohort_router_v2.json"
OUT = HERE / "batterylife_risk_return_analysis.json"
MARGINS = (0.01, 0.02, 0.05)  # relative MAPE degradation margins


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for record in source["results"]:
        routes = record.get("route_counts", {})
        active_episodes = sum(v for k, v in routes.items() if "active" in k)
        total_episodes = sum(routes.values())
        paired = []
        for cell in record.get("per_cell", []):
            b = float(cell["baseline_ape"])
            m = float(cell["method_ape"])
            paired.append({
                "held_out": cell["held_out"],
                "baseline_mape": b,
                "method_mape": m,
                "relative_change": (m - b) / max(abs(b), 1e-12),
            })
        changes = np.asarray([x["relative_change"] for x in paired], dtype=float)
        rows.append({
            "horizon": record["horizon"],
            "target": record["target"],
            "label_budget_k": record["label_budget_k"],
            "active_episode_fraction": active_episodes / max(total_episodes, 1),
            "route_counts": routes,
            "mean_relative_change": float(np.mean(changes)) if len(changes) else 0.0,
            "worst_relative_change": float(np.max(changes)) if len(changes) else 0.0,
            "improved_tied_worse_cells": [
                int(np.sum(changes < -1e-12)),
                int(np.sum(np.abs(changes) <= 1e-12)),
                int(np.sum(changes > 1e-12)),
            ],
            "noninferior_fraction": {
                str(margin): float(np.mean(changes <= margin)) if len(changes) else 1.0
                for margin in MARGINS
            },
        })
    output = {
        "source": SOURCE.name,
        "margins_relative_mape": MARGINS,
        "interpretation": "Margins are descriptive audits on frozen outputs; no threshold or method parameter was tuned here.",
        "rows": rows,
    }
    for h in (10, 20, 50):
        for k in (1, 3, 5, 10):
            subset = [r for r in rows if r["horizon"] == h and r["label_budget_k"] == k]
            if not subset:
                continue
            output[f"macro_h{h}_k{k}"] = {
                "active_episode_fraction_mean": float(np.mean([r["active_episode_fraction"] for r in subset])),
                "mean_relative_change": float(np.mean([r["mean_relative_change"] for r in subset])),
                "worst_domain_relative_change": float(np.max([r["worst_relative_change"] for r in subset])),
                "cell_rows_improved_tied_worse": [
                    int(sum(r["improved_tied_worse_cells"][0] > r["improved_tied_worse_cells"][2] for r in subset)),
                    int(sum(r["improved_tied_worse_cells"][0] == r["improved_tied_worse_cells"][2] for r in subset)),
                    int(sum(r["improved_tied_worse_cells"][0] < r["improved_tied_worse_cells"][2] for r in subset)),
                ],
                "mean_noninferior_fraction": {
                    str(margin): float(np.mean([r["noninferior_fraction"][str(margin)] for r in subset]))
                    for margin in MARGINS
                },
            }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for key in ("macro_h10_k3", "macro_h20_k3", "macro_h50_k3"):
        print(key, json.dumps(output[key], sort_keys=True))


if __name__ == "__main__":
    main()
