"""E7 support-set stability audit from the cost-audit support ledger."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
SOURCE = HERE / "batterylife_apace_cost_audit.json"
OUT = HERE / "batterylife_apace_support_stability.json"


def main():
    data = json.loads(SOURCE.read_text())
    rows = data["rows"]
    output = {"protocol": "Jaccard over 100 same-protocol episode support sets", "summary": {}}
    for horizon in (10, 20, 50):
        for dataset in sorted({r["dataset"] for r in rows}):
            for budget in (1, 3, 5, 10):
                entry = {}
                for arm in ("random", "apace"):
                    sets = [
                        set(r["support_names"])
                        for r in rows
                        if r["horizon"] == horizon and r["dataset"] == dataset
                        and r["label_budget_k"] == budget and r["arm"] == arm
                    ]
                    if len(sets) < 2:
                        entry[arm] = None
                        continue
                    scores = [
                        len(a & b) / max(len(a | b), 1)
                        for a, b in combinations(sets, 2)
                    ]
                    entry[arm] = {
                        "mean_jaccard": float(np.mean(scores)),
                        "median_jaccard": float(np.median(scores)),
                        "n_pairs": len(scores),
                    }
                output["summary"][f"{dataset}_h{horizon}_k{budget}"] = entry
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for h, k in ((10, 3), (20, 3), (50, 3)):
        values = [v for key, v in output["summary"].items()
                  if key.endswith(f"_h{h}_k{k}")]
        print(h, k, {arm: round(float(np.mean([x[arm]["mean_jaccard"] for x in values])), 4)
                     for arm in ("random", "apace")})


if __name__ == "__main__":
    main()

