"""Hierarchical domain/cell bootstrap for the multimetric audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
SOURCE = HERE / "batterylife_multimetric_audit.json"
OUT = HERE / "batterylife_multimetric_hierarchical_stats.json"
BOOTSTRAPS = 10000
METRICS = ("mae", "rmse", "mape", "smape")


def main() -> None:
    source = json.loads(SOURCE.read_text())
    output = {
        "source": SOURCE.name,
        "unit": "domain-level outer bootstrap and cell-level inner bootstrap",
        "bootstraps": BOOTSTRAPS,
        "results": {},
    }
    for h in (10, 20, 50):
        records = [r for r in source["results"] if r["horizon"] == h]
        arrays = {
            arm: [
                np.asarray(
                    [[row[metric] for metric in METRICS] for row in r["summary"][arm]["per_cell"]],
                    float,
                )
                for r in records
            ]
            for arm in ("baseline", "method")
        }
        rng = np.random.default_rng(20260820 + h)
        boot = np.empty((BOOTSTRAPS, len(METRICS)), float)
        for draw in range(BOOTSTRAPS):
            domain_indices = rng.integers(0, len(records), len(records))
            domain_means = []
            for arm in ("baseline", "method"):
                sampled_domains = []
                for index in domain_indices:
                    rows = arrays[arm][index]
                    cell_indices = rng.integers(0, len(rows), len(rows))
                    sampled_domains.append(np.mean(rows[cell_indices], axis=0))
                domain_means.append(np.mean(sampled_domains, axis=0))
            boot[draw] = 100.0 * (domain_means[0] - domain_means[1]) / np.maximum(domain_means[0], 1e-12)
        output["results"][f"h{h}_k3"] = {
            metric: {
                "relative_reduction_percent": float(np.mean(boot[:, i])),
                "ci95_percent": [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))],
                "probability_nonpositive": float(np.mean(boot[:, i] <= 0.0)),
            }
            for i, metric in enumerate(METRICS)
        }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
