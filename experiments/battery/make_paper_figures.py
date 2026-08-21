"""Generate manuscript-ready figures from frozen JSON artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "paper_figures"
OUT.mkdir(exist_ok=True)


def main():
    d = json.loads((HERE / "batterylife_apace_v2_stats.json").read_text())["results"]
    horizons, budgets = (10, 20, 50), (1, 3, 5, 10)
    baseline = np.array([[d[f"h{h}_k{k}"]["macro_baseline_mape"] for k in budgets] for h in horizons])
    method = np.array([[d[f"h{h}_k{k}"]["macro_method_mape"] for k in budgets] for h in horizons])
    reduction = 100 * (baseline - method) / baseline

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for ax, values, title in zip(axes, (baseline, method, reduction), ("Matched baseline MAPE (%)", "APACE-Cal v2 MAPE (%)", "Relative reduction (%)")):
        im = ax.imshow(values, cmap="viridis", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(budgets)), [f"K={k}" for k in budgets])
        ax.set_yticks(range(len(horizons)), [f"H={h}" for h in horizons])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "fig_main_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_main_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    primary = [d[f"h{h}_k3"] for h in horizons]
    means = [x["relative_reduction_percent"] for x in primary]
    lows = [x["hierarchical_ci95_percent"][0] for x in primary]
    highs = [x["hierarchical_ci95_percent"][1] for x in primary]
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    x = np.arange(3)
    ax.errorbar(x, means, yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)], fmt="o-", capsize=4, lw=1.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, ["H=10/K=3", "H=20/K=3", "H=50/K=3"])
    ax.set_ylabel("Relative MAPE reduction (%)")
    ax.set_title("Hierarchical bootstrap effect (six development domains)")
    fig.savefig(OUT / "fig_primary_ci.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_primary_ci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {"source": "batterylife_apace_v2_stats.json", "files": [p.name for p in sorted(OUT.iterdir())]}
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
