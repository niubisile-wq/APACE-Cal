from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent / "draft"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "baseline": "#9AA4B2",
    "method": "#2F6F9F",
    "active": "#2A9D8F",
    "fallback": "#D9A441",
    "dark": "#263238",
    "light": "#E8EEF2",
    "accent": "#C65D4B",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tiff", dpi=600)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def fig1_domain_routing():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.7)
    ax.axis("off")
    ax.set_title("APACE-Cal couples target-domain routing with few-shot calibration", loc="left", pad=12, fontsize=10.5, fontweight="bold", color=COLORS["dark"])

    # Two visual stages make the information barrier and the fixed-budget contract explicit.
    ax.add_patch(Rectangle((0.18, 3.35), 9.64, 2.55, facecolor="#EAF1F7", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.18, 0.42), 9.64, 2.55, facecolor="#F3F5F6", edgecolor="none", zorder=0))
    ax.text(0.38, 5.62, "LABEL-BLIND ROUTING STAGE", fontsize=7.2, fontweight="bold", color=COLORS["method"], va="center")
    ax.text(0.38, 2.68, "FIXED-BUDGET CALIBRATION STAGE", fontsize=7.2, fontweight="bold", color="#59636E", va="center")

    def box(x, y, w, h, title, subtitle, edge, fill="#FFFFFF", title_color=COLORS["dark"], subtitle_color="#59636E", lw=1.2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10", linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color=title_color, fontsize=8.0, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha="center", va="center", color=subtitle_color, fontsize=7.0, zorder=4)

    def arrow(x1, y1, x2, y2, color="#738291", lw=1.35, style="-|>", connectionstyle="arc3,rad=0"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, linewidth=lw, color=color, connectionstyle=connectionstyle, zorder=2))

    # Main routing path.
    box(0.55, 4.28, 1.85, 0.92, "Target cohort", "protocol + early curves", COLORS["dark"], fill="#FFFFFF")
    box(3.05, 4.28, 2.05, 0.92, "Structure signals", r"$D_p$, $D_c$, $\rho$", COLORS["method"], fill="#FFFFFF")
    box(5.80, 4.28, 1.95, 0.92, "Safety gate", r"$r(D_p,\rho,K)$", COLORS["dark"], fill="#FFFFFF")
    arrow(2.40, 4.74, 3.05, 4.74)
    arrow(5.10, 4.74, 5.80, 4.74)

    # Explicit two-way decision with one shared budget.
    box(1.00, 1.65, 2.55, 0.82, "ACTIVE", "representative support", COLORS["active"], fill="#FFFFFF", title_color=COLORS["active"])
    box(5.05, 1.65, 2.55, 0.82, "ABSTAIN", "matched-random support", COLORS["fallback"], fill="#FFFFFF", title_color="#A87413")
    arrow(6.75, 4.28, 2.28, 2.47, connectionstyle="angle3,angleA=90,angleB=0")
    arrow(6.75, 4.28, 6.32, 2.47, connectionstyle="angle3,angleA=90,angleB=0")
    ax.text(4.35, 3.52, "same label budget $K$", ha="center", va="center", fontsize=7.4, color="#59636E", zorder=4)
    ax.text(4.35, 3.22, "complete labels only for selected support cells", ha="center", va="center", fontsize=6.8, color="#59636E", zorder=4)

    box(3.05, 0.52, 2.05, 0.82, "Local calibration", "fixed interface", COLORS["method"], fill="#FFFFFF")
    box(6.05, 0.52, 2.05, 0.82, "Evaluation", "predictions + MAPE", COLORS["dark"], fill="#FFFFFF")
    arrow(2.28, 1.65, 3.55, 1.34, connectionstyle="arc3,rad=-0.08")
    arrow(6.32, 1.65, 4.60, 1.34, connectionstyle="arc3,rad=0.08")
    arrow(5.10, 0.93, 6.05, 0.93)

    # Small legend for the two actions.
    ax.plot([8.85, 9.18], [5.58, 5.58], color=COLORS["active"], linewidth=3, solid_capstyle="round")
    ax.text(9.28, 5.58, "active", fontsize=6.8, color="#59636E", va="center")
    ax.plot([8.85, 9.18], [5.30, 5.30], color=COLORS["fallback"], linewidth=3, solid_capstyle="round")
    ax.text(9.28, 5.30, "abstain", fontsize=6.8, color="#59636E", va="center")

    save(fig, "Fig1_domain_routing")


def fig2_main_performance():
    horizons = ["H=10", "H=20", "H=50"]
    baseline = np.array([43.2863, 44.4487, 48.5352])
    method = np.array([24.2773, 22.4691, 22.9422])
    x = np.arange(len(horizons))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x - width / 2, baseline, width, label="Matched random", color=COLORS["baseline"])
    ax.bar(x + width / 2, method, width, label="APACE-Cal", color=COLORS["method"])
    for xi, b, m in zip(x, baseline, method):
        ax.text(xi + width / 2, m + 1.1, f"{(b-m)/b*100:.1f}%", ha="center", fontsize=7, color=COLORS["method"])
    ax.set_xticks(x, horizons)
    ax.set_ylabel("Six-domain macro-MAPE (%)")
    ax.set_title("APACE-Cal reduces macro-MAPE across early-observation horizons", loc="left", pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_ylim(0, 58)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax.set_axisbelow(True)
    add_panel_label(ax, "A")
    save(fig, "Fig2_main_performance")


def fig3_controls():
    labels = ["Source-only", "Random-$K$\nFT", "Uncertainty-$K$\nFT", "APACE-$K$\nFT"]
    values = [85.94, 36.84, 45.97, 30.47]
    colors = [COLORS["baseline"], "#B8C6D1", "#7FA5BE", COLORS["method"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Support selection contributes beyond target-label fine-tuning", loc="left", pad=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax.set_axisbelow(True)
    add_panel_label(ax, "B")
    save(fig, "Fig3_control_comparison")


def fig4_ablation():
    horizons = np.array([10, 20, 50])
    apace = np.array([24.2773, 22.4691, 22.9422])
    no_rho = np.array([24.0456, 22.5603, 22.9422])
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(horizons, apace, marker="o", linewidth=2.2, color=COLORS["method"], label="APACE-Cal")
    ax.plot(horizons, no_rho, marker="o", linewidth=1.8, color=COLORS["accent"], label=r"No-$\rho$")
    ax.set_xticks(horizons, ["H=10", "H=20", "H=50"])
    ax.set_ylabel("Six-domain macro-MAPE (%)")
    ax.set_title("Protocol--curve concordance changes route assignment", loc="left", pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(20, 26)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax.set_axisbelow(True)
    add_panel_label(ax, "C")
    save(fig, "Fig4_routing_ablation")


def fig5_external():
    active_names = ["SNU Dataset 1", "MathWorks\nLFP/Gr"]
    active_values = [30.9246, 47.99]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1.05, 1.25]})
    x = np.arange(2)
    bars = ax1.bar(x, active_values, color=COLORS["active"], width=0.58)
    for bar, value in zip(bars, active_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=7)
    ax1.set_xticks(x, active_names)
    ax1.set_ylabel("Relative MAPE reduction (%)")
    ax1.set_title("Active external evaluations", loc="left", pad=10)
    ax1.set_ylim(0, 60)
    ax1.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax1.set_axisbelow(True)
    add_panel_label(ax1, "D")

    fallback_names = ["NA-ion", "MATR", "HUST+XJTU"]
    unchanged = [34, 169, 100]
    bars = ax2.bar(fallback_names, unchanged, color=COLORS["fallback"], width=0.58)
    for bar, value in zip(bars, unchanged):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value}", ha="center", fontsize=7)
    ax2.set_ylabel("Cells retaining matched outcome")
    ax2.set_title("Fallback audits", loc="left", pad=10)
    ax2.set_ylim(0, 190)
    ax2.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax2.set_axisbelow(True)
    add_panel_label(ax2, "E")
    fig.tight_layout(w_pad=2.2)
    save(fig, "Fig5_external_validation")


def fig6_robustness():
    labels = ["10% label\nnoise", "50% pool", "60% pool", "80% pool", "10% curve\nmissing", "20% curve\nmissing", "30% curve\nmissing"]
    values = [22.61, 21.03, 22.14, 23.52, 22.55, 22.21, 25.40]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS["method"], width=0.64)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative MAPE improvement (%)")
    ax.set_title("Performance remains positive under moderate perturbations", loc="left", pad=10)
    ax.set_ylim(0, 30)
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.6)
    ax.set_axisbelow(True)
    add_panel_label(ax, "F")
    save(fig, "Fig6_robustness")


if __name__ == "__main__":
    fig1_domain_routing()
    fig2_main_performance()
    fig3_controls()
    fig4_ablation()
    fig5_external()
    fig6_robustness()
    print(f"Wrote draft figures to {OUT}")
