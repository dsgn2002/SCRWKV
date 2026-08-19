"""P4: Violin + CDF plots of per-image Dice across models and datasets.

Reads per_image_dice.csv from viz_results/{dataset}/{experiment}/,
generates violin plot + overlaid CDF plot.

Usage: python plot_dice_distributions.py
Output: results/p4_plots/
"""
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/p4_plots")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["TUT_test", "TUT_val", "ustCrack"]
EXPERIMENTS = ["T1_baseline", "T2_upfm", "T3_scale_upfm", "E5TUT_full"]
COLORS = {"T1_baseline": "#888888", "T2_upfm": "#2196F3", "T3_scale_upfm": "#4CAF50", "E5TUT_full": "#FF9800"}

# --- Load per-image Dice ---
data = {}  # data[dataset][experiment] = [dice_values...]
for ds in DATASETS:
    data[ds] = {}
    for exp in EXPERIMENTS:
        csv_path = Path(f"viz_results/{ds}/{exp}/per_image_dice.csv")
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                dice_vals = [float(row["dice"]) for row in reader]
                data[ds][exp] = dice_vals
        else:
            print(f"  [skip] {csv_path} not found")
            data[ds][exp] = []

# --- Plot 1: Violin per dataset ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax_idx, ds in enumerate(DATASETS):
    ax = axes[ax_idx]
    positions = []
    violins = []
    for i, exp in enumerate(EXPERIMENTS):
        vals = data[ds][exp]
        if vals:
            positions.append(i + 1)
            vp = ax.violinplot(vals, positions=[i + 1], showmeans=True, showmedians=True)
            for body in vp["bodies"]:
                body.set_facecolor(COLORS[exp])
                body.set_alpha(0.7)
            vp["cmeans"].set_color("black")
            vp["cmedians"].set_color("red")

    ax.set_title(f"{ds} (n={len(data[ds][EXPERIMENTS[0]]) if data[ds][EXPERIMENTS[0]] else '?'})", fontsize=13)
    ax.set_xticks(range(1, len(EXPERIMENTS) + 1))
    ax.set_xticklabels(["T1\nBaseline", "T2\nUPFM", "T3\n+Scale", "E5\nFull"], fontsize=9)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Per-Image Dice Distribution by Dataset", fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "violin_dice.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'violin_dice.png'}")

# --- Plot 2: Overlaid CDF per dataset ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_idx, ds in enumerate(DATASETS):
    ax = axes[ax_idx]
    for exp in EXPERIMENTS:
        vals = data[ds][exp]
        if vals:
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, color=COLORS[exp], linewidth=1.5, label=exp.replace("_", " "))

    ax.set_title(f"{ds}", fontsize=13)
    ax.set_xlabel("Dice Score")
    ax.set_ylabel("CDF")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=7, loc="lower right")

fig.suptitle("Cumulative Distribution of Per-Image Dice", fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "cdf_dice.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'cdf_dice.png'}")

# --- Print zero-Dice summary ---
print("\nZero-Dice Images:")
for ds in DATASETS:
    for exp in EXPERIMENTS:
        vals = data[ds][exp]
        if vals:
            n_zero = sum(1 for v in vals if v < 1e-6)
            if n_zero > 0:
                print(f"  {ds}/{exp}: {n_zero} zero-Dice images ({100*n_zero/len(vals):.1f}%)")

print("\nDone.")
