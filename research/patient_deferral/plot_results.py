"""Generate the manuscript figure from aggregate results only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
BACKBONES = ("logistic", "gradient_boosting")
LABELS = ("Logistic", "Gradient boosting")
COLORS = ("#215b9c", "#a04d13")


def read_aggregate(filename):
    return json.loads((HERE / filename).read_text())


def plot_ranking(ax):
    for index, (filename, representation) in enumerate((
        ("results.json", "288 radiomic features"), ("public_results.json", "12 public descriptors"),
    )):
        results = read_aggregate(filename)
        for row, backbone in enumerate(BACKBONES):
            estimate = results[backbone]
            delta = 100 * estimate["mean_delta"]
            low, high = [100 * value for value in estimate["paired_percentile_interval"]]
            ax.errorbar(delta, 3.3 - index * 2.0 - row * .65,
                        xerr=[[delta - low], [high - delta]], fmt="o", capsize=3,
                        color=COLORS[row], lw=1.5, ms=4, label=LABELS[row] if index == 0 else None)
        ax.text(-.30, 3.65 - index * 2.0, representation, va="bottom", fontsize=8.5)
    ax.axvline(0, color="#606d79", linestyle="--", lw=1)
    ax.set_xlim(-.31, .06)
    ax.set_ylim(.35, 4.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-.30, -.20, -.10, 0])
    ax.set_xlabel("Change in AUGRC (percentage points)")
    ax.set_title("A. Does disagreement improve error ranking?", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(.5, -.58), frameon=False, fontsize=8)


def plot_retention(ax):
    results = read_aggregate("extension_results.json")["threshold_replay"]
    for row, backbone in enumerate(BACKBONES):
        result = results[backbone]["pooled"]["learned_margin"]
        malignant = 100 * result["accepted_malignant"] / result["n_malignant"]
        benign = 100 * (result["accepted"] - result["accepted_malignant"]) / (result["n"] - result["n_malignant"])
        ax.plot([malignant, benign], [1 - row] * 2, color="#b2bac3", lw=2)
        for color, marker, value, label in zip(
            COLORS, ("o", "s"), (malignant, benign), ("Malignant", "Benign"),
        ):
            ax.plot(value, 1 - row, marker=marker, color=color, ms=5, ls="none",
                    label=label if row == 0 else None)
            ax.text(value, 1.13 - row, f"{value:.1f}%", color=color, va="bottom", ha="center", fontsize=8)
    ax.axvline(90, color="#606d79", linestyle="--", lw=1)
    ax.set_yticks([1, 0], LABELS)
    ax.set_xlim(79, 97)
    ax.set_ylim(-.38, 1.52)
    ax.set_xticks([80, 85, 90, 95])
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Retained patients in each diagnosis group")
    ax.set_title("B. Who is retained at the frozen cutoff?", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(.5, -.58), frameon=False, fontsize=8)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0))
    plot_ranking(axes[0])
    plot_retention(axes[1])
    fig.subplots_adjust(left=.22, right=.98, top=.93, bottom=.18, hspace=1.03)
    fig.savefig(HERE / "manuscript/results.pdf", metadata={"CreationDate": None, "ModDate": None})


if __name__ == "__main__":
    main()
