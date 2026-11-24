"""Generate the manuscript figure from aggregate results only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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


def plot_partitions():
    fig, ax = plt.subplots(figsize=(6.65, 2.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    rows = (
        ("A. Five-fold ranking comparison", (
            ("Outer training", "Cross-fit image scores\nFit error rankers"),
            ("Outer training", "Refit the image model\non the entire partition"),
            ("Outer test", "Rank held-out errors\nMeasure AUGRC"),
        )),
        ("B. Replay with a frozen cutoff", (
            ("Development", "Cross-fit and fit rankers\nRefit the image model"),
            ("Calibration", "Score paired patients\nSet the cutoff"),
            ("Outer test", "Apply the fixed cutoff\nMeasure realized coverage"),
        )),
    )
    for (title, steps), y in zip(rows, (.55, .05)):
        ax.text(.014, y + .405, title, va="top", fontsize=9, fontweight="bold", color="#163148")
        for i, (heading, detail) in enumerate(steps):
            x = .012 + i * .338
            box = FancyBboxPatch((x, y), .298, .286, boxstyle="round,pad=.008",
                                 facecolor=("#edf3fa", "#f5f7f8", "#fcf3eb")[i],
                                 edgecolor="#a8b4bf", linewidth=.7)
            ax.add_patch(box)
            ax.text(x + .149, y + .216, heading, ha="center", va="center", fontweight="bold")
            ax.text(x + .149, y + .100, detail, ha="center", va="center", linespacing=1.15)
            if i < 2:
                ax.annotate("", xy=(x + .329, y + .14), xytext=(x + .305, y + .14),
                            arrowprops={"arrowstyle": "->", "color": "#526473", "lw": .9})
    fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.01)
    fig.savefig(HERE / "manuscript/partitions.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0))
    plot_ranking(axes[0])
    plot_retention(axes[1])
    fig.subplots_adjust(left=.22, right=.98, top=.93, bottom=.18, hspace=1.03)
    fig.savefig(HERE / "manuscript/results.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    plot_partitions()


if __name__ == "__main__":
    main()
