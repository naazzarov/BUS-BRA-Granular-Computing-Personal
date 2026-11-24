from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
r = json.loads((here / "results.json").read_text())
labels = ["Logistic", "Gradient boosting"]
names = ["logistic", "gradient_boosting"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, right) = plt.subplots(1, 2, figsize=(9.9, 2.7),
                                     gridspec_kw={"width_ratios":[1, 1.1]})
y = [1, 0]
for i, name in enumerate(names):
    d = r[name]
    m = 100 * d["mean_delta"]
    low, high = (100 * n for n in d["paired_percentile_interval"])
    ax.errorbar(m, y[i], xerr=[[m-low], [high-m]], fmt="o", capsize=4,
                markersize=5, color="#006d72", lw=1.7)
ax.axvline(0, color="#64748b", linestyle="--", lw=1)
ax.set_yticks(y, labels)
ax.set_ylim(-.57, 1.57)
ax.set_xticks([-.15, -.10, -.05, 0, .05], ["−0.15", "−0.10", "−0.05", "0", "0.05"])
ax.set_xlabel("Difference in AUGRC (percentage points)")
ax.set_title("A. Incremental value of disagreement", loc="left", fontweight="bold")
ax.grid(axis="x", alpha=.22)

for i, name in enumerate(names):
    p = r[name]["rules"]["learned_margin"]["operating_points"]["0.9"]
    rates = 100 * np.array([p["accepted_malignant"]/p["n_malignant"],
        (p["accepted"]-p["accepted_malignant"])/(p["n_patients"]-p["n_malignant"])])
    for j, value in enumerate(rates):
        ypos = y[i] + (.15 if j == 0 else -.15)
        right.barh(ypos, value, .22, color=["#376ab3","#08796d"][j],
                   label=["Malignant", "Benign"][j] if i == 0 else None)
        right.text(value-1.3, ypos, f"{value:.1f}", ha="right", va="center",
                   color="white", fontsize=8, fontweight="bold")
right.set_yticks(y, labels)
right.set_ylim(-.57, 1.57)
right.set_xlim(0, 100)
right.set_xlabel("Accepted patients by diagnosis (%)")
right.set_title("B. Retention at nominal 90% coverage", loc="left", fontweight="bold")
right.legend(loc="lower left", bbox_to_anchor=(0, -.55), frameon=False, ncol=2)
fig.subplots_adjust(left=.14, right=.98, top=.84, bottom=.27, wspace=.55)
fig.savefig(here/"manuscript/results.png", dpi=240, facecolor="white", bbox_inches="tight")
