"""Optional static result figure; requires matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    directory = Path(directory)
    result = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    order = ["baseline/none", "history/empty", "history/correct", "history/wrong", "history/shuffled", "history/online"]
    labels = ["Baseline", "Empty", "Correct", "Wrong body", "Shuffled", "Online"]
    rows = {row["condition"]: row for row in result["conditions"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = ["#56748c", "#9ba8b4", "#15827b", "#c37851", "#b99b69", "#438fba"]
    x = np.arange(len(order))
    axes[0].bar(x, [100*rows[key]["success_rate"] for key in order], color=colors, alpha=.8)
    for index, key in enumerate(order):
        rates = np.array(rows[key]["seed_success_rates"]) * 100
        axes[0].scatter(index+np.linspace(-.12, .12, len(rates)), rates,
                        color="#17252f", s=22, zorder=3)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 105), ylabel="Validation success (%)",
                title="Means and individual training seeds")
    axes[0].tick_params(axis="x", labelrotation=25)
    contrasts = result["paired_success_contrasts"]
    chosen = ["history/correct - baseline/none", "history/correct - history/wrong",
              "history/correct - history/empty", "history/online - history/correct"]
    names = ["Correct − baseline", "Correct − wrong", "Correct − empty", "Online − frozen"]
    for index, key in enumerate(chosen):
        value = contrasts[key]
        center, low, high = [100*value[k] for k in ["difference", "low", "high"]]
        axes[1].plot([low, high], [index, index], color="#56748c", linewidth=2)
        axes[1].scatter([center], [index], color="#15827b", zorder=3)
    axes[1].axvline(0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set(yticks=range(4), yticklabels=names, xlabel="Success difference (percentage points)",
                title="Paired effects · exploratory 95% intervals")
    axes[1].invert_yaxis()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"EASE-FM development validation · {protocol['steps']:,} training steps", fontsize=14)
    n_seeds = len(rows[order[0]]["seed_success_rates"])
    n_bodies = len(rows[order[0]]["body_success_rates"])
    n_goals = rows[order[0]]["trials"] // (n_seeds * n_bodies)
    fig.text(.5, .015, f"{n_seeds} training seeds × {n_bodies} validation bodies × {n_goals} goals; final test split unused. "
             "Intervals resample seeds, bodies and goals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(directory / "validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    plot(parser.parse_args().directory)
