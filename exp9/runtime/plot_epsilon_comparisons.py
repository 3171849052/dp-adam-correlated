"""Plot existing Exp9 epsilon results without changing frozen experiment code.

Run from the repository root:
    conda run --no-capture-output -n curve python -B -m exp9.runtime.plot_epsilon_comparisons
"""
import csv
import json

import numpy as np

from exp9 import RESULTS
from exp9.config import FINAL_SEEDS, IID, METHODS, MF, WORKLOADS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPSILONS = (2, 4, 8, 16)
TASKS = {"cv": "CV: CIFAR-100", "nlp": "NLP: SST-2"}
WORKLOAD_LABELS = {"sgd": "SGD", "momentum": "Momentum", "momentum-bias": "Momentum-Bias"}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")


def load_points():
    """Check plotted means/std against the same three raw seeds at every epsilon."""
    points = json.loads((RESULTS / "privacy_utility.json").read_text())
    raw = json.loads((RESULTS / "final_results.json").read_text())
    raw += json.loads((RESULTS / "sweep_results.json").read_text())
    expected_keys = {(task, method, eps) for task in TASKS for method in METHODS for eps in EPSILONS}
    lookup = {(p["task"], p["method"], p["epsilon"]): p for p in points}
    assert len(points) == len(lookup) == 56 and set(lookup) == expected_keys
    for (task, method, epsilon), point in lookup.items():
        runs = sorted(
            (r for r in raw if r["task"] == task and r["method"] == method
             and r["epsilon"] == epsilon and r["seed"] in FINAL_SEEDS[:3]),
            key=lambda r: r["seed"],
        )
        assert [r["seed"] for r in runs] == list(FINAL_SEEDS[:3])
        assert all(r["status"] == "completed" for r in runs)
        assert point["n"] == 3
        values = np.asarray([r["accuracy"] for r in runs])
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
        np.testing.assert_allclose(point["mean"], values.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(point["std"], values.std(ddof=1), rtol=0, atol=1e-12)
    return lookup


def draw_curve(ax, points, task, method, label, color, marker, linestyle="-"):
    rows = [points[task, method, epsilon] for epsilon in EPSILONS]
    ax.errorbar(
        EPSILONS, [100 * r["mean"] for r in rows],
        yerr=[100 * r["std"] for r in rows], label=label,
        color=color, marker=marker, linestyle=linestyle, linewidth=2,
        markersize=6, capsize=3.5, elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.6,
    )


def make_figure(points, title, range_methods=METHODS, y_limits=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    for ax, (task, label) in zip(axes, TASKS.items()):
        rows = [row for (t, method, _), row in points.items() if t == task and method in range_methods]
        low = min(100 * (r["mean"] - r["std"]) for r in rows)
        high = max(100 * (r["mean"] + r["std"]) for r in rows)
        # Share task-specific limits across the three Scale comparison figures.
        ax.set_ylim(max(0, 5 * np.floor((low - 2) / 5)), min(100, 5 * np.ceil((high + 2) / 5)))
        if y_limits and task in y_limits:
            lower, upper = y_limits[task]
            assert lower < low and high < upper, f"Error bars outside {task} axis limits"
            ax.set_ylim(lower, upper)
        ax.set_xlim(1.3, 16.7)
        ax.set_xticks(EPSILONS)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("ACC (%)")
        ax.set_title(label, fontsize=12, pad=10)
        ax.grid(axis="y", color="#D8DEE5", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.28, top=0.82, wspace=0.23)
    fig.text(
        0.5, 0.035,
        r"Mean $\pm$ sample std; 3 common seeds; hyperparameters selected at $\epsilon=8$",
        ha="center", fontsize=9, color="#52606D",
    )
    return fig, axes


def save_figure(fig, axes, name, legend_columns):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncol=legend_columns, frameon=False, fontsize=10)
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    points = load_points()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = make_figure(points, "Non-scale methods", range_methods=(IID, *MF),
                            y_limits={"nlp": (71, 81)})
    labels = ("IID", "SGD (BandInvMF)", "Momentum (BandInvMF)", "Momentum-Bias (BandInvMF)")
    for ax, task in zip(axes, TASKS):
        for method, label, color, marker in zip((IID, *MF), labels, COLORS, MARKERS):
            draw_curve(ax, points, task, method, label, color, marker)
    save_figure(fig, axes, "epsilon_acc_non_scale", 4)

    filenames = ["epsilon_acc_non_scale"]
    for workload, method in zip(WORKLOADS, MF):
        limits = {"cv": (38, 77), "nlp": (53, 81)} if workload == "sgd" else None
        fig, axes = make_figure(points, f"{WORKLOAD_LABELS[workload]} workload (BandInvMF): Scale vs. Non-scale",
                                range_methods=(method, method + "-scale") if limits else METHODS,
                                y_limits=limits)
        for ax, task in zip(axes, TASKS):
            draw_curve(ax, points, task, method, "Non-scale", COLORS[0], "o", "--")
            draw_curve(ax, points, task, method + "-scale", "Scale", "#D55E00", "s")
        name = f"epsilon_acc_scale_vs_non_scale_{workload.replace('-', '_')}"
        save_figure(fig, axes, name, 2)
        filenames.append(name)

    # Export exactly the displayed values in percent for easy reuse.
    with (RESULTS / "figures/epsilon_acc_plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "method", "epsilon", "n", "acc_mean_percent", "acc_std_percent"))
        writer.writeheader()
        for task in TASKS:
            for method in METHODS:
                for epsilon in EPSILONS:
                    row = points[task, method, epsilon]
                    writer.writerow(dict(task=task, method=method, epsilon=epsilon, n=row["n"],
                                         acc_mean_percent=100 * row["mean"], acc_std_percent=100 * row["std"]))

    lines = [
        "# Exp9 epsilon–ACC comparisons", "",
        "CV: CIFAR-100 official test; NLP: SST-2 official validation. ACC is in percent.",
        "Epsilon values: 2, 4, 8, 16. Every point uses seeds 20261111, 20261112, 20261113,",
        "including epsilon=8. Error bars show sample standard deviation (ddof=1).",
        "LR/C/eps_scale stay fixed at the configuration selected at epsilon=8.", "",
        "Source: ../privacy_utility.json; all 56 points checked against ../final_results.json and ../sweep_results.json.",
        "Displayed values: [CSV](epsilon_acc_plot_data.csv).", "",
        "Regenerate from the repository root:", "```bash",
        "conda run --no-capture-output -n curve python -B -m exp9.runtime.plot_epsilon_comparisons", "```", "",
    ]
    for name in filenames:
        lines.extend([f"![{name}]({name}.png)", "", f"[PNG]({name}.png) · [PDF]({name}.pdf)", ""])
    (RESULTS / "figures/epsilon_acc_comparisons.md").write_text("\n".join(lines))
    print("Validated 56 epsilon/method/task points against 3 common raw seeds each.")
    for name in filenames:
        print(f"Saved {RESULTS / 'figures' / name}.png / .pdf")


if __name__ == "__main__":
    main()
