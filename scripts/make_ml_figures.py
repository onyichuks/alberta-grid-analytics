"""Figures for docs/ML_TECHNICAL_NOTE.md (figures/fig05 to fig08), drawn from outputs/ml_results.json.

Usage:  python scripts/make_ml_figures.py      (after run_ml_analysis.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
BLUE, ORANGE, AMBER, GREY, INK, INK2, LINE = "#2a78d6", "#eb6834", "#eda100", "#8a96a3", "#0f1a24", "#475665", "#d6dee5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def title(ax, text):
    ax.set_title(text, fontsize=10, loc="left", color=INK)


def curve_panel(ax, tc, used, unit, label, capped):
    it = np.array(tc["iterations"])
    q = tc["q50"]
    ax.plot(it, 2 * np.array(q["train"]), color=GREY, lw=1.6, label="Training years")
    ax.plot(it, 2 * np.array(q["validation"]), color=BLUE, lw=2, label="Validation year")
    ax.axvline(used, color=INK2, lw=1, ls="--")
    note = f"iterations used: {used} (the cap)" if capped else f"iterations used: {used} (early stopping)"
    ax.annotate(note, (used, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
    b = q["best_iteration"]
    ax.plot([b], [2 * q["best_loss"]], "o", ms=7, color=BLUE, mec="white", mew=2)
    right = b > 0.7 * it.max()  # keep the label inside the plot
    ax.annotate(f"lowest validation error at {b}", (b, 2 * q["best_loss"]), xytext=(-6 if right else 6, 8),
                textcoords="offset points", ha="right" if right else "left", fontsize=8, color=INK2)
    ax.set_xlabel("Boosting iterations")
    ax.set_ylabel(f"Mean absolute error of the median ({unit})")
    title(ax, label)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    R = json.loads((ROOT / "outputs" / "ml_results.json").read_text())
    L, P = R["load"], R["price"]

    # 5. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lu, pu = L["production_iterations_2025"]["q50"], P["production_iterations_2025"]["q50"]
    curve_panel(axes[0], L["training_curve"], lu, "MW", "Load: train 2017 to 2023, validate 2024", lu >= 400)
    curve_panel(axes[1], P["training_curve"], pu, "$/MWh", "Price: train 2019 to 2023, validate 2024", pu >= 400)
    axes[0].legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_ml_training_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Learning curve: years of history
    lc = L["learning_curve"]
    sc = {(r["model"], r["period"]): r for r in L["scores"]}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot([r["train_years"] for r in lc], [r["mape_pct"] for r in lc], "-o", color=BLUE, lw=2, ms=6,
            mec="white", mew=1.5, label="Gradient boosting")
    lin = sc[("Linear quantile regression, conformal", "all test")]["mape_pct"]
    nav = sc[("Seasonal naive (same hour last week)", "all test")]["mape_pct"]
    ax.axhline(lin, color=ORANGE, lw=1.6, ls="--", label=f"Linear quantile regression, 7 years ({lin:.2f}%)")
    ax.axhline(nav, color=GREY, lw=1.6, ls=":", label=f"Same hour last week ({nav:.2f}%)")
    for r in lc:
        above = r["mape_pct"] > lin  # keep each label clear of the linear reference line
        ax.annotate(f"{r['mape_pct']:.2f}%", (r["train_years"], r["mape_pct"]), xytext=(0, 8 if above else -14),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Years of training history (each test year, 2023 to 2025)")
    ax.set_ylabel("Mean absolute percentage error (%)")
    ax.set_ylim(0, nav * 1.15)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    title(ax, "Load forecast error by amount of history")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_ml_learning_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 7. Load errors by hour and month
    colors = {"Seasonal naive": GREY, "Linear quantile regression": ORANGE, "Gradient boosting": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, xl in ((axes[0], "hour", "Hour of day (Alberta time)"), (axes[1], "month", "Month")):
        rows = L["segments"][key]
        for m, c in colors.items():
            pts = [(r[key], r["mape_pct"]) for r in rows if r["model"] == m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, lw=1.8, ms=4, label=m)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Mean absolute percentage error (%)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(list("JFMAMJJASOND"))
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    title(axes[0], "Load forecast error by hour, 2023 to July 2025")
    title(axes[1], "By month")
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_ml_load_errors.png", bbox_inches="tight")
    plt.close(fig)

    # 8. Price errors by price level, and spike warning reliability
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    rows = P["segments"]["price_band"]
    bands = list(dict.fromkeys(r["price_band"] for r in rows))
    pc = {"AESO forecast": AMBER, "Linear quantile regression": ORANGE, "Gradient boosting": BLUE}
    w = 0.26
    for i, (m, c) in enumerate(pc.items()):
        vals = [next(r["mae"] for r in rows if r["model"] == m and r["price_band"] == b) for b in bands]
        axes[0].bar(np.arange(len(bands)) + (i - 1) * w, vals, w * 0.92, color=c, label=m)
    axes[0].set_xticks(range(len(bands)))
    axes[0].set_xticklabels([b.replace(" to ", "\nto ").replace(" and above", "\nand above").replace("below ", "below\n")
                             for b in bands], fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean absolute error ($/MWh, log scale)")
    axes[0].set_xlabel("Actual pool price")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    title(axes[0], "Price forecast error by price level, 2024 to July 2025")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=LINE, lw=1.2)
    for m, c in (("Gradient boosting", BLUE), ("Logistic regression", ORANGE)):
        rel = P["spike"]["models"][m]["reliability"]
        ax.plot([r["forecast"] for r in rel], [r["observed"] for r in rel], "-o", color=c, lw=1.8, ms=6,
                mec="white", mew=1.5, label=f"{m} (Brier skill {P['spike']['models'][m]['brier_skill_pct']:.0f}%)")
    ax.set_xlabel("Forecast probability of a $200+ hour")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    title(ax, "Spike warning reliability")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_ml_price_errors.png", bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
