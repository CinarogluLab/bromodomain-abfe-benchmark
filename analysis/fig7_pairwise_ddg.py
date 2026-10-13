"""Main-text Figure 5: calculated vs experimental pairwise ddG for the 45 complex pairs, one panel per water model.
"""
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pairwise_ddg import BRD, CALC, EXP, PDB, summarize

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
OUTLIER = 3  # 5I8G / CREBBP

# Reference palette (dataviz skill), validated all-pairs on white: CVD dE 24.7, normal dE 33.6.
SERIES = "#2a78d6"      # categorical slot 1: pairs not involving 5I8G
OUTLIER_C = "#eb6834"   # categorical slot 2: pairs involving 5I8G
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND_1 = "#f0efec"      # +/-1.0 kcal/mol
BAND_05 = "#dedcd6"     # +/-0.5 kcal/mol


def oriented_pairs():
    """Rows of (pdb_i, pdb_j, ddG_exp, {model: ddG_calc}) with ddG_exp >= 0 (i binds more weakly than j)."""
    rows = []
    for i, j in itertools.combinations(range(len(PDB)), 2):
        a, b = (i, j) if EXP[i] - EXP[j] >= 0 else (j, i)
        rows.append((a, b, EXP[a] - EXP[b], {m: v[a] - v[b] for m, v in CALC.items()}))
    return rows


def write_csv(rows):
    with open(HERE / "pairwise_ddg.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["complex_A", "bromodomain_A", "complex_B", "bromodomain_B", "ddG_exp"]
                   + [f"ddG_{m}" for m in CALC])
        for a, b, dexp, dcalc in rows:
            w.writerow([PDB[a], BRD[a], PDB[b], BRD[b], f"{dexp:.2f}"] + [f"{dcalc[m]:.2f}" for m in CALC])


def main():
    rows = oriented_pairs()
    write_csv(rows)
    _, stats_all = summarize(set(range(len(PDB))))

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
    lo, hi = -5.0, 9.0
    x = np.array([lo, hi])

    for ax, model in zip(axes.flat, CALC):
        ax.fill_between(x, x - 1.0, x + 1.0, color=BAND_1, lw=0, zorder=0)
        ax.fill_between(x, x - 0.5, x + 0.5, color=BAND_05, lw=0, zorder=0)
        ax.plot(x, x, color=TEXT, lw=1.0, zorder=1)
        ax.axhline(0, color=TEXT_2, lw=0.8, zorder=1)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

        for involves_outlier, color, marker in [(False, SERIES, "o"), (True, OUTLIER_C, "^")]:
            sel = [r for r in rows if (OUTLIER in (r[0], r[1])) == involves_outlier]
            ax.scatter([r[2] for r in sel], [r[3][model] for r in sel], s=34, marker=marker, c=color,
                       edgecolors="white", linewidths=0.8, zorder=3)

        s = stats_all[model]
        ax.text(0.03, 0.97,
                f"correct order {s['sign']:.0%}\nMUE {s['mue']:.2f}   RMSE {s['rmse']:.2f}\n$r$ = {s['r']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color=TEXT, linespacing=1.4)
        ax.text(2.55, lo + 0.3, "wrong order", ha="right", va="bottom", fontsize=7.5, color=TEXT_2, style="italic")
        ax.set_title(model, fontsize=10, fontweight="bold", color=TEXT, loc="left")
        ax.set_xlim(-0.1, 2.6)
        ax.set_ylim(lo, hi)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel(r"$\Delta\Delta G_\mathrm{exp}$ (kcal mol$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta\Delta G_\mathrm{calc}$ (kcal mol$^{-1}$)")

    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=6, mfc=SERIES, mec="white", label="pairs without 5I8G (36)"),
        plt.Line2D([], [], ls="", marker="^", ms=6.5, mfc=OUTLIER_C, mec="white", label="pairs with 5I8G/CREBBP (9)"),
        plt.Rectangle((0, 0), 1, 1, fc=BAND_05, label=r"$\pm$0.5 kcal mol$^{-1}$"),
        plt.Rectangle((0, 0), 1, 1, fc=BAND_1, label=r"$\pm$1.0 kcal mol$^{-1}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.005), handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig7_pairwise_ddg.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", FIGDIR / "fig7_pairwise_ddg.pdf")


if __name__ == "__main__":
    main()
