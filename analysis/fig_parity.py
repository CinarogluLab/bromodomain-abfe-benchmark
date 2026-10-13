"""Main-text Figure 3: calculated vs experimental dG, one panel per water model.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from legs_analysis import load
from pairwise_ddg import CALC, EXP, PDB

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
COLOR = {"TIP3P": "#2a78d6", "TIP4P": "#eb6834", "TIP5P": "#1baf7a", "TIP7P": "#eda100"}
TEXT, TEXT_2 = "#0b0b0b", "#52514e"
BAND_1, BAND_2, GRID = "#dedcd6", "#f0efec", "#e4e3df"


def provisional():
    p = pd.read_csv(HERE / "mbar" / "abfe_legs_provenance.tsv", sep="\t")
    return set(map(tuple, p[p.status != "mbar_reanalysis"][["model", "pdb"]].values))


def main():
    legs, prov = load(), provisional()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6), sharex=True, sharey=True)
    lo, hi = -16.5, -6.0
    x = np.array([lo, hi])
    for ax, model in zip(axes.flat, CALC):
        calc = np.array(CALC[model])
        err = np.array([legs[model, p]["err"] for p in PDB])
        ax.fill_between(x, x - 2, x + 2, color=BAND_2, lw=0, zorder=0)
        ax.fill_between(x, x - 1, x + 1, color=BAND_1, lw=0, zorder=0)
        ax.plot(x, x, color=TEXT, lw=1.0, zorder=1)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.errorbar(EXP, calc, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=0.9, capsize=2, zorder=2)
        is_prov = np.array([(model, p) in prov for p in PDB])
        ax.scatter(EXP[~is_prov], calc[~is_prov], s=46, color=COLOR[model], edgecolors="white", linewidths=1.0, zorder=3)
        ax.scatter(EXP[is_prov], calc[is_prov], s=46, facecolors="white", edgecolors=COLOR[model], linewidths=1.6, zorder=3)
        labelled = [i for i in range(len(PDB)) if abs(calc[i] - EXP[i]) > 2]
        for i in labelled:
            # put the label on the left when another labelled point sits just to the right and below
            crowded = any(j != i and 0 < EXP[j] - EXP[i] < 0.6 and abs(calc[j] - calc[i]) < 0.8 for j in labelled)
            ax.annotate(PDB[i], (EXP[i], calc[i]), xytext=(-6, 2) if crowded else (6, -2), textcoords="offset points",
                        fontsize=7.5, color=TEXT_2, va="bottom" if crowded else "top", ha="right" if crowded else "left")
        e = calc - EXP
        txt = (f"MUE {np.abs(e).mean():.2f}\nRMSE {np.sqrt(np.mean(e**2)):.2f}\n"
               f"$r$ {stats.pearsonr(EXP, calc)[0]:.2f}\n$\\tau$ {stats.kendalltau(EXP, calc)[0]:.2f}")
        ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=TEXT_2,
                linespacing=1.3)
        ax.set_title(model, fontsize=10, fontweight="bold", color=TEXT, loc="left")
        ax.set_xlim(-11.5, -8.0)
        ax.set_xticks([-11, -10, -9])
        ax.set_ylim(lo, hi)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(r"$\Delta G_\mathrm{exp}$ (kcal mol$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta G_\mathrm{calc}$ (kcal mol$^{-1}$)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig4_parity.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", FIGDIR / "fig4_parity.pdf")


if __name__ == "__main__":
    main()
