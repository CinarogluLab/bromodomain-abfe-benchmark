"""Figure S1: calculated vs experimental dG per water model, points colored by experimental TdS,
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from legs_analysis import load
from pairwise_ddg import CALC, EXP, PDB

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
DH = np.array([-8.42, -12.11, -13.28, -7.00, -12.48, -5.51, -7.81, -17.00, -3.66, -4.80])
TDS = DH - EXP  # T*dS = dH - dG

# Diverging: blue arm (enthalpy-driven) <- neutral gray -> red arm (entropy-driven)
CMAP = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a097", "#b02f2e"])
TEXT, TEXT_2, RING = "#0b0b0b", "#52514e", "#52514e"
BAND_1, BAND_2, GRID = "#dedcd6", "#f0efec", "#e4e3df"


def main():
    legs = load()
    lim = max(abs(TDS.min()), abs(TDS.max()))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
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
        sc = ax.scatter(EXP, calc, c=TDS, cmap=CMAP, norm=norm, s=46, edgecolors=RING, linewidths=0.7, zorder=3)
        for i, p in enumerate(PDB):
            if abs(calc[i] - EXP[i]) > 2:
                ax.annotate(p, (EXP[i], calc[i]), xytext=(5, -2), textcoords="offset points", fontsize=7.5,
                            color=TEXT_2, va="top")
        ax.set_title(model, fontsize=10, fontweight="bold", color=TEXT, loc="left")
        ax.set_xlim(-11.5, -8.0)
        ax.set_ylim(lo, hi)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel(r"$\Delta G_\mathrm{exp}$ (kcal mol$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta G_\mathrm{calc}$ (kcal mol$^{-1}$)")

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes([0.9, 0.2, 0.022, 0.6])
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label(r"experimental $T\Delta S$ (kcal mol$^{-1}$)", color=TEXT)
    cax.text(0.5, 1.03, "entropy-\ndriven", transform=cax.transAxes, ha="center", va="bottom", fontsize=7.5,
             color=TEXT_2)
    cax.text(0.5, -0.03, "enthalpy-\ndriven", transform=cax.transAxes, ha="center", va="top", fontsize=7.5,
             color=TEXT_2)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"figS1_parity_tds.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", FIGDIR / "figS1_parity_tds.pdf")


if __name__ == "__main__":
    main()
