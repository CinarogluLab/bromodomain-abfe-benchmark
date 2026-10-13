"""Main-text Figure 4: signed error per complex and water model, complexes ordered by experimental TdS (enthalpy-driven left, entropy-driven right).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig_parity import COLOR, GRID, TEXT, TEXT_2, provisional
from legs_analysis import load
from metrics import TDS
from pairwise_ddg import BRD, CALC, EXP, PDB

FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def main():
    legs, prov = load(), provisional()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.linewidth": 0.6, "hatch.color": "white", "hatch.linewidth": 1.2,
    })
    order = np.argsort(TDS)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    n = len(CALC)
    width = 0.8 / n
    for k, model in enumerate(CALC):
        calc = np.array(CALC[model])
        xs = np.arange(len(PDB)) + (k - (n - 1) / 2) * width
        for j, i in enumerate(order):
            p = PDB[i]
            ax.bar(xs[j], calc[i] - EXP[i], width=width, color=COLOR[model], edgecolor="white", linewidth=1.0,
                   hatch="////" if (model, p) in prov else None, zorder=2)
            ax.errorbar(xs[j], calc[i] - EXP[i], yerr=legs[model, p]["err"], fmt="none", ecolor=TEXT_2,
                        elinewidth=0.7, capsize=1.5, zorder=3)
    ax.axhline(0, color=TEXT, lw=0.8, zorder=1)
    for y in (-2, 2):
        ax.axhline(y, color=TEXT_2, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(len(PDB)))
    ax.set_xticklabels([f"{PDB[i]}\n{BRD[i]}\n{TDS[i]:+.1f}" for i in order], fontsize=7.5)
    ax.set_ylabel(r"$\Delta G_\mathrm{calc} - \Delta G_\mathrm{exp}$ (kcal mol$^{-1}$)")
    ax.text(0.0, -0.30, r"experimental $T\Delta S$ (kcal mol$^{-1}$):  enthalpy-driven $\leftarrow$   $\rightarrow$ entropy-driven",
            transform=ax.transAxes, fontsize=7.5, color=TEXT_2, va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[m], edgecolor="white", label=m) for m in CALC]
    if prov:
        handles.append(Patch(facecolor=TEXT_2, edgecolor="white", hatch="////", label="provisional"))
    ax.legend(handles=handles, ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, handlelength=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig5_signed_error.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", FIGDIR / "fig5_signed_error.pdf")


if __name__ == "__main__":
    main()
