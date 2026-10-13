"""Table S7: accuracy of each water model when the legs are evaluated with different estimators.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pairwise_ddg import CALC, EXP, PDB

HERE = Path(__file__).resolve().parent
EST = [("MBAR, all frames (main text)", "dG_mbar_all"), ("MBAR, decorrelated frames", "dG"), ("TI", "dG_ti")]
LEGS = ["coulomb", "vdw", "restr", "ligand"]

legs = pd.read_csv(HERE / "mbar" / "mbar_legs.tsv", sep="\t").dropna(subset=["dG_mbar_all"]).set_index(["model", "pdb", "leg"])
table = pd.read_csv(HERE / "abfe_legs.tsv", sep="\t").set_index(["model", "pdb"])
num = lambda x: f"${x:.2f}$"
rows = []
for model in CALC:
    rows.append(r"\multicolumn{6}{l}{\textit{" + model + r"}} \\")
    for label, col in EST:
        calc = []
        for p in PDB:
            v = {l: (legs.loc[(model, p, l), col] if (model, p, l) in legs.index else table.loc[(model, p), l]) for l in LEGS}
            calc.append(v["ligand"] - (v["coulomb"] + v["vdw"] + v["restr"]) - table.loc[(model, p), "dG_off"])
        calc = np.array(calc); err = calc - EXP
        r = stats.pearsonr(EXP, calc)[0]; tau = stats.kendalltau(EXP, calc)[0]
        rows.append(f"{label} & {num(np.abs(err).mean())} & {num(np.sqrt(np.mean(err**2)))} & {num(err.mean())} & {num(r)} & {num(tau)} \\\\")
(HERE.parent / "tableS7_rows.tex").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
