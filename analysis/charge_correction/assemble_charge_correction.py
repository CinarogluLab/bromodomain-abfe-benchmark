"""Assemble Rocklin-corrected binding free energies for the four +1 ligands (BOARD T4).

Reads the v2 leg table (rocklin_charge_correction_v2.py) and the final uncorrected CALC.
For each charged complex and water model:
  dDG_bind = cor(complex) - cor(solvent)    (binding = charging(complex) - charging(solvent))
Primary convention: published counter-ion approximation (Q_P = 0); sensitivity: actual complex
environment charge (Q_env = -1). Terms reported separately (NET+USV, RIP, EMP, DSC).
A full corrected affinity is formed only when every term exists; TIP7P lacks DSC, so its
dG_corrected is left empty and a distinctly named analytical-only estimate is given instead.
`config_spread` is the frame-to-frame SD of the two leg totals combined in quadrature: it describes
configurational variation over 11 snapshots, not the error of the mean or total model uncertainty.
Writes charge_correction/corrected_dg_v2_<variant>.tsv and prints aggregate metrics.
Usage: python assemble_charge_correction.py [legs.tsv] [variant]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pairwise_ddg import CALC, EXP, PDB  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rocklin_legs_v2_d257_main.tsv"
variant = sys.argv[2] if len(sys.argv) > 2 else "main"
legs = pd.read_csv(path, sep="\t")
# Approximate TIP7P discrete-solvent term: O-centred quadrupole trace of the run geometry (Codex's
# gamma, charge_terms_independent / CHARGE_ASSESSMENT). TIP7P also has weak LJ sites on H, so the
# exact Rocklin treatment needs an effective quadrupole (sec V B4) that would require new sampling;
# this value is a labelled approximation and never enters dG_corrected.
GAMMA_O_TIP7P = 0.00793511          # e nm^2
K = 138.93545585                    # kJ nm / (mol e^2)
def dsc_approx(row):
    return -(2 * np.pi / 3) * K * GAMMA_O_TIP7P * row["qL"] * row["N_S"] / row["md_volume_nm3"] / 4.184
rows = []
for (model, pdb), g in legs.groupby(["model", "pdb"]):
    c = g[g.leg == "complex"].iloc[0]; s = g[g.leg == "solvent"].iloc[0]
    d = lambda k: c[k] - s[k]
    row = dict(model=model, pdb=pdb, NET_USV=d("NET_USV"), RIP=d("RIP"), EMP=d("EMP"),
               DSC=d("DSC") if pd.notna(c["DSC"]) and pd.notna(s["DSC"]) else np.nan,
               analytical_only=d("ANA"), sens_analytical=d("sens_ANA"))
    row["config_spread"] = float(np.hypot(c["total_sd"], s["total_sd"])) if pd.notna(row["DSC"]) \
        else float(np.hypot(c["ANA_sd"], s["ANA_sd"]))
    full = pd.notna(row["DSC"])
    row["correction"] = row["analytical_only"] + row["DSC"] if full else np.nan
    row["sens_correction"] = row["sens_analytical"] + row["DSC"] if full else np.nan
    row["dG_uncorrected"] = CALC[model][PDB.index(pdb)]
    row["dG_corrected"] = row["dG_uncorrected"] + row["correction"] if full else np.nan
    row["dG_analytical_only"] = row["dG_uncorrected"] + row["analytical_only"]
    if model == "TIP7P":
        row["DSC_O_approx"] = dsc_approx(c) - dsc_approx(s)
        row["correction_O_approx"] = row["analytical_only"] + row["DSC_O_approx"]
        row["dG_O_approx"] = row["dG_uncorrected"] + row["correction_O_approx"]
    row["exp"] = EXP[PDB.index(pdb)]
    rows.append(row)
cor = pd.DataFrame(rows)
out = HERE / f"corrected_dg_v2_{variant}.tsv"
cor.to_csv(out, sep="\t", index=False)
with pd.option_context("display.width", 220):
    print(cor.round(3).to_string(index=False))


def metrics(calc):
    e = calc - EXP
    return (np.abs(e).mean(), np.sqrt((e ** 2).mean()), e.mean(), stats.pearsonr(EXP, calc)[0],
            stats.kendalltau(EXP, calc)[0])


print("\nten complexes, uncorrected -> corrected (MUE RMSE bias r tau)")
for model in CALC:
    sub = cor[cor.model == model]
    if sub.dG_corrected.isna().any():
        print(f"{model}: no full correction (DSC missing); analytical-only shifts "
              + ", ".join(f"{r.pdb} {r.analytical_only:+.2f}" for _, r in sub.iterrows()))
        continue
    unc = np.array(CALC[model], float); corr = unc.copy()
    for _, r in sub.iterrows():
        corr[PDB.index(r.pdb)] = r.dG_corrected
    a, b = metrics(unc), metrics(corr)
    print(f"{model}: " + "  ".join(f"{x:.2f}->{y:.2f}" for x, y in zip(a, b)))
