"""Leg-by-leg decomposition of the ABFE results (abfe_legs.tsv).
"""
import csv
from pathlib import Path

import numpy as np

from pairwise_ddg import BRD, CALC, EXP, PDB

HERE = Path(__file__).resolve().parent
MODELS = list(CALC)


def load():
    data = {}
    with open(HERE / "abfe_legs.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v = {k: float(x) for k, x in r.items() if k not in ("model", "pdb")}
            v["complex"] = v["coulomb"] + v["vdw"] + v["restr"]
            v["dG"] = v["ligand"] - v["complex"] - v["dG_off"]
            v["err"] = np.sqrt(sum(v[f"{k}_err"] ** 2 for k in ("coulomb", "vdw", "restr", "ligand")))
            data[r["model"], r["pdb"]] = v
    return data


def main():
    d = load()

    print("== consistency with main-text Table 3 ")
    diffs = [(m, p, d[m, p]["dG"] - CALC[m][i]) for m in MODELS for i, p in enumerate(PDB)]
    worst = max(diffs, key=lambda t: abs(t[2]))
    print(f"max |recomputed - Table 3| = {abs(worst[2]):.3f} kcal/mol ({worst[0]} {worst[1]})")
    errs = np.array([d[m, p]["err"] for m in MODELS for p in PDB])
    print(f"propagated uncertainty: median {np.median(errs):.2f}, max {errs.max():.2f} kcal/mol "
          f"({max(((m, p) for m in MODELS for p in PDB), key=lambda k: d[k]['err'])})")

    print("\n== where do model-to-model differences come from? (each model minus TIP3P, 10 complexes)")
    print(f"{'model':6s} {'rms dBind':>9s} {'rms dSolv':>9s} {'rms dCplx':>9s} {'rms dOff':>8s} {'r(dSolv,dCplx)':>15s}")
    for m in MODELS[1:]:
        db = np.array([d[m, p]["dG"] - d["TIP3P", p]["dG"] for p in PDB])
        ds = np.array([d[m, p]["ligand"] - d["TIP3P", p]["ligand"] for p in PDB])
        dc = np.array([d[m, p]["complex"] - d["TIP3P", p]["complex"] for p in PDB])
        do = np.array([d[m, p]["dG_off"] - d["TIP3P", p]["dG_off"] for p in PDB])
        rms = lambda a: np.sqrt(np.mean(a**2))
        print(f"{m:6s} {rms(db):9.2f} {rms(ds):9.2f} {rms(dc):9.2f} {rms(do):8.2f} {np.corrcoef(ds, dc)[0, 1]:15.2f}")
        for i, p in enumerate(PDB):
            if abs(db[i]) > 2:
                print(f"    {p}: dBind {db[i]:+.2f} = dSolv {ds[i]:+.2f} - dCplx {dc[i]:+.2f} - dOff {do[i]:+.2f}  "
                      f"[dCoul {d[m, p]['coulomb'] - d['TIP3P', p]['coulomb']:+.2f}, "
                      f"dVdw {d[m, p]['vdw'] - d['TIP3P', p]['vdw']:+.2f}, dRestr {d[m, p]['restr'] - d['TIP3P', p]['restr']:+.2f}]")

    print("\n== analytical restraint term dG_off (all 40)")
    off = np.array([d[m, p]["dG_off"] for m in MODELS for p in PDB])
    print(f"range {off.min():.2f} to {off.max():.2f}, mean {off.mean():.2f}")

    num = lambda x: f"${x:.2f}$"   # math mode gives a true minus sign
    rows = []
    for m in MODELS:
        rows.append(r"\multicolumn{8}{l}{\textit{" + m + r"}} \\")
        for i, p in enumerate(PDB):
            v = d[m, p]
            cells = [p, BRD[i]] + [num(v[k]) for k in ("coulomb", "vdw", "restr", "ligand", "dG_off")]
            cells.append(f"${v['dG']:.2f} \\pm {v['err']:.2f}$")
            rows.append(" & ".join(cells) + r" \\")
    (HERE.parent / "tableS6_rows.tex").write_text("\n".join(rows) + "\n")
    print("\nwrote tableS6_rows.tex")


if __name__ == "__main__":
    main()
