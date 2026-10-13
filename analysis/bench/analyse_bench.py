"""Cost per ABFE estimate from the same-hardware benchmark (run_bench.sh).

GPU-days per ABFE = sum over legs of n_windows x 10.3 ns / (ns/day), with the window counts of
the protocol (restraint 12, Coulomb 11, vdW 21 in the complex; 31 in solvent) and 10.3 ns per
window (10 ns production + 0.3 ns NVT/NPT equilibration; minimization is negligible). The complex
legs use each complex's own speed; the solvent leg uses the 7BL8 ligand box for every complex
(the only one with inputs for all four models). Speeds are the mean of the two passes; the
pass-to-pass difference is reported as a noise estimate.
Writes bench_summary.tsv and prints a table.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NWIN = {"restr": 12, "coulomb": 11, "vdw": 21, "ligand": 31}
NS_PER_WINDOW = 10.3

raw = pd.read_csv(HERE / "bench_raw.tsv", sep="\t").dropna(subset=["ns_per_day"])
raw = raw[pd.to_numeric(raw.ns_per_day, errors="coerce").notna()]
raw["ns_per_day"] = raw.ns_per_day.astype(float)
speed = raw.groupby(["model", "complex", "leg"]).agg(nsday=("ns_per_day", "mean"),
                                                      spread=("ns_per_day", lambda s: s.max() - s.min()),
                                                      n=("ns_per_day", "size"), atoms=("atoms", "first")).reset_index()
speed["rel_spread_pct"] = 100 * speed.spread / speed.nsday
print("pass-to-pass timing difference: median %.1f%%, max %.1f%%" % (speed.rel_spread_pct.median(), speed.rel_spread_pct.max()))

lig = speed[speed.leg == "ligand"].set_index("model")
rows = []
for (model, cx), g in speed[speed.leg != "ligand"].groupby(["model", "complex"]):
    s = g.set_index("leg").nsday
    if model not in lig.index or not all(l in s for l in ("restr", "coulomb", "vdw")):
        continue
    days = sum(NWIN[l] * NS_PER_WINDOW / s[l] for l in ("restr", "coulomb", "vdw"))
    days += NWIN["ligand"] * NS_PER_WINDOW / lig.loc[model, "nsday"]
    rows.append(dict(model=model, complex=cx, gpu_days=days, complex_atoms=int(g.atoms.iloc[0]),
                     complex_nsday=g.nsday.mean(), ligand_nsday=lig.loc[model, "nsday"]))
cost = pd.DataFrame(rows)
ref = cost[cost.model == "TIP3P"].set_index("complex").gpu_days
cost["relative"] = [d / ref[c] for d, c in zip(cost.gpu_days, cost.complex)]
summary = cost.groupby("model").agg(gpu_days=("gpu_days", "mean"), gpu_days_min=("gpu_days", "min"),
                                    gpu_days_max=("gpu_days", "max"), relative=("relative", "mean"),
                                    rel_min=("relative", "min"), rel_max=("relative", "max"),
                                    complex_atoms=("complex_atoms", "mean"), complex_nsday=("complex_nsday", "mean"),
                                    ligand_nsday=("ligand_nsday", "first")).reset_index()
summary.to_csv(HERE / "bench_summary.tsv", sep="\t", index=False)
cost.to_csv(HERE / "bench_per_complex.tsv", sep="\t", index=False)
with pd.option_context("display.width", 200):
    print(cost.round(2).to_string(index=False))
    print()
    print(summary.round(2).to_string(index=False))
