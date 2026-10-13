"""Build abfe_legs.tsv from the MBAR re-analysis of the raw dhdl files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEGS = ["coulomb", "vdw", "restr", "ligand"]

old = pd.read_csv(HERE / "mbar" / "abfe_legs_v5_alchemical_analysis.tsv", sep="\t")
new = pd.read_csv(HERE / "mbar" / "mbar_legs.tsv", sep="\t").dropna(subset=["dG_mbar_all"])
new["sigma"] = np.maximum(new["err"], (new["dG_first_half"] - new["dG_second_half"]).abs() / 2)
new = new.set_index(["model", "pdb", "leg"])

rows, prov = [], []
for _, o in old.iterrows():
    row = {"model": o.model, "pdb": o.pdb}
    for leg in LEGS:
        key = (o.model, o.pdb, leg)
        if key in new.index:
            n = new.loc[key]
            row[leg], row[f"{leg}_err"] = round(float(n.dG_mbar_all), 3), round(float(n.sigma), 3)
            prov.append(dict(model=o.model, pdb=o.pdb, leg=leg, status="mbar_reanalysis",
                             dG_ti=round(float(n.dG_ti), 3), dG_mbar_decorrelated=round(float(n.dG), 3),
                             min_neff=int(n.min_neff), short_windows=n.short if isinstance(n.short, str) else ""))
        else:
            row[leg], row[f"{leg}_err"] = o[leg], o[f"{leg}_err"]
            prov.append(dict(model=o.model, pdb=o.pdb, leg=leg, status="provisional_v5_value"))
    row["dG_off"] = o.dG_off
    rows.append(row)

pd.DataFrame(rows).to_csv(HERE / "abfe_legs.tsv", sep="\t", index=False)
pd.DataFrame(prov).to_csv(HERE / "mbar" / "abfe_legs_provenance.tsv", sep="\t", index=False)
p = pd.DataFrame(prov)
print("provisional legs:", p[p.status != "mbar_reanalysis"][["model", "pdb", "leg"]].values.tolist())
d = pd.DataFrame(rows)
d["dG"] = d.ligand - (d.coulomb + d.vdw + d.restr) - d.dG_off
for m in ["TIP3P", "TIP4P", "TIP5P", "TIP7P"]:
    print(f'    "{m}": [' + ", ".join(f"{v:.2f}" for v in d[d.model == m].set_index("pdb").loc[
        ["3MXF", "4ZQL", "5DKH", "5I8G", "5IGL", "5LJ0", "6V1B", "7BL8", "7L9G", "8AG2"], "dG"]) + "],")
