"""Rocklin et al. (2013) finite-size correction for the four +1 ligands (BOARD T4).

Follows the published implementation used by Alibay et al. (2022, Commun. Chem.;
github.com/IAlibay/fragment-opt-abfe-benchmark/scripts/rocklin_correction) with the rocklinc
package (github.com/xiki-tempula/rocklinc, commit ff23752) and APBS 3.4.1:
  * frames from the fully interacting state of each leg (complex: Coulomb lambda 0; solvent: ligand
    leg state 0), 11 frames at 1, 1.9, ..., 10 ns, molecules made whole and centred;
  * counter-ions and water excluded from the PB calculations; protein charge set to zero in the
    analytical terms (Rocklin sec. II D, "QP artificially set to zero" when counter-ions neutralise),
    with the real protein charges kept in APBS for the protein integrated potential;
  * radii from `gmx editconf -mead` (force-field sigma/2, as in the published example); charges at
    full precision from the run input via MDAnalysis (the PQR keeps only two decimals);
  * APBS grid 257^3 over the box (complex box enlarged by 10% as in the example).
Water-model constants (static permittivity eps_S; quadrupole trace gamma_s for the discrete-solvent
term) are set per model below; the TIP7P DSC term is left out (multiple LJ sites; see
colab/CHARGE_ASSESSMENT.md) and reported as missing.

Leg correction = Delta Delta G_cor for charging the ligand in that leg. Binding correction =
cor(complex) - cor(solvent), added to dG_bind (cycle: binding = charging(complex) - charging(solvent)).

Usage: python rocklin_charge_correction.py <jobs.tsv> <workdir> [--jobs 3] [--eps-variant main|alt]
"""
import argparse, os, re, subprocess, sys, json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GMX = os.environ.get("GMX", "gmx")
APBS = os.environ.get("APBS", "apbs")
TEMP = 298.15
WATER_RESNAMES = ("SOL", "TP7", "TIP3P", "TIP3", "HOH", "WAT")
ION_RESNAMES = ("NA", "CL", "Na", "Cl")
# eps_S: Vega & Abascal 2011 (TIP3P 94, TIP4P 50, TIP5P 91); TIP7P reproduces the experimental
# value (Yu et al. 2019), 78.4 at 298 K. Variant 'alt': Rocklin 2013 TIP3P 97, Mahoney & Jorgensen
# 2000 TIP5P 81.5. gamma_s (e nm^2): quadrupole trace about O from the actual run geometry.
# TIP7P: 76.098 = the model's own static dielectric constant at 298 K (Yu et al. 2019, Table 2);
# it replaced the experimental 78.4 used in the first v2 run (archived as v2eps78p4_*), 2026-10-09.
EPS = {"main": {"TIP3P": 94.0, "TIP4P": 50.0, "TIP5P": 91.0, "TIP7P": 76.098},
       "alt": {"TIP3P": 97.0, "TIP4P": 50.0, "TIP5P": 81.5, "TIP7P": 76.098}}
GAMMA = {"TIP3P": 0.00764137, "TIP4P": 0.00929481, "TIP5P": 0.00205444, "TIP7P": None}


def run(cmd, cwd, stdin=None):
    p = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
    return p.stdout


def charges_from_tpr(tpr, wd):
    out = run([GMX, "dump", "-s", tpr], wd)
    q = [float(x) for x in re.findall(r"atom\[\s*\d+\]=\{type=\s*\d+, typeB=\s*\d+, ptype=\s*\w+, m=[^,]+, q=\s*([-+0-9.eE]+)", out)]
    if not q:  # older/newer dump layouts
        q = [float(x) for x in re.findall(r"atom\[\s*\d+\]=\{[^}]*?\bq=\s*([-+0-9.eE]+)", out)]
    return np.array(q)


def prepare(job, wd):
    os.makedirs(wd, exist_ok=True)
    done = [os.path.join(wd, f) for f in ("full.pqr", "frames.xtc")]
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in done):
        return np.array([float(l.split()[-1]) for l in open(done[0]) if l.startswith(("ATOM", "HETATM"))])
    run([GMX, "editconf", "-f", job["tpr"], "-mead", "full.pqr"], wd)
    run([GMX, "editconf", "-f", job["tpr"], "-o", "conf.gro"], wd)
    radii = np.array([float(l.split()[-1]) for l in open(os.path.join(wd, "full.pqr")) if l.startswith(("ATOM", "HETATM"))])
    centre = "Protein" if job["leg"] == "complex" else "LIG"
    run([GMX, "trjconv", "-s", job["tpr"], "-f", job["xtc"], "-b", "1000", "-dt", "900",
         "-pbc", "mol", "-center", "-o", "frames.xtc"], wd, stdin=f"{centre}\nSystem\n")
    return radii


def analyse(job_and_args):
    job, workdir, variant = job_and_args
    import MDAnalysis as mda
    import quantities as pq
    import rocklinc
    wd = os.path.join(workdir, f"{job['model']}_{job['pdb']}_{job['leg']}")
    radii = prepare(job, wd)
    # topology, residue names and full-precision charges from the run input (expanded per molecule)
    u = mda.Universe(job["tpr"], os.path.join(wd, "frames.xtc"))
    if len(u.trajectory) != 11:               # incomplete frames from an interrupted run: redo
        for f in ("full.pqr", "frames.xtc"):
            os.remove(os.path.join(wd, f))
        radii = prepare(job, wd)
        u = mda.Universe(job["tpr"], os.path.join(wd, "frames.xtc"))
    assert len(u.trajectory) == 11, len(u.trajectory)
    assert len(radii) == u.atoms.n_atoms, (len(radii), u.atoms.n_atoms)
    u.add_TopologyAttr("radii", radii)
    lig = "resname LIG"
    water = "resname " + " ".join(WATER_RESNAMES)
    keep = u.select_atoms("not resname " + " ".join(ION_RESNAMES))
    qL = float(u.select_atoms(lig).charges.sum())
    eps, gamma = EPS[variant][job["model"]], GAMMA[job["model"]]

    class Water:  # rocklinc reads epsilon_S and gamma_s from the class
        epsilon_S = eps
        gamma_s = (gamma if gamma is not None else 0.0) * pq.e * pq.nm ** 2

    rows = []
    for i, ts in enumerate(u.trajectory):
        fd = os.path.join(wd, f"f{i:02d}"); os.makedirs(fd, exist_ok=True)
        cache = os.path.join(fd, "result.json")
        if os.path.exists(cache):             # finished in an earlier (interrupted) run
            rows.append(json.load(open(cache))); continue
        cwd = os.getcwd(); os.chdir(fd)
        try:
            corr = rocklinc.RocklinCorrection(u.dimensions[:3], qL, 0.0, TEMP, water=Water)
            if job["leg"] == "complex":
                u.dimensions[:3] *= 1.10
            charges = u.atoms.charges.copy()
            corr.make_APBS_input(universe=keep, ligand_selection=lig, solvent_selection=water,
                                 in_prot_only="prot_only.pqr", in_lig_in_prot="lig_in_prot.pqr",
                                 in_lig_only="lig_only.pqr", apbs_in="apbs.in")
            u.atoms.charges = charges
            with open("apbs.log", "w") as log:
                subprocess.run([APBS, "apbs.in"], stdout=log, stderr=subprocess.STDOUT, check=True)
            for f in os.listdir("."):
                if f.endswith("-PE0.dx"):
                    os.replace(f, f.replace("-PE0.dx", ".dx"))
            if job["leg"] == "solvent":
                corr.read_APBS(IP=0)
            else:
                corr.read_APBS()
            total = corr.compute()
            out = dict(frame=i, time_ps=float(ts.time), qL=qL, eps=eps, NS=int(corr.NS),
                       box_nm=float(np.cbrt(np.prod(ts.dimensions[:3])) / 10))
            for line in corr.output:
                m = re.match(r"ΔΔG_(NET|RIP|EMP)\(L\) = (-?[\d.]+) kJ/mol", line)
                if m: out[m.group(1)] = float(m.group(2)) / 4.184
                m = re.match(r"ΔΔG_NET\(L\) \+ ΔΔG_USV\(L\) = (-?[\d.]+) kJ/mol", line)
                if m: out["NET_USV"] = float(m.group(1)) / 4.184
                m = re.match(r"= ΔΔG_ANA\(L\): (-?[\d.]+) kJ/mol \+ ΔΔG_DSC\(L\): (-?[\d.]+) kJ/mol", line)
                if m: out["ANA"] = float(m.group(1)) / 4.184; out["DSC"] = float(m.group(2)) / 4.184
            if gamma is None:
                out["DSC"] = float("nan")
            out["total"] = out["ANA"] + (0.0 if gamma is None else out["DSC"])
            rows.append(out)
            json.dump(out, open("result.json", "w"))
            for f in os.listdir("."):          # keep logs and inputs, drop the 200 MB grids
                if f.endswith(".dx"): os.remove(f)
        finally:
            os.chdir(cwd)
    with open(os.path.join(wd, "frames.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    keys = ["NET", "NET_USV", "RIP", "EMP", "ANA", "DSC", "total"]
    summary = dict(model=job["model"], pdb=job["pdb"], leg=job["leg"], variant=variant, n_frames=len(rows),
                   qL=qL, eps=eps, NS=rows[0]["NS"], box_nm=np.mean([r["box_nm"] for r in rows]),
                   xtc=job["xtc"], tpr=job["tpr"])
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        summary[k] = float(v.mean()); summary[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k not in ("xtc", "tpr")}), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile"); ap.add_argument("workdir")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--eps-variant", default="main", choices=["main", "alt"])
    ap.add_argument("--only", default=None, help="model:pdb:leg,...")
    a = ap.parse_args()
    import csv
    jobs = list(csv.DictReader(open(a.jobfile), delimiter="\t"))
    if a.only:
        sel = set(a.only.split(","))
        jobs = [j for j in jobs if f"{j['model']}:{j['pdb']}:{j['leg']}" in sel]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(analyse, [(j, a.workdir, a.eps_variant) for j in jobs]))
    import pandas as pd
    pd.DataFrame(res).to_csv(os.path.join(a.workdir, f"rocklin_legs_{a.eps_variant}.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
