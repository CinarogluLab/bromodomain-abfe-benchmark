"""Rocklin et al. (2013) finite-size correction, version 2 (BOARD T4; fixes from Codex's audit,
colab/CHARGE_NUMERICAL_AUDIT.md). Version 1 (rocklin_charge_correction.py, the rocklinc/Alibay
protocol as published) is kept as a protocol comparison.

Changes relative to v1:
  1. RIP and EMP use the MD periodic volume (Rocklin eq 17), not the APBS integration volume; the
     PB domain (box enlarged 10% per axis for complexes) only defines the integration region.
  2. APBS grids are integrated with the nodal trapezoidal rule and the naked-charge reference is
     integrated over the same rectangular domain (charge_terms_independent.residuals_from_dx).
  3. NET/USV use the Ewald self-potential of the actual cell (wigner_inverse_nm), so the one
     orthorhombic solvent box (TIP3P/5LJ0) is treated exactly.
  4. Retained crystal waters (residue TP3) count as solvent: excluded from PB, included in N_S.
  5. PQR charges written with 10 decimals (write_precise_pqr); the PB reference charges are the
     sums actually written.
  6. All terms kept at full precision, with raw IP/IL/IL_SLV and both volumes; TIP7P DSC stays
     missing (no full correction is formed for TIP7P).
  7. Counter-ion convention: primary = published approximation (Q_P = 0 in the analytical terms,
     Rocklin sec II D / Alibay et al.). Sensitivity = actual environment charge of the complex
     (Q_env = -1, protein + all fixed ions), computed from the same integrals.
Frames, radii, APBS settings and water constants are those of v1.

Usage: python rocklin_charge_correction_v2.py <jobfile> <workdir> [--jobs 6] [--eps-variant main|alt]
       [--keep-dx model:pdb:leg:frame,...] [--dime 257]
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import charge_terms_independent as cti                       # noqa: E402
from rocklin_charge_correction import prepare, EPS, GAMMA, ION_RESNAMES, TEMP  # noqa: E402

APBS = os.environ.get("APBS", "apbs")
WATER_RESNAMES = ("SOL", "TP7", "TP3", "TIP3P", "TIP3", "HOH", "WAT")
VERSION = "v2"

APBS_BLOCK = """elec name {name}
    mg-manual
    dime {dime} {dime} {dime}
    glen {x:.4f} {y:.4f} {z:.4f}
    gcent mol 3
    mol {mol}
    lpbe
    bcfl mdh
    pdie 1.0
    sdie {sdie}
    chgm spl4
    srfm smol
    srad 1.4
    swin 0.3
    sdens 40.0
    temp {t}
    calcenergy no
    calcforce no
    write pot dx {name}
end
"""


def write_apbs(path, glen, eps, dime, with_protein):
    blocks = ["read\n    mol pqr prot_only.pqr\n    mol pqr lig_in_prot.pqr\n    mol pqr lig_only.pqr\nend\n"]
    if with_protein:
        blocks.append(APBS_BLOCK.format(name="protein_RIP_het", mol=1, sdie=eps, dime=dime, t=TEMP,
                                        x=glen[0], y=glen[1], z=glen[2]))
    blocks.append(APBS_BLOCK.format(name="ligand_RIP_het", mol=2, sdie=eps, dime=dime, t=TEMP,
                                    x=glen[0], y=glen[1], z=glen[2]))
    blocks.append(APBS_BLOCK.format(name="ligand_RIP_hom", mol=2, sdie=1.0, dime=dime, t=TEMP,
                                    x=glen[0], y=glen[1], z=glen[2]))
    open(path, "w").write("\n".join(blocks) + "quit\n")


def analyse(args):
    job, workdir, variant, keep_dx, dime = args
    import MDAnalysis as mda
    wd = os.path.join(workdir, f"{job['model']}_{job['pdb']}_{job['leg']}")
    radii = prepare(job, wd)
    u = mda.Universe(job["tpr"], os.path.join(wd, "frames.xtc"))
    assert len(u.trajectory) == 11 and len(radii) == u.atoms.n_atoms
    u.add_TopologyAttr("radii", radii)
    lig = u.select_atoms("resname LIG")
    water = u.select_atoms("resname " + " ".join(WATER_RESNAMES))
    solute = u.select_atoms("not resname " + " ".join(WATER_RESNAMES + ION_RESNAMES))
    rest = solute - lig
    complex_leg = job["leg"] == "complex"
    if not complex_leg:
        assert rest.n_atoms == 0, "solvent leg should contain only ligand, water and ions"
    n_water = len(water.residues)
    qL = float(lig.charges.sum())
    q_env_actual = float((u.atoms - lig).charges.sum())      # protein + fixed ions (complex) / ions (solvent)
    eps, gamma = EPS[variant][job["model"]], GAMMA[job["model"]]
    q_all = u.atoms.charges.copy()
    rows = []
    for i, ts in enumerate(u.trajectory):
        # cache namespace includes the dielectric variant (main keeps its original name so that the
        # published main-run caches stay valid); every cached record is also checked for epsilon
        tag = f"{VERSION}_d{dime}" if variant == "main" else f"{VERSION}_{variant}_d{dime}"
        fd = os.path.join(wd, f"{tag}_f{i:02d}"); os.makedirs(fd, exist_ok=True)
        cache = os.path.join(fd, "result.json")
        if os.path.exists(cache):
            rec = json.load(open(cache))
            if abs(rec["eps"] - eps) > 1e-9:
                raise RuntimeError(f"cached frame {cache} has eps {rec['eps']}, expected {eps}")
            rows.append(rec); continue
        box_md = ts.dimensions[:3].copy() / 10.0               # nm, before any PB enlargement
        glen = ts.dimensions[:3] * (1.10 if complex_leg else 1.0)   # Angstrom, PB domain
        # PQR files: ligand charges only / non-ligand solute charges only / ligand (centering)
        u.atoms.charges = q_all
        lig_q, rest_q = lig.charges.copy(), rest.charges.copy() if rest.n_atoms else None
        lig.charges = 0.0
        cti.write_precise_pqr(solute, os.path.join(fd, "prot_only.pqr")); q_prot_pb = float(solute.charges.sum())
        u.atoms.charges = q_all
        if rest.n_atoms:
            rest.charges = 0.0
        cti.write_precise_pqr(solute, os.path.join(fd, "lig_in_prot.pqr")); q_lig_pb = float(solute.charges.sum())
        u.atoms.charges = q_all
        cti.write_precise_pqr(lig, os.path.join(fd, "lig_only.pqr"))
        write_apbs(os.path.join(fd, "apbs.in"), glen, eps, dime, with_protein=complex_leg)
        with open(os.path.join(fd, "apbs.log"), "w") as log:
            subprocess.run([APBS, "apbs.in"], cwd=fd, stdout=log, stderr=subprocess.STDOUT, check=True)
        for f in os.listdir(fd):
            if f.endswith("-PE0.dx"):
                os.replace(os.path.join(fd, f), os.path.join(fd, f.replace("-PE0.dx", ".dx")))
        res = cti.residuals_from_dx(
            ligand_heterogeneous=os.path.join(fd, "ligand_RIP_het.dx"),
            ligand_homogeneous=os.path.join(fd, "ligand_RIP_hom.dx"),
            protein_heterogeneous=os.path.join(fd, "protein_RIP_het.dx") if complex_leg else None,
            q_ligand_pb=q_lig_pb, q_protein_pb=q_prot_pb, epsilon=eps, temperature=TEMP)
        v_md = float(np.prod(box_md))
        lat = cti.wigner_inverse_nm(np.diag(box_md))
        common = dict(volume_nm3=v_md, q_ligand=qL, epsilon=eps, water_count=n_water,
                      gamma_e_nm2=gamma, IP=res["IP"], IL=res["IL"], IL_SLV=res["IL_SLV"],
                      lattice_inverse_nm=lat)
        primary = cti.terms(q_environment=0.0, **common)
        sens = cti.terms(q_environment=(q_env_actual if complex_leg else 0.0), **common)
        out = dict(frame=i, time_ps=float(ts.time), box_md_nm=box_md.tolist(), md_volume_nm3=v_md,
                   pb_volume_nm3=res["pb_volume_nm3"], lattice_inverse_nm=lat, qL=qL,
                   q_lig_pb=q_lig_pb, q_prot_pb=q_prot_pb, q_env_actual=q_env_actual, N_S=n_water,
                   eps=eps, gamma=gamma, IP=res["IP"], IL=res["IL"], IL_SLV=res["IL_SLV"],
                   **{k: primary[k] for k in ("NET", "USV", "NET_USV", "RIP", "EMP", "ANA", "DSC", "total")},
                   **{"sens_" + k: sens[k] for k in ("NET_USV", "RIP", "EMP", "ANA", "total")})
        json.dump(out, open(cache, "w"))
        rows.append(out)
        if f"{job['model']}:{job['pdb']}:{job['leg']}:{i}" not in keep_dx:
            for f in os.listdir(fd):
                if f.endswith(".dx"):
                    os.remove(os.path.join(fd, f))
    json.dump(rows, open(os.path.join(wd, f"frames_{VERSION}_d{dime}_{variant}.json"), "w"), indent=1)
    summary = dict(model=job["model"], pdb=job["pdb"], leg=job["leg"], variant=variant, dime=dime,
                   n_frames=len(rows), qL=qL, q_env_actual=q_env_actual, eps=eps, N_S=n_water,
                   md_volume_nm3=float(np.mean([r["md_volume_nm3"] for r in rows])),
                   pb_volume_nm3=float(np.mean([r["pb_volume_nm3"] for r in rows])))
    for k in ("IP", "IL", "IL_SLV", "NET_USV", "RIP", "EMP", "ANA", "DSC", "total",
              "sens_NET_USV", "sens_RIP", "sens_EMP", "sens_ANA", "sens_total"):
        vals = [r[k] for r in rows]
        if any(v is None for v in vals):
            summary[k] = None; summary[k + "_sd"] = None
        else:
            v = np.array(vals, float); summary[k] = float(v.mean()); summary[k + "_sd"] = float(v.std(ddof=1))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile"); ap.add_argument("workdir")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--eps-variant", default="main", choices=["main", "alt"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--keep-dx", default="")
    ap.add_argument("--dime", type=int, default=257)
    a = ap.parse_args()
    import csv
    jobs = list(csv.DictReader(open(a.jobfile), delimiter="\t"))
    if a.only:
        sel = set(a.only.split(","))
        jobs = [j for j in jobs if f"{j['model']}:{j['pdb']}:{j['leg']}" in sel]
    keep = set(x for x in a.keep_dx.split(",") if x)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(analyse, [(j, a.workdir, a.eps_variant, keep, a.dime) for j in jobs]))
    import pandas as pd
    pd.DataFrame(res).to_csv(os.path.join(a.workdir, f"rocklin_legs_{VERSION}_d{a.dime}_{a.eps_variant}.tsv"),
                             sep="\t", index=False)


if __name__ == "__main__":
    main()
