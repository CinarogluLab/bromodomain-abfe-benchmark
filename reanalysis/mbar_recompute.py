"""Recompute every ABFE leg by MBAR directly from the GROMACS dhdl files (BOARD T1).

Discovery: every dhdl*.xvg under the four model trees and under an extraction directory
for archived runs (see extract_dhdl.sh). Analysis copies (paths containing "RESULTS") and
GROMACS backups ("#...#") are ignored. The leg is identified from the number of lambda
states in the xvg header (12 restraint, 11 Coulomb, 21 vdW, 31 combined solvent) and the
state index from the "lambda state N" subtitle, not from folder names.

Protocol: T = 298.15 K, first 1 ns of each window discarded (as in the original
alchemical_analysis -s 1000), each state subsampled to uncorrelated frames (alchemlyb
decorrelate_u_nk, method "dE"), MBAR. Leg value = f(last state) - f(state 0) in kcal/mol,
the sign convention of analysis/abfe_legs.tsv.

Usage:
  python mbar_recompute.py --archives <dir with extracted archives> --out <prefix>
Writes <prefix>_windows.tsv (one row per dhdl file chosen), <prefix>_legs.tsv and a
comparison with abfe_legs.tsv on stdout.
"""
import argparse, collections, glob, os, re, sys

import os as _os
_os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep the GPU free for GROMACS
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import pandas as pd
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.preprocessing import decorrelate_u_nk
from alchemlyb.estimators import MBAR

ROOT = os.environ.get("ABFE_RUN_ROOT", "/path/to/run_tree")
MODELS = ("TIP3P", "TIP4P", "TIP5P", "TIP7P")
PDBS = ("3MXF", "4ZQL", "5DKH", "5I8G", "5IGL", "5LJ0", "6V1B", "7BL8", "7L9G", "8AG2")
LEG_BY_NSTATES = {12: "restr", 11: "coulomb", 21: "vdw", 31: "ligand"}
T = 298.15
KT = 0.0019872043 * T          # kcal/mol
DISCARD_PS = 1000.0
FULL_PS = 10000.0


def header(fn):
    """Return (state index, number of foreign-lambda columns, last time in ps)."""
    state, nforeign = None, 0
    with open(fn, "rb") as f:
        for raw in f:
            line = raw.decode(errors="replace")
            if not line.startswith(("@", "#")):
                break
            m = re.search(r"state (\d+)", line) if "subtitle" in line else None
            if m:
                state = int(m.group(1))
            if re.match(r"@ s\d+ legend .*to \(", line) or re.match(r'@ s\d+ legend ".*xD\\f\{\}H \\xl\\f\{\} to', line):
                nforeign += 1
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))
        tail = [l for l in f.read().decode(errors="replace").splitlines() if l and l[0] not in "#@"]
    last = 0.0
    if tail:
        ncol = len(tail[-2].split()) if len(tail) > 1 else None
        rec = tail[-1] if ncol is None or len(tail[-1].split()) == ncol else tail[-2]
        last = float(rec.split()[0])   # last complete record (the final line may be truncated)
    return state, nforeign, last


def identify(path):
    rel = path.split("/All_Waters/")[-1] if "/All_Waters/" in path else path
    model = next((m for m in MODELS if m in rel.upper()), None)
    up = rel.upper().replace("5LJ/", "5LJ0/").replace("5LJ__", "5LJ0__")
    pdb = next((p for p in PDBS if p in up), None)
    return model, pdb


def discover(archive_dir):
    files = glob.glob(f"{ROOT}/TIP*P/**/dhdl*.xvg", recursive=True)
    if archive_dir:
        files += glob.glob(f"{archive_dir}/**/dhdl*.xvg", recursive=True)
    rows = []
    for fn in files:
        if "RESULTS" in fn.upper() or "#" in os.path.basename(fn):
            continue
        model, pdb = identify(fn if not archive_dir or not fn.startswith(archive_dir)
                              else os.path.relpath(fn, archive_dir))
        try:
            state, nf, last = header(fn)
        except (OSError, IndexError, ValueError):
            continue
        leg = LEG_BY_NSTATES.get(nf)
        rows.append(dict(model=model, pdb=pdb, leg=leg, state=state, nstates=nf,
                         last_ps=last, size=os.path.getsize(fn), path=fn))
    return pd.DataFrame(rows)


def sha256(fn, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def choose(df, prefer=None):
    """One file per (model, pdb, leg, state).

    Rule (deterministic): a path matching --prefer (e.g. an authorised rerun directory)
    wins; otherwise the copy with the longest production, then the lexically first path.
    Every alternative copy is written to <out>_alternatives.tsv with its hash, so no
    trajectory is chosen silently; copies whose data hash differs are flagged."""
    df = df.dropna(subset=["model", "pdb", "leg", "state"]).copy()
    df["preferred"] = df.path.str.contains(prefer) if prefer else False
    df["copies"] = df.groupby(["model", "pdb", "leg", "state"])["path"].transform("count")
    df = df.sort_values(["preferred", "last_ps", "path"], ascending=[False, False, True])
    multi = df[df.copies > 1].copy()
    multi["sha256"] = [sha256(p) for p in multi.path]
    chosen = df.drop_duplicates(["model", "pdb", "leg", "state"])
    multi["chosen"] = multi.path.isin(chosen.path)
    return chosen.sort_values(["model", "pdb", "leg", "state"]), multi


def mbar_leg(paths, decorrelate=True):
    """MBAR on decorrelated samples (primary), plus MBAR on all frames and TI as checks.

    Also returns the smallest effective sample size of any window (from the dE series) and
    a forward/reverse convergence pair (first and last half of the retained data)."""
    from alchemlyb.parsing.gmx import extract_dHdl
    from alchemlyb.estimators import TI
    from pymbar.timeseries import statistical_inefficiency
    u_all, u_dec, h_all, neff = [], [], [], []
    for p in paths:
        d = extract_u_nk(p, T=T)
        d = d[d.index.get_level_values("time") >= DISCARD_PS]
        h = extract_dHdl(p, T=T)
        h = h[h.index.get_level_values("time") >= DISCARD_PS]
        dd = decorrelate_u_nk(d, method="dE", remove_burnin=False)
        u_all.append(d); u_dec.append(dd); h_all.append(h)
        neff.append(len(dd))
    # endpoints: the state columns must be ordered from the coupled (first) to the
    # decoupled (last) state, which holds when every lambda component is non-decreasing
    lam = np.array([list(c) if isinstance(c, tuple) else [c] for c in u_all[0].columns], dtype=float)
    assert (np.diff(lam, axis=0) >= -1e-9).all(), "lambda columns not monotonic"
    res = {}
    m = MBAR().fit(pd.concat(u_dec if decorrelate else u_all))
    res["dG"], res["err"] = m.delta_f_.iloc[0, -1] * KT, m.d_delta_f_.iloc[0, -1] * KT
    m = MBAR().fit(pd.concat(u_all))
    res["dG_mbar_all"] = m.delta_f_.iloc[0, -1] * KT
    t = TI().fit(pd.concat(h_all))
    res["dG_ti"] = t.delta_f_.iloc[0, -1] * KT
    halves = []
    for part in (0, 1):
        sub = []
        for d in u_dec:
            n = len(d) // 2
            sub.append(d.iloc[:n] if part == 0 else d.iloc[n:])
        halves.append(MBAR().fit(pd.concat(sub)).delta_f_.iloc[0, -1] * KT)
    res["dG_first_half"], res["dG_second_half"] = halves
    res["min_neff"] = int(min(neff))
    res["min_neff_state"] = int(np.argmin(neff))
    return res


def analyse_leg(task):
    model, pdb, leg, g, decorrelate = task
    n_exp = {"restr": 12, "coulomb": 11, "vdw": 21, "ligand": 31}[leg]
    states = sorted(g.state.astype(int))
    short = g[g.last_ps < FULL_PS - 10]
    row = dict(model=model, pdb=pdb, leg=leg, n_states=len(states), expected=n_exp,
               missing=",".join(str(s) for s in range(n_exp) if s not in states),
               short=",".join(f"{int(s)}:{t/1000:.1f}ns" for s, t in zip(short.state, short.last_ps)),
               multi_copy=int((g.copies > 1).sum()))
    if len(states) == n_exp:
        try:
            row.update(mbar_leg(list(g.sort_values("state").path), decorrelate=decorrelate))
        except Exception as e:  # report and continue
            row["error"] = repr(e)[:200]
    print(model, pdb, leg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                            if k not in ("model", "pdb", "leg")}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="e.g. TIP3P:4ZQL:vdw")
    ap.add_argument("--legs", default=None, help="comma list of legs to analyse, e.g. restr")
    ap.add_argument("--no-decorrelate", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--prefer", default=None, help="regex of paths to prefer, e.g. vdw_rerun")
    a = ap.parse_args()

    win, alts = choose(discover(a.archives), a.prefer)
    win.to_csv(f"{a.out}_windows.tsv", sep="\t", index=False)
    alts.to_csv(f"{a.out}_alternatives.tsv", sep="\t", index=False)

    tasks = []
    for (model, pdb, leg), g in win.groupby(["model", "pdb", "leg"]):
        if a.only and f"{model}:{pdb}:{leg}" not in a.only.split(","):
            continue
        if a.legs and leg not in a.legs.split(","):
            continue
        tasks.append((model, pdb, leg, g, not a.no_decorrelate))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(analyse_leg, tasks))
    legs = pd.DataFrame(rows)
    legs.to_csv(f"{a.out}_legs.tsv", sep="\t", index=False)

    ref = pd.read_csv(os.path.join(os.path.dirname(__file__), "../analysis/abfe_legs.tsv"), sep="\t")
    ref = ref.melt(id_vars=["model", "pdb"], value_vars=["coulomb", "vdw", "restr", "ligand"],
                   var_name="leg", value_name="table")
    cmp_ = legs.merge(ref, on=["model", "pdb", "leg"], how="outer")
    cmp_["diff"] = cmp_["dG"] - cmp_["table"]
    cmp_.to_csv(f"{a.out}_compare.tsv", sep="\t", index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(cmp_[["model", "pdb", "leg", "table", "dG", "err", "diff", "dG_ti", "min_neff", "missing", "short"]].round(3))


if __name__ == "__main__":
    main()
