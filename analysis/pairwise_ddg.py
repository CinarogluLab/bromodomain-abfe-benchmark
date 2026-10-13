"""Pairwise ddG analysis for the pairwise-comparison figure (main-text Figure 5; all 45 complex pairs, single-run dG from Table 3)."""
import itertools
import numpy as np
from scipy import stats

PDB = ["3MXF", "4ZQL", "5DKH", "5I8G", "5IGL", "5LJ0", "6V1B", "7BL8", "7L9G", "8AG2"]
BRD = ["BRD4-BD1", "TRIM24", "SMARCA2", "CREBBP", "TAF1L", "ATAD2", "BRD9", "BAZ2A", "BRD2-BD2", "BPTF"]
EXP = np.array([-9.64, -9.07, -9.79, -10.26, -9.72, -10.99, -10.14, -9.49, -8.49, -9.38])
# dG_bind from abfe_legs.tsv (build_legs.py): MBAR re-analysis of the complete raw data for all 160 legs
# (2026-10-06/07, including the TIP3P/5DKH and TIP3P/3MXF completions and the TIP5P/7BL8 archive data).
CALC = {
    "TIP3P": [-9.41, -8.54, -7.23, -13.91, -7.67, -14.00, -11.12, -10.82, -9.04, -9.45],
    "TIP4P": [-9.37, -9.40, -10.75, -14.56, -9.52, -14.31, -12.79, -13.67, -10.22, -11.14],
    "TIP5P": [-10.01, -10.67, -6.91, -15.18, -9.42, -10.13, -11.33, -12.04, -10.90, -11.08],
    "TIP7P": [-9.12, -8.93, -12.82, -13.97, -7.49, -14.33, -12.38, -11.30, -7.34, -11.33],
}
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, np.inf)]


def pairs(values, keep):
    idx = [i for i in range(len(PDB)) if i in keep]
    return np.array([values[i] - values[j] for i, j in itertools.combinations(idx, 2)])


def summarize(keep):
    dexp = pairs(EXP, keep)
    out = {}
    for model, calc in CALC.items():
        dcalc = pairs(np.array(calc), keep)
        err = dcalc - dexp
        out[model] = dict(
            n=len(dexp),
            sign=np.mean(np.sign(dcalc) == np.sign(dexp)),
            mue=np.mean(np.abs(err)),
            rmse=np.sqrt(np.mean(err**2)),
            r=stats.pearsonr(dexp, dcalc)[0],
            within1=np.mean(np.abs(err) <= 1.0),
            within05=np.mean(np.abs(err) <= 0.5),
            sign_by_bin=[(np.sum((np.abs(dexp) >= lo) & (np.abs(dexp) < hi)),
                          np.mean(np.sign(dcalc) == np.sign(dexp), where=(np.abs(dexp) >= lo) & (np.abs(dexp) < hi)))
                         for lo, hi in BINS],
        )
    return dexp, out


if __name__ == "__main__":
    for label, keep in [("all 10 complexes", set(range(10))), ("excluding 5I8G", set(range(10)) - {3})]:
        dexp, res = summarize(keep)
        print(f"\n== {label}: {len(dexp)} pairs; |ddG_exp| range {np.abs(dexp).min():.2f}-{np.abs(dexp).max():.2f}, "
              f"median {np.median(np.abs(dexp)):.2f}; bins <0.5/0.5-1/>1: "
              + "/".join(str(np.sum((np.abs(dexp) >= lo) & (np.abs(dexp) < hi))) for lo, hi in BINS))
        for m, s in res.items():
            bins = "  ".join(f"{n}:{f:.0%}" for n, f in s["sign_by_bin"])
            print(f"{m}: sign {s['sign']:.0%}  MUE {s['mue']:.2f}  RMSE {s['rmse']:.2f}  r {s['r']:.2f}  "
                  f"|err|<=1 {s['within1']:.0%}  |err|<=0.5 {s['within05']:.0%}  sign by bin [{bins}]")
