"""Aggregate accuracy statistics quoted in the main text (Table 4, Results, Conclusions, Table S3).
"""
import numpy as np
from scipy import stats

from pairwise_ddg import CALC, EXP, PDB

DH = np.array([-8.42, -12.11, -13.28, -7.00, -12.48, -5.51, -7.81, -17.00, -3.66, -4.80])  # Table 2, ITC
TDS = DH - EXP


def main():
    n = len(EXP)
    print(f"{'model':6s} {'MUE':>5s} {'RMSE':>5s} {'bias':>6s} {'r [95% CI, Fisher z]':>22s} {'tau (p)':>13s} "
          f"{'offset-corr RMSE':>17s} {'r(err,TdS) (p)':>16s}")
    for model, values in CALC.items():
        calc = np.array(values)
        err = calc - EXP
        rmse = np.sqrt(np.mean(err**2))
        r = stats.pearsonr(EXP, calc)[0]
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
        tau, p_tau = stats.kendalltau(EXP, calc)
        centered = np.sqrt(np.mean((err - err.mean()) ** 2))
        r_tds, p_tds = stats.pearsonr(err, TDS)
        print(f"{model:6s} {np.abs(err).mean():5.2f} {rmse:5.2f} {err.mean():6.2f} "
              f"{r:5.2f} [{lo:5.2f}, {hi:5.2f}]      {tau:5.2f} ({p_tau:.3f}) "
              f"{centered:8.2f} ({100 * (1 - centered / rmse):3.0f}%) {r_tds:9.2f} ({p_tds:.3f})")

    spread = np.ptp(np.array(list(CALC.values())), axis=0)
    print("\nper-complex spread across water models (max - min, kcal/mol):")
    print("  " + "  ".join(f"{p} {s:.2f}" for p, s in sorted(zip(PDB, spread), key=lambda t: -t[1])))


if __name__ == "__main__":
    main()
