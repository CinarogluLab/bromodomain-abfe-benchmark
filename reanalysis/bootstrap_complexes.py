"""Paired bootstrap over complexes, never over the 45 dependent pairs.

Run with the plot environment. Provisional until MBAR re-analysis is complete.
The common provisional analysis excludes 5DKH from every model, because
its currently stored TIP3P value represents an incomplete endpoint path.
An additional sensitivity analysis excludes 5I8G from that same common set.
"""
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / 'analysis'
sys.path.insert(0, str(ANALYSIS))
from pairwise_ddg import CALC, EXP, PDB

SEED = 20261006
NBOOT = 20000
CHARGED = {'5LJ0', '6V1B', '7L9G', '8AG2'}


def metrics(experimental, predicted):
    err = predicted - experimental
    x = experimental - experimental.mean(axis=-1, keepdims=True)
    y = predicted - predicted.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    r = np.divide((x*y).sum(axis=-1), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    i, j = np.triu_indices(experimental.shape[-1], 1)
    sx = np.sign(experimental[..., i] - experimental[..., j])
    sy = np.sign(predicted[..., i] - predicted[..., j])
    denominator = np.sqrt((sx != 0).sum(axis=-1) * (sy != 0).sum(axis=-1))
    tau = np.divide((sx*sy).sum(axis=-1), denominator, out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
    return {'MUE': np.abs(err).mean(axis=-1), 'RMSE': np.sqrt((err*err).mean(axis=-1)),
            'bias': err.mean(axis=-1), 'r': r, 'tau_b': tau}


def interval(values):
    finite = values[np.isfinite(values)]
    return (*np.quantile(finite, [0.025, 0.975]), len(finite))


def main():
    rng = np.random.default_rng(SEED)
    rows, differences, charge_rows = [], [], []
    models = list(CALC)
    scopes = {'common_provisional': [i for i, p in enumerate(PDB) if p != '5DKH'],
              'common_without_5I8G': [i for i, p in enumerate(PDB) if p not in {'5DKH', '5I8G'}]}
    for scope, keep in scopes.items():
        experimental = EXP[keep]
        idx = rng.integers(0, len(keep), size=(NBOOT, len(keep)))
        draws = {}
        points = {}
        for model in models:
            predicted = np.array(CALC[model])[keep]
            draws[model] = metrics(experimental[idx], predicted[idx])
            points[model] = metrics(experimental, predicted)
            for metric, values in draws[model].items():
                lo, hi, nfinite = interval(values)
                rows.append(dict(scope=scope, n_complexes=len(keep), model=model, metric=metric,
                                 point=float(points[model][metric]), ci_low=lo, ci_high=hi,
                                 finite_bootstraps=nfinite))
        for a, b in itertools.combinations(models, 2):
            for metric in draws[a]:
                lo, hi, nfinite = interval(draws[a][metric] - draws[b][metric])
                differences.append(dict(scope=scope, n_complexes=len(keep), model_a=a, model_b=b,
                                        metric=metric, point=float(points[a][metric]-points[b][metric]),
                                        ci_low=lo, ci_high=hi, finite_bootstraps=nfinite))
        for charge, group in [('neutral', [i for i in keep if PDB[i] not in CHARGED]),
                              ('plus_one', [i for i in keep if PDB[i] in CHARGED])]:
            for model in models:
                vals = metrics(EXP[group], np.array(CALC[model])[group])
                charge_rows.append(dict(scope=scope, charge=charge, n_complexes=len(group), model=model,
                                        MUE=float(vals['MUE']), RMSE=float(vals['RMSE']), bias=float(vals['bias'])))
    for name, data in [('bootstrap_metrics.tsv', rows), ('bootstrap_differences.tsv', differences), ('charge_strata.tsv', charge_rows)]:
        with (HERE / name).open('w') as stream:
            writer = csv.DictWriter(stream, data[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(data)
    provenance = dict(seed=SEED, nboot=NBOOT, source_sha256=hashlib.sha256((ANALYSIS/'pairwise_ddg.py').read_bytes()).hexdigest(),
                      PDB=PDB, EXP=EXP.tolist(), CALC=CALC, scopes=scopes, provisional=True)
    (HERE/'bootstrap_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    lines = ['# Complex-level bootstrap (provisional)', '',
             'The stored ABFE values require re-analysis. These results are diagnostic, not final manuscript numbers.', '',
             f'Paired percentile bootstrap: {NBOOT:,} draws, seed {SEED}. Each draw resamples whole '
             'complexes and uses the same indices for every water model. Experimental observations are held '
             'fixed. Intervals describe sensitivity to this benchmark composition; they do not estimate '
             'trajectory convergence, replicate variation, or experimental uncertainty. This purposively '
             'selected small benchmark is not a random sample of all bromodomain systems.', '',
             'The common set excludes 5DKH for every model so incomplete TIP3P data cannot distort '
             'paired comparisons. An additional sensitivity analysis excludes the shared 5I8G outlier. '
             'Neither exclusion changes the underlying CALC data. Missing TIP5P/7BL8 provenance remains unresolved.', '',
             'Pearson r and Kendall tau-b are undefined for all-identical resamples; finite draw counts '
             'are reported. No bootstrap p-values or claims of superiority are inferred from these intervals.', '',
             '| Scope | Model | MUE [95% CI] | RMSE [95% CI] |', '|---|---|---:|---:|']
    for scope in scopes:
        for model in models:
            selected = {r['metric']: r for r in rows if r['scope'] == scope and r['model'] == model}
            fmt = lambda r: f'{r["point"]:.2f} [{r["ci_low"]:.2f}, {r["ci_high"]:.2f}]'
            lines.append(f'| {scope} | {model} | {fmt(selected["MUE"])} | {fmt(selected["RMSE"])} |')
    lines += ['', 'Paired MUE differences (A minus B), common set:', '']
    for r in differences:
        if r['scope'] == 'common_provisional' and r['metric'] == 'MUE':
            lines.append(f'- {r["model_a"]} − {r["model_b"]}: {r["point"]:.2f} [{r["ci_low"]:.2f}, {r["ci_high"]:.2f}] kcal/mol.')
    lines += ['', 'Charge groups contain only four +1 complexes and five neutral complexes in the common set. '
              'They differ in protein and ligand identity; any error difference is descriptive and cannot '
              'identify an electrostatic artifact or water mechanism. See `charge_strata.tsv`.', '',
              'The 45 pairwise differences share the same ten complexes and must not be treated as 45 independent observations.']
    (HERE/'BOOTSTRAP_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
