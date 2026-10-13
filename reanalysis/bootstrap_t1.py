"""Paired complex bootstrap for T1 estimator sensitivity; leaves manuscript data intact.

Run with the plot environment. BCa and percentile intervals are diagnostics for
benchmark composition, not trajectory or measurement uncertainty.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from bootstrap_complexes import metrics, EXP, PDB, CALC, SEED, NBOOT, CHARGED

HERE = Path(__file__).resolve().parent
OUT = HERE / 't6_t1_results'
SOURCES = [HERE/'t1_results/t1_legs.tsv', HERE/'t1_results/t1_restr_legs.tsv']
OFF = HERE.parent/'analysis/abfe_legs.tsv'
ESTIMATORS = {'all_frame_MBAR': 'dG_mbar_all', 'decorrelated_MBAR': 'dG', 'TI': 'dG_ti'}


def intervals(draws, point, jack):
    """BCa with midpoint tie handling; disclose undefined resamples explicitly."""
    finite = np.asarray(draws)[np.isfinite(draws)]
    result = dict(finite_bootstraps=len(finite), invalid_bootstraps=len(draws)-len(finite))
    low, high = np.quantile(finite, [.025, .975])
    result.update(percentile_low=float(low), percentile_high=float(high))
    delta = np.mean(jack)-jack
    denominator = 6 * np.sum(delta**2)**1.5
    if not np.all(np.isfinite(jack)) or denominator == 0:
        result.update(bca_low=np.nan, bca_high=np.nan, bca_status='undefined_jackknife')
        return result
    acceleration = np.sum(delta**3)/denominator
    fraction = (np.sum(finite < point)+.5*np.sum(finite == point))/len(finite)
    if not 0 < fraction < 1:
        result.update(bca_low=np.nan, bca_high=np.nan, bca_status='extreme_bias')
        return result
    z0 = norm.ppf(fraction)
    z = z0+norm.ppf([.025, .975])
    divisor = 1-acceleration*z
    adjusted = norm.cdf(z0+z/divisor)
    if np.any(divisor <= 0) or adjusted[0] >= adjusted[1]:
        result.update(bca_low=np.nan, bca_high=np.nan, bca_status='unstable_adjustment')
        return result
    low, high = np.quantile(finite, adjusted)
    result.update(bca_low=float(low), bca_high=float(high), bca_acceleration=float(acceleration),
                  bca_bias_z=float(z0), bca_status='conditional_finite_draws' if len(finite)<len(draws) else 'ok')
    return result


def load_predictions():
    legs = {}
    for source in SOURCES:
        for row in csv.DictReader(source.open(), delimiter='\t'):
            if int(row['n_states']) != int(row['expected']) or row.get('error'):
                continue
            if not all(row.get(column) and np.isfinite(float(row[column])) for column in ESTIMATORS.values()):
                continue
            legs[row['model'],row['pdb'],row['leg']] = row
    off = {(r['model'],r['pdb']): float(r['dG_off']) for r in csv.DictReader(OFF.open(), delimiter='\t')}
    assembled = {}
    for name, column in ESTIMATORS.items():
        assembled[name] = {}
        for model in CALC:
            values = []
            for pdb in PDB:
                needed = [(model,pdb,leg) for leg in ['ligand','coulomb','vdw','restr']]
                if not all(key in legs for key in needed):
                    values.append(np.nan)
                    continue
                sol,elec,vdw,restr = [float(legs[key][column]) for key in needed]
                values.append(sol-elec-vdw-restr-off[model,pdb])
            assembled[name][model] = np.asarray(values)
    return assembled


def main():
    OUT.mkdir(exist_ok=True)
    predictions = load_predictions()
    common = [i for i in range(len(PDB)) if all(np.isfinite(v[i]) for models in predictions.values() for v in models.values())]
    scopes = {'common_complete': common, 'common_without_5I8G': [i for i in common if PDB[i]!='5I8G']}
    rows, differences, strata = [], [], []
    for scope, keep in scopes.items():
        experimental = EXP[keep]
        draws_idx = np.random.default_rng(SEED).integers(0,len(keep),size=(NBOOT,len(keep)))
        jack_idx = np.array([np.delete(np.arange(len(keep)),i) for i in range(len(keep))])
        for estimator, models in predictions.items():
            draws, points, jack = {}, {}, {}
            for model, all_values in models.items():
                values = all_values[keep]
                draws[model] = metrics(experimental[draws_idx],values[draws_idx])
                points[model] = metrics(experimental,values)
                jack[model] = metrics(experimental[jack_idx],values[jack_idx])
                for metric in draws[model]:
                    rows.append(dict(scope=scope,estimator=estimator,model=model,metric=metric,n_complexes=len(keep),
                                     point=float(points[model][metric]),**intervals(draws[model][metric],points[model][metric],jack[model][metric])))
            for a,b in itertools.combinations(models,2):
                for metric in draws[a]:
                    point = points[a][metric]-points[b][metric]
                    differences.append(dict(scope=scope,estimator=estimator,model_a=a,model_b=b,metric=metric,n_complexes=len(keep),
                                            point=float(point),**intervals(draws[a][metric]-draws[b][metric],point,jack[a][metric]-jack[b][metric])))
            for charge, group in [('neutral',[i for i in keep if PDB[i] not in CHARGED]),
                                  ('plus_one',[i for i in keep if PDB[i] in CHARGED])]:
                for model, values in models.items():
                    summary = metrics(EXP[group],values[group])
                    strata.append(dict(scope=scope,estimator=estimator,model=model,charge=charge,n_complexes=len(group),
                                       **{key: float(summary[key]) for key in ['MUE','RMSE','bias']}))
    for name, data in [('metrics',rows),('paired_differences',differences),('charge_strata',strata)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (OUT/f'{name}.tsv').open('w') as stream:
            writer=csv.DictWriter(stream,fields,delimiter='\t'); writer.writeheader(); writer.writerows(data)
    provenance=dict(seed=SEED,nboot=NBOOT,provisional=True,scopes={s:[PDB[i] for i in k] for s,k in scopes.items()},
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES+[OFF,HERE/'bootstrap_t1.py',HERE/'bootstrap_complexes.py',HERE.parent/'analysis/pairwise_ddg.py']},
                    experimental=dict(zip(PDB,EXP.tolist())),
                    predictions={e:{m:[float(v) if np.isfinite(v) else None for v in a] for m,a in models.items()} for e,models in predictions.items()})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for row in differences:
        if row['scope']=='common_complete' and row['metric']=='MUE':
            print(row['estimator'],row['model_a'],row['model_b'],*[round(row[k],3) for k in ['point','bca_low','bca_high']])
    print('Common set:',provenance['scopes']['common_complete'])


if __name__=='__main__':
    main()
