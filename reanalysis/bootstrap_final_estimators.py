"""Paired BCa comparison of completed paths under three estimators."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from bootstrap_complexes import CALC, EXP, PDB, SEED, NBOOT, metrics
from bootstrap_t1 import intervals

HERE=Path(__file__).resolve().parent
ANALYSIS=HERE.parent/'analysis'
OUT=HERE/'t6_final_estimators'

def main():
    OUT.mkdir(exist_ok=True)
    source=ANALYSIS/'mbar/abfe_legs_provenance.tsv'
    records=list(csv.DictReader(source.open(),delimiter='\t'))
    assert len(records)==160 and all(r['status']=='mbar_reanalysis' for r in records)
    raw_source=ANALYSIS/'mbar/mbar_legs.tsv'
    raw_records=list(csv.DictReader(raw_source.open(),delimiter='\t'))
    assert len(raw_records)==160 and all(not r.get('error') and float(r['n_states'])==float(r['expected']) for r in raw_records)
    legs={(r['model'],r['pdb'],r['leg']):r for r in raw_records}
    offsets={(r['model'],r['pdb']):float(r['dG_off']) for r in csv.DictReader((ANALYSIS/'abfe_legs.tsv').open(),delimiter='\t')}
    predictions={'all_frame_MBAR':{m:np.asarray(v) for m,v in CALC.items()}}
    for estimator,column in [('decorrelated_MBAR','dG'),('TI','dG_ti')]:
        predictions[estimator]={}
        for model in CALC:
            vals=[]
            for pdb in PDB:
                sol,elec,vdw,restr=[float(legs[model,pdb,leg][column]) for leg in ('ligand','coulomb','vdw','restr')]
                vals.append(sol-elec-vdw-restr-offsets[model,pdb])
            predictions[estimator][model]=np.array(vals)
    rows=[]; differences=[]
    for scope,keep in [('all_ten',list(range(10))),('without_5I8G',[i for i,p in enumerate(PDB) if p!='5I8G'])]:
        x=EXP[keep]
        idx=np.random.default_rng(SEED).integers(0,len(keep),size=(NBOOT,len(keep)))
        jidx=np.array([np.delete(np.arange(len(keep)),i) for i in range(len(keep))])
        for estimator,models in predictions.items():
            draws={};points={};jack={}
            for model,values in models.items():
                y=values[keep];draws[model]=metrics(x[idx],y[idx]);points[model]=metrics(x,y);jack[model]=metrics(x[jidx],y[jidx])
                for metric in draws[model]:
                    rows.append(dict(scope=scope,n_complexes=len(keep),estimator=estimator,model=model,metric=metric,point=float(points[model][metric]),**intervals(draws[model][metric],points[model][metric],jack[model][metric])))
            for a,b in itertools.combinations(models,2):
                for metric in draws[a]:
                    point=points[a][metric]-points[b][metric]
                    differences.append(dict(scope=scope,n_complexes=len(keep),estimator=estimator,model_a=a,model_b=b,metric=metric,point=float(point),**intervals(draws[a][metric]-draws[b][metric],point,jack[a][metric]-jack[b][metric])))
    for name,data in [('metrics.tsv',rows),('paired_differences.tsv',differences)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(data)
    sources=[source,raw_source,ANALYSIS/'abfe_legs.tsv',ANALYSIS/'pairwise_ddg.py',Path(__file__),HERE/'bootstrap_complexes.py',HERE/'bootstrap_t1.py']
    (OUT/'provenance.json').write_text(json.dumps(dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},seed=SEED,nboot=NBOOT,predictions={e:{m:v.tolist() for m,v in ms.items()} for e,ms in predictions.items()},note='All-frame predictions use manuscript CALC rounded to two decimals; TI/decorrelated legs use full-precision canonical mbar_legs.tsv. No provisional inputs.'),indent=2)+'\n')
    lines=['# Completed-data estimator sensitivity','', 'All 160 legs have reanalysis status. Paired whole-complex bootstrap, 20,000 draws, seed 20261006. BCa intervals describe composition of this benchmark with fixed experimental and calculated values; they do not quantify trajectory, experimental, or finite-size uncertainty. The all-frame estimates use manuscript CALC (two decimals); the other estimators use full-precision canonical mbar_legs.tsv.','', '| Scope | Estimator | Comparison | ΔMUE | BCa 95% interval |','|---|---|---|---:|---:|']
    for r in differences:
        if r['metric']=='MUE' and r['model_a']=='TIP3P':
            lines.append(f"| {r['scope']} | {r['estimator']} | {r['model_a']} − {r['model_b']} | {r['point']:.3f} | [{r['bca_low']:.3f}, {r['bca_high']:.3f}] |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__': main()
