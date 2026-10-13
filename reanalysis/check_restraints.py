"""Recompute Boresch restraint release corrections from actual topology terms.

Run with Python 3; uses only the standard library. kJ/mol, nm, radians;
the reported correction is in kcal/mol. Checks all topol.top variants.
"""
import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'systems'
RT = 0.00831446261815324 * 298.15
V0 = 1.6605390671738466  # nm^3 per molecule at 1 mol/L


def restraint(path):
    text = path.read_text(errors='replace')
    match = re.search(r'\[\s*intermolecular_interactions\s*\]', text)
    if not match:
        return None
    terms = {'bonds': [], 'angles': [], 'dihedrals': []}
    section = ''
    for line in text[match.end():].splitlines():
        line = line.split(';', 1)[0].strip()
        if line.startswith('['):
            section = line.strip('[] ').strip()
        elif line and section in terms and not line.startswith('#'):
            terms[section].append(line.split())
    if [len(terms[k]) for k in terms] != [1, 2, 3]:
        return None
    # One bond function 6, two angle function 1, three harmonic dihedral function 2.
    if terms['bonds'][0][2] != '6' or any(t[3] != '1' for t in terms['angles']) or any(t[4] != '2' for t in terms['dihedrals']):
        raise ValueError(f'Unsupported restraint functions: {path}')
    b = terms['bonds'][0]
    r0, kr = float(b[5]), float(b[6])
    a = [(math.radians(float(t[6])), float(t[7])) for t in terms['angles']]
    kd = [float(t[8]) for t in terms['dihedrals']]
    if kr <= 0 or any(k <= 0 for _, k in a) or any(k <= 0 for k in kd):
        raise ValueError(f'Nonpositive force constant: {path}')
    logratio = (math.log(8 * math.pi**2 * V0)
                + 0.5 * math.log(kr * a[0][1] * a[1][1] * math.prod(kd))
                - math.log(r0**2 * math.sin(a[0][0]) * math.sin(a[1][0]))
                - 3 * math.log(2 * math.pi * RT))
    return r0, a[0][0], a[1][0], -RT * logratio / 4.184


def main():
    with (HERE.parent / 'analysis/abfe_legs.tsv').open() as stream:
        expected = {(r['model'], r['pdb']): float(r['dG_off']) for r in csv.DictReader(stream, delimiter='\t')}
    rows = []
    missing = []
    for (model, pdb), target in sorted(expected.items()):
        folder = ROOT / model / pdb
        count = 0
        for path in sorted(folder.rglob('topol.top')):
            result = restraint(path)
            if result is None:
                continue
            r0, theta_a, theta_b, value = result
            rows.append(dict(model=model, pdb=pdb, source=str(path), r_nm=r0,
                             theta_a_deg=math.degrees(theta_a), theta_b_deg=math.degrees(theta_b),
                             computed_kcal=value, stored_kcal=target, difference_kcal=value-target))
            count += 1
        if count == 0:
            missing.append(f'{model}/{pdb}')
    with (HERE / 'restraint_corrections.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Boresch standard-state correction check', '',
             'Computed from all readable `topol.top` files containing six supported intermolecular restraint terms.', '',
             'Expression: −RT ln[8π² V° √(kr kθA kθB kφA kφB kφC) / '
             '(r0² sinθA sinθB (2πRT)³)], T = 298.15 K, V° = 1.660539 nm³.', '',
             'This is the rigid-rotator/harmonic-oscillator approximation to restraint release '
             'to the 1 M standard state. The standard volume and orientational factor are already included. '
             'Subtracting a negative release correction in the project cycle adds a positive contribution to ΔGbind.', '',
             'Source: Boresch et al. 2003, https://doi.org/10.1021/jp0217839; '
             'Boresch 2024, https://doi.org/10.1021/acs.jcim.4c00442.', '',
             'The TSV checks numerical consistency, not the history of the original calculation or '
             'whether an additional unrecorded term was applied elsewhere.', '',
             '| Model | PDB | Topologies checked | Computed range | Stored | Max abs difference |',
             '|---|---|---:|---:|---:|---:|']
    for key, target in sorted(expected.items()):
        found = [r['computed_kcal'] for r in rows if (r['model'], r['pdb']) == key]
        if found:
            lines.append(f'| {key[0]} | {key[1]} | {len(found)} | {min(found):.5f} to {max(found):.5f} | {target:.5f} | {max(abs(v-target) for v in found):.5f} |')
    lines += ['', 'No readable restraint topology: ' + (', '.join(missing) if missing else 'none') + '.']
    (HERE / 'RESTRAINT_EVIDENCE.md').write_text('\n'.join(lines) + '\n')
    print(f'Checked {len(rows)} topologies. Missing: {missing}')
    print(f'Maximum difference from stored correction: {max(abs(r["difference_kcal"]) for r in rows):.6f} kcal/mol')


if __name__ == '__main__':
    main()
