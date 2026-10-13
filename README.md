# Bromodomain ABFE with four water models

Simulation inputs and analysis scripts for the absolute binding free energy (ABFE) study of ten bromodomain–ligand complexes, each computed with the TIP3P, TIP4P, TIP5P and TIP7P water models.


The repository contains every input that defines the 3,000 production windows behind the reported values: starting structures, force-field and ligand parameters, GROMACS topologies, restraint definitions, index files and the parameter (`.mdp`) files. It also contains the scripts that turn the free-energy estimates into the tables and figures of the paper.


## Benchmark

| PDB | Bromodomain | Ligand net charge | ΔG_exp (kcal/mol) |
|---|---|---|---|
| 3MXF | BRD4-BD1 | 0 | −9.64 |
| 4ZQL | TRIM24 | 0 | −9.07 |
| 5DKH | SMARCA2 | 0 | −9.79 |
| 5I8G | CREBBP | 0 | −10.26 |
| 5IGL | TAF1L | 0 | −9.72 |
| 5LJ0 | ATAD2 | +1 | −10.99 |
| 6V1B | BRD9 | +1 | −10.14 |
| 7BL8 | BAZ2A | 0 | −9.49 |
| 7L9G | BRD2-BD2 | +1 | −8.49 |
| 8AG2 | BPTF | +1 | −9.38 |



## Leg folders:

| folder | stage | λ states |
|---|---|---|
| `complex_restraints` | Boresch restraints switched on in the complex | 12 |
| `complex_coulomb` | ligand charges switched off in the complex | 11 |
| `complex_vdw` | ligand Lennard-Jones interactions switched off in the complex (soft core) | 21 |
| `ligand_solvent` | ligand decoupled in water (11 Coulomb + 20 Lennard-Jones states) | 31 |

The Boresch restraints (one distance, two angles and three dihedrals) are defined in the `[ intermolecular_interactions ]` section of each complex `topol.top`. The binding free energy is

ΔG_bind = ΔG_solv − (ΔG_elec + ΔG_vdW + ΔG_restr) − ΔG_off,

where ΔG_off is the analytical restraint release and standard-state term (Boresch et al. 2003, eq 32), computed from the restraint parameters in the topologies (`reanalysis/check_restraints.py`).

## Simulation protocol

Each λ window was prepared independently from `solv.gro`:

1. steepest-descent minimisation (`em`);
2. 100 ps NVT with heavy-atom position restraints (`nvt`);
3. 100 ps NPT with restraints and the Berendsen barostat (`npt-restr`);
4. 100 ps unrestrained NPT with Parrinello–Rahman (`npt`);
5. 10 ns production (`md`).

The protocol uses the stochastic dynamics integrator with a 2 fs time step at 298.15 K, PME, 1.0 nm cutoffs and dispersion correction. Energy differences to all λ states were written every 0.2 ps. 

To regenerate a production run-input file, for example TIP3P/4ZQL, vdW state 5:

```bash
cd systems/TIP3P/4ZQL/complex_vdw
gmx grompp -f lambda_05/em_5.mdp -c solv.gro -p topol.top -o em_5.tpr -maxwarn 3
gmx mdrun -deffnm em_5
# then nvt -> npt-restr -> npt -> md, each starting from the previous stage, as in the leg's run script
```

## Analysis

Free energies were estimated with MBAR (alchemlyb 2.5.0, pymbar 4.2.0) after discarding the first 1 ns of each window. The per-leg results are in `analysis/mbar/mbar_legs.tsv`. Run `analysis/build_legs.py` first; it writes the assembled legs (`analysis/abfe_legs.tsv`) that the other scripts read. Derived tables, figures and summaries are not stored in the repository and are recreated by the scripts below.

| step | script | needs |
|---|---|---|
| MBAR from raw dhdl files | `reanalysis/mbar_recompute.py` | the dhdl files (`ABFE_RUN_ROOT`) |
| leg table | `analysis/build_legs.py` | `analysis/mbar/mbar_legs.tsv` |
| binding free energies, metrics, pairwise ΔΔG | `analysis/legs_analysis.py`, `pairwise_ddg.py`, `metrics.py`, `text_numbers.py` | `analysis/abfe_legs.tsv` |
| estimator sensitivity | `analysis/estimator_sensitivity.py` | `analysis/mbar/` |
| paired bootstrap intervals | `reanalysis/bootstrap_final_estimators.py` | `analysis/` |
| Boresch ΔG_off from topologies | `reanalysis/check_restraints.py` | `systems/` |
| charge (finite-size) correction | `analysis/charge_correction/rocklin_charge_correction_v2.py`, `assemble_charge_correction.py`, `analysis/charge_table.py` | production trajectories and GROMACS/APBS for the first step; stored results in `analysis/charge_correction/*.tsv` |
| ligand similarity (Table 1) | `analysis/similarity_table.py` | `analysis/ligands.tsv` |
| figures | `analysis/fig_*.py`, `analysis/fig1_tree.R` | `analysis/` |
| cost benchmark | `analysis/bench/` | the `.tpr` files (`ABFE_RUN_ROOT`) |

Scripts in `analysis/` are run from that folder and write tables and figures one level up (`../table*_rows.tex`, `../numbers_*.tex`, `../figures/`).

Environment:
- Python 3.11+ with NumPy, SciPy, pandas, Matplotlib and RDKit 2025.09;
- alchemlyb 2.5.0 and pymbar 4.2.0 for the re-analysis (set `JAX_PLATFORMS=cpu`);
- R with ape, ggplot2, ggtext, ggtree, RColorBrewer and tidytree for Figure 1;
- GROMACS and APBS 3.4.1 for the charge correction.

## Licence and citation

*These files are licensed under the [MIT License](https://opensource.org/licenses/MIT). Please cite the papers below when using these files.
