#!/bin/bash
# Same-hardware throughput benchmark of the four water models (BOARD T11).
#
# Runs short segments of real production run inputs (md_5.tpr of each leg, i.e. lambda state 5)
# on this workstation (AMD Ryzen 7 5700X, NVIDIA GeForce RTX 4060) with one GROMACS build and
# identical mdrun settings, so that only the water model differs.
#   complex legs (coulomb, vdw, restr): 4ZQL, 8AG2, 7L9G   (all four models have these inputs)
#   solvent leg (ligand):               7BL8               (the only complex with all four locally)
# Each run: 50 000 steps (100 ps), timing reset at the halfway point (-resethway), no output
# of final coordinates. Two passes give an estimate of run-to-run timing noise.
#
# The TIP3P/5DKH vdW rerun (R1) is suspended (SIGSTOP) during the benchmark and resumed (SIGCONT)
# afterwards, including on error or interrupt, so the two jobs do not share the GPU.
set -u
GMX=${GMX:-gmx}          # GROMACS 2026.3 (reads every production tpr)
ROOT=${ABFE_RUN_ROOT:?set ABFE_RUN_ROOT to the run tree}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$HERE/work
mkdir -p "$WORK"

# --- suspend R1 -----------------------------------------------------------------------------
R1PIDS=""
for p in $(pgrep -f run_rerun.sh) $(pgrep -x gmx); do
    case "$(readlink /proc/$p/cwd 2>/dev/null)" in *vdw_rerun*) R1PIDS="$R1PIDS $p";; esac
done
resume() { [ -n "$R1PIDS" ] && kill -CONT $R1PIDS 2>/dev/null && echo "resumed R1 ($R1PIDS) $(date)"; }
trap resume EXIT
[ -n "$R1PIDS" ] && kill -STOP $R1PIDS && echo "suspended R1 ($R1PIDS) $(date)"
sleep 5

tpr() {  # model complex leg -> production tpr for lambda state 5
    local m=$1 c=$2 leg=$3 f
    f=$(find "$ROOT/$m/$c" "$ROOT/$m/ligandlar_tip4p/$c" -path "*${leg}*" -name md_5.tpr 2>/dev/null | grep -v '#' | sort | head -1)
    [ -z "$f" ] && [ "$leg" = restr ] && f=$(find "$ROOT/$m/$c" -ipath '*rest*' -name md_5.tpr 2>/dev/null | sort | head -1)
    echo "$f"
}

echo -e "pass\tmodel\tcomplex\tleg\tns_per_day\tatoms\ttpr" > "$HERE/bench_raw.tsv"
for pass in 1 2; do
  for c_leg in 4ZQL:coulomb 4ZQL:vdw 4ZQL:restr 8AG2:coulomb 8AG2:vdw 8AG2:restr \
               7L9G:coulomb 7L9G:vdw 7L9G:restr 7BL8:ligand; do
    c=${c_leg%%:*}; leg=${c_leg##*:}
    for m in TIP3P TIP4P TIP5P TIP7P; do
        f=$(tpr $m $c $leg)
        [ -z "$f" ] && { echo "no tpr for $m $c $leg"; continue; }
        d="$WORK/p${pass}_${m}_${c}_${leg}"; mkdir -p "$d"; cd "$d" || exit 1
        "$GMX" mdrun -s "$f" -deffnm bench -nsteps 50000 -resethway -noconfout \
               -ntmpi 1 -ntomp 16 -pin on > mdrun.out 2>&1
        nsd=$(awk '/^Performance:/{print $2}' bench.log)
        atoms=$("$GMX" dump -s "$f" 2>/dev/null | awk '/^ *natoms/{print $3; exit}')
        echo -e "$pass\t$m\t$c\t$leg\t${nsd:-NA}\t${atoms:-NA}\t$f" | tee -a "$HERE/bench_raw.tsv"
        rm -f bench.xtc bench.trr bench.edr bench.cpt bench_prev.cpt bench.xvg   # keep only logs
    done
  done
done
echo "benchmark done $(date)"
