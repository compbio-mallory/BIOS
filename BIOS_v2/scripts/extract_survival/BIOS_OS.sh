#!/bin/bash
# BIOS_OS.sh — OVERALL SURVIVAL ONLY for a BIOS version (no PFI, per professor).
#   analytic: survival_analysis.py (log-rank chi2 + Cox + KM), runs inline, seconds
#   permutation: slurm job COPIED from the v2 benchmark perm-OS slurm, only the version string changed
# Never overwrites: aborts if any target exists; writes with noclobber.
#
# Usage (from BIOS_v2 root):
#   bash scripts/extract_survival/BIOS_OS.sh <version> [dry|apply|submit] [CANCER]
#     dry    (default) checks, prints every path, shows slurm diff vs v2. Creates nothing.
#     apply  creates perm slurm + survival_results/<version>/logs, runs analytic OS inline.
#     submit sbatch the perm-OS slurm files (refuses if permutation_OS.txt already exists).
set -u
V=${1:?usage: <version> [dry|apply|submit] [cancer]}
MODE=${2:-dry}
ONLY=${3:-}

BASEV=v2_benchmark_600ep_2026-09-10
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

case "$MODE" in dry|apply|submit) ;; *) echo "mode must be dry|apply|submit"; exit 1;; esac
[ "$V" = "$BASEV" ] && { echo "version equals base version; refusing"; exit 1; }

# ---------- checks (all modes) ----------
fail=0
for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  IN=$ROOT/$C/results/subtyping_results/BIOS/$V/clusters.csv
  SRC=$ROOT/$C/configs/BIOS_perm_OS_${BASEV}.slurm
  DST=$ROOT/$C/configs/BIOS_perm_OS_${V}.slurm
  OUT=$ROOT/$C/results/survival_results/BIOS/$V

  echo "================ $C ================"
  [ -f "$IN" ]  || { echo "  MISSING clusters: $IN"; fail=1; }
  [ -f "$SRC" ] || { echo "  MISSING template: $SRC"; fail=1; continue; }

  if [ "$MODE" = "submit" ]; then
    [ -f "$DST" ] || { echo "  NOT CREATED YET (run apply first): $DST"; fail=1; }
    [ -e "$OUT/permutation_OS.txt" ] && { echo "  permutation_OS.txt already exists (already submitted?): $OUT"; fail=1; }
    continue
  fi

  [ -e "$DST" ] && { echo "  ALREADY EXISTS, will not touch: $DST"; fail=1; }
  [ -e "$OUT" ] && { echo "  ALREADY EXISTS, will not touch: $OUT"; fail=1; }
  sed "s/${BASEV}/${V}/g" "$SRC" | grep -q "$BASEV" && { echo "  base version still present in generated slurm"; fail=1; }

  echo "  would create: $DST"
  echo "  would create: $OUT/logs/"
  echo "  apply would write (analytic, inline): $OUT/survival_OS.csv, cox_OS.csv, KM_OS.png"
  echo "  submit would write (permutation job): $OUT/permutation_OS.txt"
  echo "  --- perm slurm diff vs v2 (only version string should change) ---"
  diff "$SRC" <(sed "s/${BASEV}/${V}/g" "$SRC") | sed 's/^/    /'
done

if [ $fail -ne 0 ]; then echo; echo "ABORTED: problems above. Nothing was created or submitted."; exit 1; fi
[ "$MODE" = "dry" ] && { echo; echo "DRY RUN ONLY. Nothing created. Re-run with 'apply'."; exit 0; }

# ---------- apply ----------
if [ "$MODE" = "apply" ]; then
  set -o noclobber
  cd "$ROOT/scripts/lib" || exit 1
  export PYTHONPATH=$ROOT/scripts/lib:${PYTHONPATH:-}
  for C in $CANCERS; do
    [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
    SRC=$ROOT/$C/configs/BIOS_perm_OS_${BASEV}.slurm
    DST=$ROOT/$C/configs/BIOS_perm_OS_${V}.slurm
    OUT=$ROOT/$C/results/survival_results/BIOS/$V
    sed "s/${BASEV}/${V}/g" "$SRC" > "$DST" || exit 1
    chmod +x "$DST"
    mkdir -p "$OUT/logs"
    echo "--- $C analytic OS ---"
    $PYBIN survival_analysis.py --cancer "$C" --method BIOS --version "$V" --endpoint OS \
      || { echo "analytic OS failed for $C; stopping"; exit 1; }
  done
  echo; echo "Created and analytic OS done. Inspect, then run with 'submit'."
  exit 0
fi

# ---------- submit ----------
for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  sbatch "$ROOT/$C/configs/BIOS_perm_OS_${V}.slurm" && echo "submitted perm OS $C"
done
squeue -u "$USER" -o "%.10i %.24j %.8T %.10M"