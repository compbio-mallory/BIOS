#!/bin/bash
# Clinical parameter enrichment for BIOS clusters, all 9 cancers.
# Follows Rappoport & Shamir (2018): 6 parameters, Bonferroni, count significant.
#
# Usage:  bash scripts/extract_label/BIOS.sh <version> [cancer]
set -u
VERSION=${1:?usage: BIOS.sh <version> [cancer]}
ONLY=${2:-}
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=BIOS

CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  IN=$ROOT/$C/results/subtyping_results/$METHOD/$VERSION/clusters.csv
  if [ ! -f "$IN" ]; then
    echo "SKIP $C: no clusters at $IN"
    continue
  fi
  cd $ROOT/scripts/lib
  export PYTHONPATH=$ROOT/scripts/lib:${PYTHONPATH:-}
  $PYBIN extract_labels.py --cancer $C --method $METHOD --version $VERSION
  echo
done