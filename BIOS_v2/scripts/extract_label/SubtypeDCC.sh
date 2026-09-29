#!/bin/bash
# Clinical parameter enrichment for Subtype-DCC clusters.
# Usage:  bash scripts/extract_label/SubtypeDCC.sh <version> [cancer]
set -u
VERSION=${1:?usage: SubtypeDCC.sh <version> [cancer]}
ONLY=${2:-}
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=SubtypeDCC

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