#!/bin/bash
# make_BIOS_variant.sh — create config + slurm for a BIOS variant by COPYING the
# v2 benchmark files and changing only: version string, bio_dim, seed.
# Never overwrites: aborts if any target exists; writes with noclobber.
#
# Usage (from BIOS_v2 root):
#   bash scripts/subtyping/make_BIOS_variant.sh <new_version> <bio_dim> <seed> [dry|apply|submit] [CANCER]
#     dry    (default) print every path that would be created + the diff vs v2. Creates nothing.
#     apply  create the yaml, slurm, and empty results/logs dir.
#     submit sbatch the slurm files created by apply.
#   Optional CANCER restricts to one cancer (e.g. submit UVM first as a smoke test).
#
# Example:
#   bash scripts/subtyping/make_BIOS_variant.sh v2_biodim0_seed42_600ep_2026-09-22 0 42
set -u
NEWV=${1:?usage: <new_version> <bio_dim> <seed> [dry|apply|submit] [cancer]}
BIODIM=${2:?bio_dim missing}
SEED=${3:?seed missing}
MODE=${4:-dry}
ONLY=${5:-}

BASEV=v2_benchmark_600ep_2026-09-10
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

case "$MODE" in dry|apply|submit) ;; *) echo "mode must be dry|apply|submit"; exit 1;; esac
[ "$NEWV" = "$BASEV" ] && { echo "new version equals base version; refusing"; exit 1; }

make_yaml() {  # $1 = source yaml
  sed -e "s/${BASEV}/${NEWV}/g" \
      -e "s/^bio_dim:\( *\)16\$/bio_dim:\1${BIODIM}/" \
      -e "s/^seed:\( *\)42\$/seed:\1${SEED}/" "$1"
}
make_slurm() { # $1 = source slurm
  sed -e "s/${BASEV}/${NEWV}/g" "$1"
}

# ---------- checks (all modes) ----------
fail=0
for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  SRC_Y=$ROOT/$C/configs/BIOS_${BASEV}.yaml
  SRC_S=$ROOT/$C/configs/BIOS_${BASEV}.slurm
  DST_Y=$ROOT/$C/configs/BIOS_${NEWV}.yaml
  DST_S=$ROOT/$C/configs/BIOS_${NEWV}.slurm
  OUT=$ROOT/$C/results/subtyping_results/BIOS/$NEWV

  echo "================ $C ================"
  for f in "$SRC_Y" "$SRC_S"; do
    [ -f "$f" ] || { echo "  MISSING source: $f"; fail=1; }
  done
  [ $fail -eq 1 ] && continue

  if [ "$MODE" = "submit" ]; then
    for f in "$DST_Y" "$DST_S" "$OUT/logs"; do
      [ -e "$f" ] || { echo "  NOT CREATED YET (run apply first): $f"; fail=1; }
    done
    continue
  fi

  for f in "$DST_Y" "$DST_S" "$OUT"; do
    [ -e "$f" ] && { echo "  ALREADY EXISTS, will not touch: $f"; fail=1; }
  done

  # the generated yaml must contain the new values and no trace of the base version
  Y=$(make_yaml "$SRC_Y")
  echo "$Y" | grep -Eq "^bio_dim: *${BIODIM}\$" || { echo "  bio_dim line not found/replaced in $SRC_Y"; fail=1; }
  echo "$Y" | grep -Eq "^seed: *${SEED}\$"      || { echo "  seed line not found/replaced in $SRC_Y"; fail=1; }
  echo "$Y" | grep -q "$BASEV" && { echo "  base version still present in generated yaml"; fail=1; }
  make_slurm "$SRC_S" | grep -q "$BASEV" && { echo "  base version still present in generated slurm"; fail=1; }

  echo "  would create: $DST_Y"
  echo "  would create: $DST_S"
  echo "  would create: $OUT/logs/   (empty)"
  echo "  --- yaml diff vs v2 (only version, bio_dim, seed should change) ---"
  diff "$SRC_Y" <(make_yaml "$SRC_Y") | sed 's/^/    /'
  echo "  --- slurm diff vs v2 (only version string should change) ---"
  diff "$SRC_S" <(make_slurm "$SRC_S") | sed 's/^/    /'
done

if [ $fail -ne 0 ]; then echo; echo "ABORTED: problems above. Nothing was created or submitted."; exit 1; fi
[ "$MODE" = "dry" ] && { echo; echo "DRY RUN ONLY. Nothing created. Re-run with 'apply' to create."; exit 0; }

# ---------- apply ----------
if [ "$MODE" = "apply" ]; then
  set -o noclobber
  for C in $CANCERS; do
    [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
    SRC_Y=$ROOT/$C/configs/BIOS_${BASEV}.yaml
    SRC_S=$ROOT/$C/configs/BIOS_${BASEV}.slurm
    DST_Y=$ROOT/$C/configs/BIOS_${NEWV}.yaml
    DST_S=$ROOT/$C/configs/BIOS_${NEWV}.slurm
    OUT=$ROOT/$C/results/subtyping_results/BIOS/$NEWV
    make_yaml  "$SRC_Y" > "$DST_Y" || exit 1
    make_slurm "$SRC_S" > "$DST_S" || exit 1
    chmod +x "$DST_S"
    mkdir -p "$OUT/logs"
    echo "created $C: $(basename "$DST_Y"), $(basename "$DST_S"), $OUT/logs/"
  done
  echo; echo "Created. Inspect, then run with 'submit'."
  exit 0
fi

# ---------- submit ----------
for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  sbatch "$ROOT/$C/configs/BIOS_${NEWV}.slurm" && echo "submitted $C"
done
squeue -u "$USER" -o "%.10i %.40j %.8T"
