#!/bin/bash
# Run BIOS subtyping on all 9 benchmark cancers.
# Generates one config and one slurm file per cancer, then submits them.
#
# Usage:  bash scripts/subtyping/run_BIOS.sh <version>
# e.g.:   bash scripts/subtyping/run_BIOS.sh v2_benchmark_600ep_2026-09-15
set -u
VERSION=${1:?usage: run_BIOS.sh <version> [cancer]}
ONLY=${2:-}          # optional: run just this cancer

ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=BIOS

# cancer:K   K values follow Subtype-DCC / Subtype-GAN
CANCERS="BRCA:5 UCEC:4 LUAD:3 BLCA:5 KIRC:4 PAAD:2 SKCM:4 STAD:3 UVM:4"

for entry in $CANCERS; do
  C=${entry%%:*}
  K=${entry##*:}
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  OUT=$ROOT/$C/results/subtyping_results/$METHOD/$VERSION
  CFG=$ROOT/$C/configs/${METHOD}_${VERSION}.yaml
  SLURM=$ROOT/$C/configs/${METHOD}_${VERSION}.slurm
  mkdir -p $OUT/logs $ROOT/$C/configs

  # ---- config ----
  cat > $CFG << EOF
# ${METHOD} ${VERSION} — ${C}
cancer_type:      ${C}
cluster_number:   ${K}
head_type:        linear
bio_dim:          16
n_anchors:        40
bio_anchor_file:  ${ROOT}/${C}/anchors/bio_anchors_${C}_gsva_H40.csv
epochs:           600
lambda_bio:       0.1
batch_size:       64
seed:             42
model_path:       ${OUT}/checkpoint
EOF

  # ---- slurm ----
  cat > $SLURM << EOF
#!/bin/bash
#SBATCH --job-name=${METHOD}_${C}_${VERSION}
#SBATCH --output=${OUT}/logs/%x_%j.out
#SBATCH --error=${OUT}/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# ${METHOD} subtyping — ${C}, K=${K}, version ${VERSION}
# Config: ${CFG}
# Output: ${OUT}

ROOT=${ROOT}
PYBIN=${PYBIN}
cd \$ROOT/scripts/lib
export PYTHONPATH=\$ROOT/scripts/lib:\$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== config ==="
cat ${CFG}
echo "=== train: \$(date) ==="
\$PYBIN train_bioanchor.py --config ${CFG} || exit 1

echo "=== extract clusters: \$(date) ==="
\$PYBIN extract_clusters.py --cancer ${C} --version ${VERSION} \\
       --config ${CFG} --method ${METHOD} || exit 1

echo "=== DONE \$(date) ==="
EOF

  chmod +x $SLURM
  sbatch $SLURM
  echo "submitted $METHOD $C (K=$K)"
  echo "   config: $CFG"
  echo "   slurm:  $SLURM"
done

echo
squeue -u $USER -o "%.10i %.34j %.8T"