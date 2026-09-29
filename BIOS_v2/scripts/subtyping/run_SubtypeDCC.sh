#!/bin/bash
# Subtype-DCC subtyping on the 9 benchmark cancers.
# Usage:  bash scripts/subtyping/run_SubtypeDCC.sh <feature_dim> [cancer]
#   e.g.: bash scripts/subtyping/run_SubtypeDCC.sh 128
set -u
FD=${1:?usage: run_SubtypeDCC.sh <feature_dim> [cancer]}
ONLY=${2:-}
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
DCC=/gpfs/research/fangroup/ug25b/BIOS/reference_data/Subtype-DCC/Subtype-DCC
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=SubtypeDCC
VERSION=v1_fd${FD}_600ep_$(date +%Y-%m-%d)

CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  OUT=$ROOT/$C/results/subtyping_results/$METHOD/$VERSION
  SLURM=$ROOT/$C/configs/${METHOD}_fd${FD}_${VERSION}.slurm
  mkdir -p $OUT/logs $ROOT/$C/configs

  cat > $SLURM << EOF
#!/bin/bash
#SBATCH --job-name=DCC_${C}_fd${FD}
#SBATCH --output=${OUT}/logs/%x_%j.out
#SBATCH --error=${OUT}/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# Subtype-DCC, ${C}, feature_dim ${FD}, 600 epochs
# Their code: ${DCC}
# K comes from their cancer_dict in train.py

cd ${DCC}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results save

echo "=== DCC ${C} fd=${FD} | \$(date) ==="
${PYBIN} -u train.py --cancer_type ${C} --feature_dim ${FD} || exit 1

echo "=== convert clusters ==="
${PYBIN} ${ROOT}/scripts/lib/convert_dcc_clusters.py \\
    --cancer ${C} --version ${VERSION} \\
    --dcc_file ${DCC}/results/${C}_fd${FD}.dcc \\
    --config_note "feature_dim=${FD}, 600ep, batch 64, lr 3e-4, seed 21" || exit 1

echo "=== DONE \$(date) ==="
EOF

  chmod +x $SLURM
  sbatch $SLURM
  echo "submitted $METHOD $C (fd=$FD) -> $OUT"
done

echo "version: $VERSION"
squeue -u $USER -o "%.10i %.24j %.8T"