#!/bin/bash
# Survival analysis for Subtype-DCC clusters.
# Analytic runs inline; permutation is submitted as one job per endpoint.
#
# Usage:  bash scripts/extract_survival/SubtypeDCC.sh <version> [cancer]
set -u
VERSION=${1:?usage: SubtypeDCC.sh <version> [cancer]}
ONLY=${2:-}
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=SubtypeDCC
MAXPERMS=10000000
CORES=16

CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

for C in $CANCERS; do
  [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
  IN=$ROOT/$C/results/subtyping_results/$METHOD/$VERSION/clusters.csv
  OUT=$ROOT/$C/results/survival_results/$METHOD/$VERSION
  SURV=$ROOT/$C/ground_truth/survival_endpoints_${C}_CDR.csv
  mkdir -p $OUT/logs

  if [ ! -f "$IN" ]; then
    echo "SKIP $C: no clusters at $IN"
    continue
  fi

  for EP in OS PFI; do
    cd $ROOT/scripts/lib
    export PYTHONPATH=$ROOT/scripts/lib:${PYTHONPATH:-}
    $PYBIN survival_analysis.py --cancer $C --method $METHOD \
        --version $VERSION --endpoint $EP

    SLURM=$ROOT/$C/configs/${METHOD}_perm_${EP}_${VERSION}.slurm
    cat > $SLURM << EOF
#!/bin/bash
#SBATCH --job-name=perm_DCC_${C}_${EP}
#SBATCH --output=${OUT}/logs/%x_%j.out
#SBATCH --error=${OUT}/logs/%x_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# Permutation log-rank, ${METHOD} ${C} ${EP}, version ${VERSION}
# Clusters: ${IN}

cd ${ROOT}/scripts/lib/analysis
export PYTHONPATH=${ROOT}/scripts/lib:\${PYTHONPATH:-}
echo "start \$(date)"
${PYBIN} permutation_survival.py \\
    --clusters ${IN} --survival ${SURV} \\
    --format cdr --endpoint ${EP} --max-perms ${MAXPERMS} --cores ${CORES} \\
    > ${OUT}/permutation_${EP}.txt
echo "end \$(date)"
tail -3 ${OUT}/permutation_${EP}.txt
EOF
    chmod +x $SLURM
    sbatch $SLURM
  done
  echo "--- $C submitted (OS + PFI) ---"
done

squeue -u $USER -o "%.10i %.26j %.8T"