# #!/bin/bash
# # Survival analysis for BIOS clusters, all 9 cancers.
# # Runs analytic (chi-square) and permutation log-rank, OS and PFI.
# #
# # Usage:  bash scripts/extract_survival/BIOS.sh <version> [cancer]
# set -u
# VERSION=${1:?usage: BIOS.sh <version> [cancer]}
# ONLY=${2:-}
# ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
# PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
# METHOD=BIOS
# MAXPERMS=10000000

# CANCERS="BRCA UCEC LUAD BLCA KIRC PAAD SKCM STAD UVM"

# for C in $CANCERS; do
#   [ -n "$ONLY" ] && [ "$C" != "$ONLY" ] && continue
#   IN=$ROOT/$C/results/subtyping_results/$METHOD/$VERSION/clusters.csv
#   OUT=$ROOT/$C/results/survival_results/$METHOD/$VERSION
#   SLURM=$ROOT/$C/configs/${METHOD}_survival_${VERSION}.slurm
#   mkdir -p $OUT/logs

#   if [ ! -f "$IN" ]; then
#     echo "SKIP $C: no clusters at $IN"
#     continue
#   fi

#   cat > $SLURM << EOF
# #!/bin/bash
# #SBATCH --job-name=surv_${METHOD}_${C}_${VERSION}
# #SBATCH --output=${OUT}/logs/%x_%j.out
# #SBATCH --error=${OUT}/logs/%x_%j.err
# #SBATCH --time=14-00:00:00
# #SBATCH --cpus-per-task=40
# #SBATCH --mem=16G
# #SBATCH --partition=genacc_q
# #
# # Survival for ${METHOD} on ${C}, version ${VERSION}
# # Clusters:  ${IN}
# # Output:    ${OUT}

# cd ${ROOT}/scripts/lib
# export PYTHONPATH=${ROOT}/scripts/lib:\$PYTHONPATH

# for EP in OS PFI; do
#   echo "=== analytic \$EP: \$(date) ==="
#   ${PYBIN} survival_analysis.py --cancer ${C} --method ${METHOD} \\
#       --version ${VERSION} --endpoint \$EP || exit 1

#   echo "=== permutation \$EP: \$(date) ==="
#   ${PYBIN} analysis/permutation_survival.py \\
#       --clusters ${IN} \\
#       --survival ${ROOT}/${C}/ground_truth/survival_endpoints_${C}_CDR.csv \\
#       --format cdr --endpoint \$EP --max-perms ${MAXPERMS} \\
#       > ${OUT}/permutation_\$EP.txt || exit 1
#   tail -3 ${OUT}/permutation_\$EP.txt
# done
# echo "=== DONE \$(date) ==="
# EOF

#   chmod +x $SLURM
#   sbatch $SLURM
#   echo "submitted survival $C  -> $OUT"
# done

# squeue -u $USER -o "%.10i %.42j %.8T"
#----------------------------------------------------------------------------
#!/bin/bash
# Survival analysis for BIOS clusters.
# Analytic runs inline; permutation is submitted as one job per endpoint.
#
# Usage:  bash scripts/extract_survival/BIOS.sh <version> [cancer]
set -u
VERSION=${1:?usage: BIOS.sh <version> [cancer]}
ONLY=${2:-}
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
METHOD=BIOS
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
    echo "SKIP $C: no clusters"
    continue
  fi

  for EP in OS PFI; do
    # analytic, inline (seconds)
    cd $ROOT/scripts/lib
    export PYTHONPATH=$ROOT/scripts/lib:${PYTHONPATH:-}
    $PYBIN survival_analysis.py --cancer $C --method $METHOD \
        --version $VERSION --endpoint $EP

    # permutation, one job per endpoint
    SLURM=$ROOT/$C/configs/${METHOD}_perm_${EP}_${VERSION}.slurm
    cat > $SLURM << EOF
#!/bin/bash
#SBATCH --job-name=perm_${C}_${EP}
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

squeue -u $USER -o "%.10i %.24j %.8T %.10M"