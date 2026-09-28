#!/bin/bash
#SBATCH --job-name=bios_gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
REPO=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${CANCER:?pass --export=CANCER=...}"
: "${K:?pass --export=K=...}"
HEAD=${HEAD:-linear}
BIODIM=${BIODIM:-16}
EPOCHS=${EPOCHS:-200}
ANCHOR_MODE=${ANCHOR_MODE:-gsva_H40}
NANCH=${NANCH:-40}
ANCHORS=data/bio_anchors/bio_anchors_${CANCER}_${ANCHOR_MODE}.csv
MODELDIR=save/${SLURM_JOB_NAME}
mkdir -p logs $MODELDIR results

echo "=== gate: anchors cover omics ==="
$PYBIN -c "
import pandas as pd, sys
a = pd.read_csv('$ANCHORS', index_col=0)
cn = pd.read_csv('../subtype_file/fea/${CANCER}/CN.fea', index_col=0)
miss = set(cn.columns) - set(a.index)
if miss: sys.exit(f'ABORT: {len(miss)} omics patients lack anchors')
print('gate passed:', len(cn.columns), 'patients')
" || exit 1

echo "=== $CANCER | head=$HEAD | k=$K | $(date) ==="
$PYBIN scripts/train/train_bioanchor.py \
    --head_type $HEAD --cancer_type $CANCER \
    --batch_size 64 --cluster_number $K --epochs $EPOCHS \
    --bio_dim $BIODIM --n_anchors $NANCH --lambda_bio 0.1 \
    --bio_anchor_file $ANCHORS --model_path $MODELDIR
echo "=== DONE $(date) ==="

