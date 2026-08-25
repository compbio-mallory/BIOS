#!/bin/bash
#SBATCH --job-name=bios_v3_ucec_hinge
#SBATCH --output=logs/v3_ucec_hinge_%j.out
#SBATCH --error=logs/v3_ucec_hinge_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
REPO=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs save/model_ucec_v3_H40_hinge results

echo "=== host: $(hostname) | start: $(date) ==="

echo "=== STEP 1: filter H50 -> H40 (cancer-relevant panel) ==="
$PYBIN pathway_filter.py \
    --in  data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv \
    --out data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv \
    || { echo "pathway filter failed"; exit 1; }

echo "=== STEP 1b: HARD GATE - anchors cover all omics patients, 40 cols, no NaN ==="
$PYBIN -c "
import pandas as pd, sys
a = pd.read_csv('data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv', index_col=0)
cn = pd.read_csv('../subtype_file/fea/UCEC/CN.fea', index_col=0)
miss = set(cn.columns) - set(a.index)
print('anchor shape:', a.shape)
print('overlap:', len(set(a.index) & set(cn.columns)), '| missing:', len(miss))
if a.shape[1] != 40: print('ABORT - expected 40 anchors, got', a.shape[1]); sys.exit(1)
if miss: print('ABORT - missing:', list(miss)[:10]); sys.exit(1)
if a.isnull().any().any(): print('ABORT - NaNs present'); sys.exit(1)
print('gate passed')
" || { echo "alignment gate failed"; exit 1; }

echo "=== STEP 2: train BIOS UCEC (attention_query_v3, H40, k=4) ==="
$PYBIN scripts/train/train_bioanchor.py \
    --head_type       attention_query_v3 \
    --cancer_type     UCEC \
    --batch_size      64 \
    --cluster_number  4 \
    --epochs          400 \
    --bio_dim         16 \
    --n_anchors       40 \
    --lambda_bio      0.1 \
    --lambda_entropy  0.02 \
    --lambda_div      0.05 \
    --bio_anchor_file data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv \
    --model_path      save/model_ucec_v3_H40_hinge

echo "=== DONE: $(date) ==="