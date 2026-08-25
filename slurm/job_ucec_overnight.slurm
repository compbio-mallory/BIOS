#!/bin/bash
#SBATCH --job-name=bios_ucec_overnight
#SBATCH --output=logs/ucec_overnight_%j.out
#SBATCH --error=logs/ucec_overnight_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs save/model_ucec_gsva_decorr_M4 results

echo "=== STEP 1: GSVA on full symbol-mapped UCEC transcriptome ==="
$PYBIN compute_gsva_anchors.py \
    --rna_file     data/raw/UCEC_HiSeqV2_symbols.tsv.gz \
    --cancer_type  UCEC \
    --mode         decorr_M4 \
    --out_file     data/bio_anchors/bio_anchors_UCEC_gsva_decorr_M4.csv || { echo "GSVA failed"; exit 1; }

echo "=== STEP 1b: HARD GATE — anchors must cover all 510 omics patients ==="
$PYBIN -c "
import pandas as pd, sys
a = pd.read_csv('data/bio_anchors/bio_anchors_UCEC_gsva_decorr_M4.csv')
cn = pd.read_csv('../subtype_file/fea/UCEC/CN.fea', index_col=0)
miss = set(cn.columns) - set(a['patient_id'])
print('overlap:', len(set(a['patient_id']) & set(cn.columns)), '| missing:', len(miss))
if miss: print('ABORT — missing:', list(miss)[:10]); sys.exit(1)
print('NaN:', a.isnull().any().any())
" || { echo "alignment gate failed"; exit 1; }

echo "=== STEP 2: train BIOS UCEC (linear, decorr_M4, k=4) ==="
$PYBIN scripts/train/train_bioanchor.py \
    --head_type       linear \
    --cancer_type     UCEC \
    --batch_size      64 \
    --cluster_number  4 \
    --epochs          250 \
    --lambda_bio      0.1 \
    --bio_dim         4 \
    --bio_anchor_file data/bio_anchors/bio_anchors_UCEC_gsva_decorr_M4.csv \
    --model_path      save/model_ucec_gsva_decorr_M4

echo "=== DONE: $(date) ==="
