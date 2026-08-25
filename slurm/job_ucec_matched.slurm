#!/bin/bash
#SBATCH --job-name=bios_ucec_matched
#SBATCH --output=logs/ucec_matched_%j.out
#SBATCH --error=logs/ucec_matched_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs save/model_ucec_matched results

echo "=== STEP 1: GSVA endometrial-matched anchors ==="
$PYBIN compute_gsva_anchors.py \
    --rna_file     data/raw/UCEC_HiSeqV2_symbols.tsv.gz \
    --cancer_type  UCEC_matched \
    --mode         ucec_matched \
    --out_file     data/bio_anchors/bio_anchors_UCEC_gsva_ucec_matched.csv || { echo "GSVA failed"; exit 1; }

echo "=== STEP 1b: alignment gate (must cover all omics patients) ==="
$PYBIN -c "
import pandas as pd, sys
a=pd.read_csv('data/bio_anchors/bio_anchors_UCEC_gsva_ucec_matched.csv')
cn=pd.read_csv('../subtype_file/fea/UCEC/CN.fea',index_col=0)
miss=set(cn.columns)-set(a['patient_id'])
print('overlap:',len(set(a['patient_id'])&set(cn.columns)),'| missing:',len(miss))
if miss: print('ABORT missing:',list(miss)[:10]); sys.exit(1)
print('NaN:',a.isnull().any().any())
" || { echo "alignment gate failed"; exit 1; }

echo "=== STEP 2: train BIOS UCEC (endometrial-matched anchors, k=4, 600ep) ==="
$PYBIN -u scripts/train/train_bioanchor.py \
    --head_type linear --cancer_type UCEC --batch_size 64 \
    --cluster_number 4 --epochs 600 --lambda_bio 0.1 --bio_dim 4 \
    --bio_anchor_file data/bio_anchors/bio_anchors_UCEC_gsva_ucec_matched.csv \
    --model_path save/model_ucec_matched
echo "=== DONE $(date) ==="
