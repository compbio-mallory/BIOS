#!/bin/bash
#SBATCH --job-name=aqv2_ucec_bd16
#SBATCH --output=logs/aqv2_ucec_bd16_%j.out
#SBATCH --error=logs/aqv2_ucec_bd16_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs save/model_aqv2_ucec_bd16
$PYBIN -u scripts/train/train_bioanchor.py \
    --head_type attention_query_v2 \
    --cancer_type UCEC --cluster_number 4 \
    --bio_dim 16 --n_anchors 50 \
    --bio_anchor_file data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv \
    --lambda_bio 0.1 --lambda_entropy 0.1 --epochs 600 --batch_size 64 \
    --model_path save/model_aqv2_ucec_bd16
echo "done $(date)"
