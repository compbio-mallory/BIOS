#!/bin/bash
#SBATCH --job-name=aq_ucec_bd32
#SBATCH --output=logs/aq_ucec_bd32_%j.out
#SBATCH --error=logs/aq_ucec_bd32_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs save/model_aq_ucec_bd32

$PYBIN -u scripts/train/train_bioanchor.py \
    --head_type attention_query \
    --cancer_type UCEC --cluster_number 4 \
    --bio_dim 32 --n_anchors 50 \
    --bio_anchor_file data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv \
    --lambda_bio 0.1 --epochs 600 --batch_size 64 \
    --model_path save/model_aq_ucec_bd32
echo "done bd32 $(date)"
