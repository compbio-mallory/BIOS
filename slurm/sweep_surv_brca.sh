#!/bin/bash
#SBATCH --job-name=sweep_surv_BRCA
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genacc_q
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
/gpfs/home/ug25b/.conda/envs/bios/bin/python sweep_survival.py \
  --model_dir save/model_BRCA_linear_gsva_H40_b16 --cancer BRCA --k 5 \
  --anchors data/bio_anchors/bio_anchors_BRCA_gsva_H40.csv \
  --clinical data/labels/brca_survival_cdr.csv \
  --endpoint OS
