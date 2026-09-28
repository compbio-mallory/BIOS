#!/bin/bash
#SBATCH --job-name=sweep_UCEC_H40
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genacc_q
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
PY=/gpfs/home/ug25b/.conda/envs/bios/bin/python
$PY sweep_checkpoints.py --model_dir save/model_UCEC_linear_gsva_H40_b16 \
  --cancer UCEC --k 4 --anchors data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv
$PY sweep_survival.py --model_dir save/model_UCEC_linear_gsva_H40_b16 \
  --cancer UCEC --k 4 --anchors data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv \
  --clinical data/labels/ucec_survival_cdr.csv --endpoint OS
