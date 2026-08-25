#!/bin/bash
#SBATCH --job-name=baseline_ucec
#SBATCH --output=logs/baseline_ucec_%j.out
#SBATCH --error=logs/baseline_ucec_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs results save/model_baseline_ucec

$PYBIN -u scripts/train/train_baseline_OLD.py \
    --cancer_type UCEC --batch_size 32 --cluster_number 4 \
    --epochs 600 --model_path save/model_baseline_ucec
echo "training done $(date)"
