#!/bin/bash
#SBATCH --job-name=DCC_SKCM_fd128
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/SKCM/results/subtyping_results/SubtypeDCC/v1_fd128_600ep_2026-09-18/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/SKCM/results/subtyping_results/SubtypeDCC/v1_fd128_600ep_2026-09-18/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# Subtype-DCC, SKCM, feature_dim 128, 600 epochs
# Their code: /gpfs/research/fangroup/ug25b/BIOS/reference_data/Subtype-DCC/Subtype-DCC
# K comes from their cancer_dict in train.py

cd /gpfs/research/fangroup/ug25b/BIOS/reference_data/Subtype-DCC/Subtype-DCC
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results save

echo "=== DCC SKCM fd=128 | $(date) ==="
/gpfs/home/ug25b/.conda/envs/bios/bin/python -u train.py --cancer_type SKCM --feature_dim 128 || exit 1

echo "=== convert clusters ==="
/gpfs/home/ug25b/.conda/envs/bios/bin/python /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib/convert_dcc_clusters.py \
    --cancer SKCM --version v1_fd128_600ep_2026-09-18 \
    --dcc_file /gpfs/research/fangroup/ug25b/BIOS/reference_data/Subtype-DCC/Subtype-DCC/results/SKCM_fd128.dcc \
    --config_note "feature_dim=128, 600ep, batch 64, lr 3e-4, seed 21" || exit 1

echo "=== DONE $(date) ==="
