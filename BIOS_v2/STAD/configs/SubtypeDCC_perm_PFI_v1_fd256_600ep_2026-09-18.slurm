#!/bin/bash
#SBATCH --job-name=perm_DCC_STAD_PFI
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/survival_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/survival_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/logs/%x_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# Permutation log-rank, SubtypeDCC STAD PFI, version v1_fd256_600ep_2026-09-18
# Clusters: /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/subtyping_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/clusters.csv

cd /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib/analysis
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib:${PYTHONPATH:-}
echo "start $(date)"
/gpfs/home/ug25b/.conda/envs/bios/bin/python permutation_survival.py \
    --clusters /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/subtyping_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/clusters.csv --survival /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/ground_truth/survival_endpoints_STAD_CDR.csv \
    --format cdr --endpoint PFI --max-perms 10000000 --cores 16 \
    > /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/survival_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/permutation_PFI.txt
echo "end $(date)"
tail -3 /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/STAD/results/survival_results/SubtypeDCC/v1_fd256_600ep_2026-09-18/permutation_PFI.txt
