#!/bin/bash
#SBATCH --job-name=perm_KIRC_OS
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/survival_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/survival_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/logs/%x_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# Permutation log-rank, BIOS KIRC OS, version v2_biodim0_seed21_600ep_2026-09-28
# Clusters: /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/subtyping_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/clusters.csv

cd /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib/analysis
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib:${PYTHONPATH:-}
echo "start $(date)"
/gpfs/home/ug25b/.conda/envs/bios/bin/python permutation_survival.py \
    --clusters /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/subtyping_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/clusters.csv --survival /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/ground_truth/survival_endpoints_KIRC_CDR.csv \
    --format cdr --endpoint OS --max-perms 10000000 --cores 16 \
    > /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/survival_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/permutation_OS.txt
echo "end $(date)"
tail -3 /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/KIRC/results/survival_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/permutation_OS.txt
