#!/bin/bash
#SBATCH --job-name=perm_UCEC
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/analysis/permutation_test/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/analysis/permutation_test/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genacc_q
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
cd $ROOT/scripts/lib/analysis
/gpfs/home/ug25b/.conda/envs/bios/bin/python permutation_survival.py \
  --clusters $ROOT/UCEC/results/subtyping_results/BIOS/v0_benchmark_200ep_2026-08-25/UCEC_linear_H40_clusters.csv \
  --survival /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC/data/labels/ucec_survival_cdr.csv \
  --format cdr --max-perms 100000
