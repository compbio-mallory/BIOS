#!/bin/bash
#SBATCH --job-name=perm_timing
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/perm_timing_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/perm_timing_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
V=v2_benchmark_600ep_2026-09-10
cd $ROOT/scripts/lib/analysis
export PYTHONPATH=$ROOT/scripts/lib:$PYTHONPATH
echo "start $(date)"
/gpfs/home/ug25b/.conda/envs/bios/bin/python permutation_survival.py \
  --clusters $ROOT/UVM/results/subtyping_results/BIOS/$V/clusters.csv \
  --survival $ROOT/UVM/ground_truth/survival_endpoints_UVM_CDR.csv \
  --format cdr --endpoint OS --max-perms 50000
echo "end $(date)"
