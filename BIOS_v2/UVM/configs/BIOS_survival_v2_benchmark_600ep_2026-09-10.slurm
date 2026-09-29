#!/bin/bash
#SBATCH --job-name=surv_BIOS_UVM_v2_benchmark_600ep_2026-09-10
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/survival_results/BIOS/v2_benchmark_600ep_2026-09-10/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/survival_results/BIOS/v2_benchmark_600ep_2026-09-10/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genacc_q
#
# Survival for BIOS on UVM, version v2_benchmark_600ep_2026-09-10
# Clusters:  /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/subtyping_results/BIOS/v2_benchmark_600ep_2026-09-10/clusters.csv
# Output:    /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/survival_results/BIOS/v2_benchmark_600ep_2026-09-10

cd /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/scripts/lib:$PYTHONPATH

for EP in OS PFI; do
  echo "=== analytic $EP: $(date) ==="
  /gpfs/home/ug25b/.conda/envs/bios/bin/python survival_analysis.py --cancer UVM --method BIOS \
      --version v2_benchmark_600ep_2026-09-10 --endpoint $EP || exit 1

  echo "=== permutation $EP: $(date) ==="
  /gpfs/home/ug25b/.conda/envs/bios/bin/python analysis/permutation_survival.py \
      --clusters /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/subtyping_results/BIOS/v2_benchmark_600ep_2026-09-10/clusters.csv \
      --survival /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/ground_truth/survival_endpoints_UVM_CDR.csv \
      --format cdr --endpoint $EP --max-perms 1000000 \
      > /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/survival_results/BIOS/v2_benchmark_600ep_2026-09-10/permutation_$EP.txt || exit 1
  tail -3 /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/UVM/results/survival_results/BIOS/v2_benchmark_600ep_2026-09-10/permutation_$EP.txt
done
echo "=== DONE $(date) ==="
