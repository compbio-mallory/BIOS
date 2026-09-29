#!/bin/bash
#SBATCH --job-name=BIOS_BLCA_v2_biodim0_seed21_600ep_2026-09-28
#SBATCH --output=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/results/subtyping_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/logs/%x_%j.out
#SBATCH --error=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/results/subtyping_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
#
# BIOS subtyping — BLCA, K=5, version v2_biodim0_seed21_600ep_2026-09-28
# Config: /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/configs/BIOS_v2_biodim0_seed21_600ep_2026-09-28.yaml
# Output: /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/results/subtyping_results/BIOS/v2_biodim0_seed21_600ep_2026-09-28

ROOT=/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd $ROOT/scripts/lib
export PYTHONPATH=$ROOT/scripts/lib:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== config ==="
cat /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/configs/BIOS_v2_biodim0_seed21_600ep_2026-09-28.yaml
echo "=== train: $(date) ==="
$PYBIN train_bioanchor.py --config /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/configs/BIOS_v2_biodim0_seed21_600ep_2026-09-28.yaml || exit 1

echo "=== extract clusters: $(date) ==="
$PYBIN extract_clusters.py --cancer BLCA --version v2_biodim0_seed21_600ep_2026-09-28 \
       --config /gpfs/research/fangroup/ug25b/BIOS/BIOS_v2/BLCA/configs/BIOS_v2_biodim0_seed21_600ep_2026-09-28.yaml --method BIOS || exit 1

echo "=== DONE $(date) ==="
