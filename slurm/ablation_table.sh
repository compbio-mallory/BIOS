#!/bin/bash
#SBATCH --job-name=ablation_table
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genacc_q
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
/gpfs/home/ug25b/.conda/envs/bios/bin/python ablation_table.py
