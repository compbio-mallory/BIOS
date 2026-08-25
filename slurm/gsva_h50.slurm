#!/bin/bash
#SBATCH --job-name=gsva_h50
#SBATCH --output=logs/gsva_h50_%j.out
#SBATCH --error=logs/gsva_h50_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "start $(date) on $(hostname)"

echo "=== UCEC H50 GSVA (50 pathways) ==="
$PYBIN -u compute_gsva_anchors.py \
    --rna_file data/raw/UCEC_HiSeqV2_symbols.tsv.gz \
    --cancer_type UCEC --mode H50 --processes 8 \
    --out_file data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv || echo "UCEC FAILED"

echo "=== BRCA H50 GSVA (50 pathways) ==="
$PYBIN -u compute_gsva_anchors.py \
    --rna_file data/raw/BRCA_HiSeqV2_full.tsv.gz \
    --cancer_type BRCA --mode H50 --processes 8 \
    --out_file data/bio_anchors/bio_anchors_BRCA_gsva_H50.csv || echo "BRCA FAILED"

echo "done $(date)"
