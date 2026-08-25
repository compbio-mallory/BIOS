#!/bin/bash
#SBATCH --job-name=gsva_gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q
# usage: sbatch --job-name=gsva_LUAD --export=CANCER=LUAD slurm/run_gsva_generic.sh
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
: "${CANCER:?pass --export=CANCER=...}"

echo "=== $CANCER H50 GSVA | $(date) ==="
$PYBIN -u compute_gsva_anchors.py \
    --rna_file data/raw/${CANCER}_HiSeqV2_symbols.tsv.gz \
    --cancer_type $CANCER --mode H50 --processes 8 \
    --out_file data/bio_anchors/bio_anchors_${CANCER}_gsva_H50.csv \
  || { echo "$CANCER GSVA FAILED"; exit 1; }

echo "=== $CANCER H50 -> H40 filter ==="
$PYBIN pathway_filter.py \
    --in  data/bio_anchors/bio_anchors_${CANCER}_gsva_H50.csv \
    --out data/bio_anchors/bio_anchors_${CANCER}_gsva_H40.csv \
  || { echo "$CANCER filter FAILED"; exit 1; }

echo "=== $CANCER coverage gate vs omics cohort ==="
$PYBIN -c "
import pandas as pd, sys
a = pd.read_csv('data/bio_anchors/bio_anchors_${CANCER}_gsva_H40.csv', index_col=0)
cn = pd.read_csv('../subtype_file/fea/${CANCER}/CN.fea', index_col=0)
miss = set(cn.columns) - set(a.index)
print('anchors:', a.shape, '| omics patients:', len(cn.columns), '| missing:', len(miss))
if a.shape[1] != 40: sys.exit('ABORT: expected 40 anchors')
if a.isnull().any().any(): sys.exit('ABORT: NaNs')
if miss: print('WARNING: missing', len(miss), 'e.g.', list(miss)[:5])
"
echo "=== DONE $(date) ==="