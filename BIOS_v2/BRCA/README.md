# BRCA

TCGA breast cancer. n = 1031 (four-omics complete cases). K = 5 (PAM50).
Ground truth: PAM50 subtypes, 803 labeled patients.

## Data
- `raw/` — CN.fea, meth.fea, miRNA.fea, rna.fea; BRCA_HiSeqV2_full.tsv.gz
  (gene-symbol RNA used for GSVA); BRCA_clinicalMatrix.txt
- `anchors/` — bio_anchors_BRCA_gsva_H40.csv (40 filtered Hallmark pathways);
  bio_anchors_BRCA_gsva_decorr_M4.csv (4 decorrelated: inflammatory response,
  EMT, E2F targets, MYC targets V1); plus earlier variants
- `ground_truth/` — ground_truth_BRCA.csv (PAM50); brca_survival_cdr.csv
  (TCGA-CDR OS / PFI / DSS endpoints)

## Results

### v1_ablation_600ep_2026-09-03
12 configurations: 3 heads x bio_dim {4,16} x pathways {4,40}.
Fixed: 600 epochs, lambda_bio 0.1, batch 64, K=5, last epoch only.
Best V-measure 0.4722 (linear, bio_dim 16, 4 pathways).
Best survival -log10 p 2.426 (linear, bio_dim 16, 40 pathways).
Full numbers in ablation_table.csv.

### v0b_reproduction_2x2_200ep_2026-08-31
Reproduction of the poster configuration and 2x2 attribution of anchor count
versus bio_dim. Poster config (MLP, 4 decorrelated anchors, bio_dim 4, best
checkpoint at epoch 300, V=0.4931) reproduced at V=0.4801 with a fixed
200-epoch checkpoint. Checkpoints no longer retained.

### v0_benchmark_200ep_2026-08-25
Part of the 10-cancer benchmark run. V=0.4729, ARI=0.3626 against PAM50;
survival OS -log10 p = 1.69. Checkpoints no longer retained.
