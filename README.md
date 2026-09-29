# BIOS — Bio-anchored Interpretable Omics Subtyping

Deep clustering for multi-omics cancer subtyping, built on Subtype-DCC
(Zhao et al., Briefings in Bioinformatics 2023), with a bio-anchor head that
ties part of the embedding to per-patient GSVA Hallmark pathway scores.

## Layout

- `BIOS_v2/` — current pipeline.
  - `scripts/subtyping/`, `scripts/extract_survival/`, `scripts/extract_label/` — launchers
  - `scripts/lib/` — pipeline code (training, cluster extraction, survival, clinical labels)
  - `scripts/lib/analysis/` — permutation survival, method comparison, sweeps
  - `{CANCER}/configs/` — one config + one slurm file per run
  - `{CANCER}/results/{subtyping,survival,label}_results/{METHOD}/{version}/`
  - `{CANCER}/anchors/`, `{CANCER}/ground_truth/`
  - `analysis/` — cross-method tables
- Every result file begins with `# source: <clusters.csv path>` for provenance.

## Earlier version

- Tag `BIOS_v1`: the complete v1 tree (Subtype-DCC-based layout), archived before BIOS_v2.
- Branches `MLP` and `attention`: v1 head-ablation variants (`main` at `BIOS_v1` is the linear head).

## Not included

Raw TCGA omics (`{CANCER}/raw/`), model checkpoints (`*.tar`), external
reference code (original Subtype-DCC, Rappoport & Shamir benchmark, other
methods), and scratch files.
