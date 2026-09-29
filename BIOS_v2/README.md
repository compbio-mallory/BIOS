# BIOS

Biologically-anchored deep clustering for multi-omics cancer subtyping.
Built on Subtype-DCC (Zhao et al., Brief Bioinform 2023).

## Structure

    scripts/            shared code, same for every cancer, not edited per run
      modules/          model architecture (AE, network, bio-anchor heads)
      utils/            config loader, save helpers
      reference/        Hallmark gene sets, TCGA-CDR survival table, probemap
      slurm/            job template
      config_template.yaml
    <CANCER>/           one folder per cancer type
      raw/              omics (.fea), RNA expression, clinical matrix
      anchors/          GSVA bio-anchor files
      ground_truth/     subtype labels, survival endpoints
      configs/          current working configs
      results/<version>/  one folder per run set
    analysis/           investigations not tied to a single cancer

## Version history

| version | date | scope | config | key result |
|---|---|---|---|---|
| v0_benchmark_200ep | 2026-08-25 | 10 cancers | linear, H40, bio_dim 16, 200 ep | BRCA V=0.473; UCEC survival -log10p=6.10 |
| v0b_reproduction_2x2 | 2026-08-31 | BRCA, UCEC | 2x2 anchors x bio_dim + poster repro | poster reproduced within 0.018 |
| v1_ablation_600ep | 2026-09-03 | BRCA | 12 configs, 600 ep | best V 0.4722 (linear/16/4); best survival 2.426 (linear/16/40) |

## Conventions

- Epochs fixed at 600 (Subtype-DCC Table S1); only the last epoch is saved.
- Config file names encode all varied parameters. The slurm job name, log file,
  checkpoint folder, and cluster CSV all share that same name.
- `<CANCER>/configs/` holds current configs. `results/<version>/configs/` holds a
  frozen copy of what actually ran; those are never edited.
