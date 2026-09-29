# v1_ablation_600ep — BRCA head / bio_dim / pathway ablation

Run 2026-09-03. Slurm job IDs 13093017–13093028.

Purpose: isolate the contribution of head type, bio_dim, and anchor count,
changing one parameter at a time.

Fixed across all 12 runs: BRCA, K=5, 600 epochs, lambda_bio 0.1, batch 64,
final-epoch checkpoint only. Attention configs additionally set
lambda_entropy 0.02 and lambda_div 0.05.

Contents:
- configs/      the 12 configuration files (frozen copy)
- logs/         slurm output per run
- clusters/     per-run cluster assignments
- checkpoints/  final checkpoint per run
- ablation_table.csv   V-measure, ARI, survival -log10 p per configuration

Caveat: the runs were launched with parameters passed through slurm exports
rather than by reading these YAML files. All parameters match except `seed`:
the training script used seed 21 from config_template.yaml, while these files
list 42. Future runs will read the config files directly.
