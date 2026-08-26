# BIOS
Bio-anchored Interpretable Omics Subtyping

## Branch: attention (ablation)
Attention-based bio-anchor head (v3: cluster-conditioned queries + pathway gate).
Run: --head_type attention_query_v3 (see slurm/run_v3_ucec_hinge.sh)
Diagnostics: probe_v3_gate.py, eval_v3_proper.py
Result: gate stable (H=1.78) but not enrichment-aligned (rho=-0.10); see ablation section.
