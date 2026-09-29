"""
restructure.py — reorganize BIOS_v2 into the method/version layout.
MOVES within BIOS_v2 (the old Subtype-DCC tree is untouched either way).
Run with --dry-run first.
"""
import argparse, os, shutil, glob

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

ROOT = "/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2"
CANCERS = ["BRCA", "UCEC", "LUAD", "BLCA", "KIRC", "PAAD", "SKCM", "STAD", "UVM", "GBM"]

ops = []          # (src, dst)
mkdirs = []       # empty dirs to create

def mv(src, dst):
    ops.append((src, dst))

# ---- scripts: python into lib/, analysis tools into lib/analysis/ ----
LIB = f"{ROOT}/scripts/lib"
for f in ["train_bioanchor.py", "dataloader.py", "paths.py", "survival_analysis.py",
          "cluster_enrichment.py", "compute_gsva_anchors.py", "pathway_filter.py",
          "align_cohort.py", "probe_clusters_generic.py", "validate_anchors.py",
          "config_template.yaml"]:
    p = f"{ROOT}/scripts/{f}"
    if os.path.exists(p):
        mv(p, f"{LIB}/{f}")

for f in ["sweep_checkpoints.py", "sweep_survival.py", "plot_loss.py",
          "eval_grid.py", "ablation_table.py"]:
    p = f"{ROOT}/scripts/{f}"
    if os.path.exists(p):
        mv(p, f"{LIB}/analysis/{f}")

for d in ["modules", "utils", "reference"]:
    p = f"{ROOT}/scripts/{d}"
    if os.path.isdir(p):
        mv(p, f"{LIB}/{d}")

# slurm template -> keep as a reference under lib
if os.path.exists(f"{ROOT}/scripts/slurm/run_bios.sh"):
    mv(f"{ROOT}/scripts/slurm/run_bios.sh", f"{LIB}/slurm_template.sh")

mkdirs += [f"{ROOT}/scripts/subtyping",
           f"{ROOT}/scripts/extract_survival",
           f"{ROOT}/scripts/extract_label"]

# ---- per-cancer results: existing versions -> subtyping_results/BIOS/ ----
for C in CANCERS:
    res = f"{ROOT}/{C}/results"
    if not os.path.isdir(res):
        continue
    for v in sorted(os.listdir(res)):
        vp = f"{res}/{v}"
        if os.path.isdir(vp) and v.startswith("v"):
            mv(vp, f"{res}/subtyping_results/BIOS/{v}")
    mkdirs += [f"{res}/survival_results/BIOS",
               f"{res}/label_results/BIOS"]

# ---- report ----
print("MOVE")
for s, d in ops:
    print(f"  {s[len(ROOT)+1:]}\n    -> {d[len(ROOT)+1:]}")
print("\nCREATE (empty)")
for d in mkdirs:
    print(f"  {d[len(ROOT)+1:]}")
print(f"\n{len(ops)} moves, {len(mkdirs)} new dirs")

if a.dry_run:
    print("\nDRY RUN — nothing changed.")
else:
    for d in mkdirs:
        os.makedirs(d, exist_ok=True)
    for s, d in ops:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.move(s, d)
    print("\nDone.")