"""
migrate.py — build BIOS_v2 from the current Subtype-DCC tree.
COPIES only; the old tree is untouched. Run with --dry-run first.
"""
import argparse, os, shutil, glob

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--dest", default="/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2")
a = ap.parse_args()

SRC = "/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC"
OMICS = "/gpfs/research/fangroup/ug25b/BIOS/subtype_file/fea"
CANCERS = ["BRCA", "UCEC", "LUAD", "BLCA", "KIRC", "PAAD", "SKCM", "STAD", "GBM", "UVM"]

ops = []

def add(src, dst, kind="file"):
    ops.append((src, dst, kind))

# ---- scripts/ ----
for f in ["compute_gsva_anchors.py", "pathway_filter.py", "align_cohort.py",
          "survival_analysis.py", "cluster_enrichment.py", "validate_anchors.py",
          "probe_clusters_generic.py", "sweep_checkpoints.py", "sweep_survival.py",
          "plot_loss.py", "ablation_table.py", "dataloader.py", "eval_grid.py"]:
    if os.path.exists(f"{SRC}/{f}"):
        add(f"{SRC}/{f}", f"{a.dest}/scripts/{f}")

add(f"{SRC}/scripts/train/train_bioanchor.py", f"{a.dest}/scripts/train_bioanchor.py")
add(f"{SRC}/modules", f"{a.dest}/scripts/modules", "dir")
add(f"{SRC}/utils", f"{a.dest}/scripts/utils", "dir")
add(f"{SRC}/slurm/run_bios_generic.sh", f"{a.dest}/scripts/slurm/run_bios.sh")
add(f"{SRC}/config/config.yaml", f"{a.dest}/scripts/config_template.yaml")

# ---- shared reference data ----
add(f"{SRC}/data/genesets", f"{a.dest}/scripts/reference/genesets", "dir")
for p in [f"{SRC}/data/labels/TCGA-CDR.xlsx", f"{SRC}/data/raw/gencode_probemap.txt"]:
    if os.path.exists(p):
        add(p, f"{a.dest}/scripts/reference/{os.path.basename(p)}")

# ---- analysis artifacts ----
for p in (glob.glob(f"{SRC}/results/sweep_*.png") +
          glob.glob(f"{SRC}/results/loss_*.png") +
          glob.glob(f"{SRC}/results/survival/*.png")):
    add(p, f"{a.dest}/analysis/epoch_investigation_2026-09-01/{os.path.basename(p)}")

for p in (glob.glob(f"{SRC}/results/UCEC_gate_*") +
          glob.glob(f"{SRC}/results/v3_hinge_cluster_gates.csv") +
          glob.glob(f"{SRC}/results/*_comparison.csv") +
          glob.glob(f"{SRC}/results/*_full_metrics*.csv")):
    add(p, f"{a.dest}/analysis/attention_v3_investigation/{os.path.basename(p)}")

# ---- per-cancer ----
for C in CANCERS:
    base = f"{a.dest}/{C}"
    for om in ["CN.fea", "meth.fea", "miRNA.fea", "rna.fea"]:
        p = f"{OMICS}/{C}/{om}"
        if os.path.exists(p):
            add(p, f"{base}/raw/{om}")
    for pat in [f"{C}_HiSeqV2_symbols.tsv.gz", f"{C}_clinicalMatrix.txt",
                f"{C}_HiSeqV2_full.tsv.gz"]:
        p = f"{SRC}/data/raw/{pat}"
        if os.path.exists(p):
            add(p, f"{base}/raw/{pat}")
    for p in glob.glob(f"{SRC}/data/bio_anchors/bio_anchors_{C}_*.csv"):
        add(p, f"{base}/anchors/{os.path.basename(p)}")
    for p in glob.glob(f"{SRC}/data/ground_truth/ground_truth_{C}*.csv"):
        add(p, f"{base}/ground_truth/{os.path.basename(p)}")
    for p in glob.glob(f"{SRC}/data/labels/{C.lower()}_survival_cdr.csv"):
        add(p, f"{base}/ground_truth/{os.path.basename(p)}")

    V0 = f"{base}/results/v0_benchmark_200ep_2026-08-25"
    for p in (glob.glob(f"{SRC}/results/{C}_linear_H40_clusters.csv") +
              glob.glob(f"{SRC}/results/{C}_aligned_linear_H40_clusters.csv") +
              glob.glob(f"{SRC}/results/{C}_cluster_enrichment.csv") +
              glob.glob(f"{SRC}/results/{C}_*evaluation.csv")):
        add(p, f"{V0}/{os.path.basename(p)}")

    VR = f"{base}/results/v0b_reproduction_2x2_200ep_2026-08-31"
    for p in glob.glob(f"{SRC}/results/{C}_[ABCDE]_*_clusters.csv"):
        add(p, f"{VR}/{os.path.basename(p)}")

# ---- BRCA ablation v1 ----
V1 = f"{a.dest}/BRCA/results/v1_ablation_600ep_2026-09-03"
for p in glob.glob(f"{SRC}/configs/ablation_brca/*.yaml"):
    add(p, f"{a.dest}/BRCA/configs/{os.path.basename(p)}")
    add(p, f"{V1}/configs/{os.path.basename(p)}")
for p in glob.glob(f"{SRC}/logs/head_*.out"):
    add(p, f"{V1}/logs/{os.path.basename(p)}")
for p in glob.glob(f"{SRC}/results/head_*_clusters.csv"):
    add(p, f"{V1}/clusters/{os.path.basename(p)}")
if os.path.exists(f"{SRC}/results/ablation_brca_table.csv"):
    add(f"{SRC}/results/ablation_brca_table.csv", f"{V1}/ablation_table.csv")
for d in sorted(glob.glob(f"{SRC}/save/head_*/")):
    ck = f"{d}checkpoint_600.tar"
    if os.path.exists(ck):
        add(ck, f"{V1}/checkpoints/{os.path.basename(d.rstrip('/'))}_checkpoint_600.tar")

# ---- report ----
print(f"{'KIND':5s} {'SOURCE':70s} -> DEST")
print('-' * 140)
total = 0
for src, dst, kind in ops:
    if kind == "file" and os.path.exists(src):
        total += os.path.getsize(src)
    print(f"{kind:5s} {src[-68:]:70s} -> {dst[len(a.dest)+1:]}")
print('-' * 140)
print(f"{len(ops)} operations, ~{total/1e9:.1f} GB (dirs not counted)")

if a.dry_run:
    print("\nDRY RUN — nothing copied.")
else:
    for src, dst, kind in ops:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if kind == "dir":
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print(f"\nCopied to {a.dest}. Old tree untouched.")