"""
align_cohort.py — write .fea copies restricted to patients with GSVA anchors.

Originals untouched; output goes to ../subtype_file/fea/{CANCER}_aligned/.
Use for cancers where a few omics patients lack RNA/anchors (BLCA: 1, GBM: 7).

Usage (from Subtype-DCC repo root):
    python align_cohort.py --cancer BLCA
    python align_cohort.py --cancer GBM

Then symlink the anchor file to the aligned name and submit training with
CANCER={CANCER}_aligned (see notes printed at the end).
"""
import argparse
import os

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--cancer", required=True, help="e.g. BLCA or GBM")
args = p.parse_args()

base = f"../subtype_file/fea/{args.cancer}"
outd = f"../subtype_file/fea/{args.cancer}_aligned"
anchor_fp = f"data/bio_anchors/bio_anchors_{args.cancer}_gsva_H40.csv"

if not os.path.isdir(base):
    raise SystemExit(f"omics dir not found: {base} (run from repo root?)")
if not os.path.exists(anchor_fp):
    raise SystemExit(f"anchor file not found: {anchor_fp}")

os.makedirs(outd, exist_ok=True)
anch = pd.read_csv(anchor_fp, index_col=0)

total_dropped = None
for f in ["CN.fea", "meth.fea", "miRNA.fea", "rna.fea"]:
    df = pd.read_csv(f"{base}/{f}", header=0, index_col=0, sep=",")
    keep = [c for c in df.columns if c in anch.index]
    dropped = [c for c in df.columns if c not in anch.index]
    df[keep].to_csv(f"{outd}/{f}")
    print(f"{f}: {len(df.columns)} -> {len(keep)} patients "
          f"(dropped {len(dropped)}: {dropped[:7]})")
    if total_dropped is None:
        total_dropped = set(dropped)
    elif set(dropped) != total_dropped:
        print("  WARNING: dropped set differs across omics files — check cohorts!")

print(f"\nDone -> {outd}")
print("Next steps:")
print(f"  cd data/bio_anchors && ln -sf bio_anchors_{args.cancer}_gsva_H40.csv "
      f"bio_anchors_{args.cancer}_aligned_gsva_H40.csv && cd ../..")
print(f"  sbatch --job-name=bios_{args.cancer} "
      f"--export=CANCER={args.cancer}_aligned,K=<K> slurm/run_bios_generic.sh")