"""
convert_dcc_clusters.py — Subtype-DCC output -> standard clusters.csv.

Their train.py writes results/{CANCER}.dcc with columns sample_name, dcc
(clusters 1-indexed). This converts to patient_id, cluster (0-indexed) with
the provenance header used across the pipeline.

Usage:
  python convert_dcc_clusters.py --cancer BRCA --version v1_fd128_600ep_2026-09-15 \
      --dcc_file /path/to/results/BRCA.dcc
"""
import argparse, os, sys
from datetime import datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import subtyping_dir

ap = argparse.ArgumentParser()
ap.add_argument("--cancer", required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--dcc_file", required=True)
ap.add_argument("--method", default="SubtypeDCC")
ap.add_argument("--config_note", default="")
a = ap.parse_args()

d = pd.read_csv(a.dcc_file, sep='\t')
assert {'sample_name', 'dcc'} <= set(d.columns), d.columns.tolist()
d['patient_id'] = d['sample_name'].astype(str).str.strip()
d['cluster'] = pd.to_numeric(d['dcc']) - 1          # their output is 1-indexed

outdir = subtyping_dir(a.cancer, a.method, a.version)
os.makedirs(outdir, exist_ok=True)
fp = f"{outdir}/clusters.csv"
with open(fp, 'w') as fh:
    fh.write(f"# method: {a.method}\n")
    fh.write(f"# cancer: {a.cancer}\n")
    fh.write(f"# version: {a.version}\n")
    fh.write(f"# source: {a.dcc_file} (converted, 1-indexed -> 0-indexed)\n")
    if a.config_note:
        fh.write(f"# config: {a.config_note}\n")
    fh.write(f"# generated: {datetime.now():%Y-%m-%d %H:%M} by convert_dcc_clusters.py\n")
    d[['patient_id', 'cluster']].to_csv(fh, index=False)

print(f"{a.cancer}: {len(d)} patients, {d['cluster'].nunique()} clusters "
      f"{d['cluster'].value_counts().sort_index().to_dict()} -> {fp}")