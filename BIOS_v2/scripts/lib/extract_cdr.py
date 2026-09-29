"""
extract_cdr.py — per-cancer survival endpoints from the TCGA-CDR table.

Source: TCGA-CDR.xlsx, Liu et al., Cell 2018, "An Integrated TCGA Pan-Cancer
Clinical Data Resource" (downloaded resource, not produced here).
Output: {CANCER}/ground_truth/survival_endpoints_{CANCER}_CDR.csv

Usage:
    python extract_cdr.py               # all 9
    python extract_cdr.py --cancer UVM  # one
"""
import argparse, os, sys
from datetime import datetime
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, reference

CANCERS = ["BRCA", "UCEC", "LUAD", "BLCA", "KIRC", "PAAD", "SKCM", "STAD", "UVM"]

ap = argparse.ArgumentParser()
ap.add_argument("--cancer", default=None)
a = ap.parse_args()

src = reference("TCGA-CDR.xlsx")
cdr = pd.read_excel(src, sheet_name=0, index_col=0)
print(f"CDR table: {cdr.shape[0]} patients, columns include "
      f"{[c for c in cdr.columns if c in ('type','OS','OS.time','PFI','PFI.time')]}")

cols = ['bcr_patient_barcode']
for ep in ['OS', 'PFI', 'DSS', 'DFI']:
    if ep in cdr.columns and f'{ep}.time' in cdr.columns:
        cols += [ep, f'{ep}.time']

for C in ([a.cancer] if a.cancer else CANCERS):
    sub = cdr[cdr['type'] == C][cols]
    outdir = f"{ROOT}/{C}/ground_truth"
    os.makedirs(outdir, exist_ok=True)
    fp = f"{outdir}/survival_endpoints_{C}_CDR.csv"
    with open(fp, 'w') as fh:
        fh.write(f"# source: TCGA-CDR.xlsx (Liu et al., Cell 2018), rows where type == {C}\n")
        fh.write(f"# extracted: {datetime.now():%Y-%m-%d %H:%M} by extract_cdr.py\n")
        sub.to_csv(fh, index=False)
    ev = int(pd.to_numeric(sub['OS'], errors='coerce').sum()) if 'OS' in sub else 0
    print(f"{C}: {len(sub)} patients, {ev} OS events -> {fp}")