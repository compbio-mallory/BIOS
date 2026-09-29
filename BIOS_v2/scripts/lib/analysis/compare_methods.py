"""
compare_methods.py — collect survival and label results across methods/versions
into one comparison table.

Reads the saved survival_{OS,PFI}.csv, permutation_{OS,PFI}.txt and labels.csv
for each cancer x method x version. Writes analysis/method_comparison.csv.

Usage:
  python compare_methods.py BIOS:v2_benchmark_600ep_2026-09-10 \
      SubtypeDCC:v1_fd128_600ep_2026-09-18 SubtypeDCC:v1_fd256_600ep_2026-09-18
"""
import os, re, sys
import pandas as pd

ROOT = "/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2"
CANCERS = ["BRCA", "UCEC", "LUAD", "BLCA", "KIRC", "PAAD", "SKCM", "STAD", "UVM"]

# Subtype-DCC Table 1 (Zhao et al. 2023): OS -log10 p / significant clinical params
PUBLISHED = {
    "BRCA": (1.11, 5), "BLCA": (2.33, 6), "KIRC": (8.79, 6), "LUAD": (1.69, 4),
    "PAAD": (3.75, 1), "SKCM": (5.94, 4), "STAD": (1.48, 2), "UCEC": (5.46, 1),
    "UVM": (2.77, 0),
}

def read_analytic(c, m, v, ep):
    fp = f"{ROOT}/{c}/results/survival_results/{m}/{v}/survival_{ep}.csv"
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp, comment='#')
    return float(d['neglog10_p'].iloc[0])

def read_perm(c, m, v, ep):
    fp = f"{ROOT}/{c}/results/survival_results/{m}/{v}/permutation_{ep}.txt"
    if not os.path.exists(fp):
        return None
    for line in open(fp):
        if line.startswith("permutation:"):
            mm = re.search(r"-log10p=(>?)([\d.]+)", line)
            if mm:
                return (">" if mm.group(1) else "") + mm.group(2)
    return "running"

def read_labels(c, m, v):
    fp = f"{ROOT}/{c}/results/label_results/{m}/{v}/labels.csv"
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp, comment='#')
    tested = int((d['status'] == 'tested').sum())
    sig = int(d['significant'].sum())
    if 'significant_perm' in d.columns:
        sig_p = int(d['significant_perm'].sum())
        return f"{sig}/{tested} | perm {sig_p}/{tested}"
    return f"{sig}/{tested}"

runs = [a.split(":", 1) for a in sys.argv[1:]]
if not runs:
    sys.exit("usage: compare_methods.py METHOD:version [METHOD:version ...]")

rows = []
for c in CANCERS:
    row = {"cancer": c,
           "published_OS": PUBLISHED[c][0],
           "published_labels": PUBLISHED[c][1]}
    for m, v in runs:
        tag = f"{m}_{v.split('_')[1]}" if m == "SubtypeDCC" else m
        row[f"{tag}_OS_chi2"] = read_analytic(c, m, v, "OS")
        row[f"{tag}_OS_perm"] = read_perm(c, m, v, "OS")
        row[f"{tag}_PFI_chi2"] = read_analytic(c, m, v, "PFI")
        row[f"{tag}_PFI_perm"] = read_perm(c, m, v, "PFI")
        row[f"{tag}_labels"] = read_labels(c, m, v)
    rows.append(row)

df = pd.DataFrame(rows)
out = f"{ROOT}/analysis/method_comparison.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write(f"# runs: {' '.join(sys.argv[1:])}\n")
    fh.write("# published: Subtype-DCC Table 1 (Zhao et al., Brief Bioinform 2023)\n")
    df.to_csv(fh, index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(df.to_string(index=False))
print(f"\nsaved {out}")