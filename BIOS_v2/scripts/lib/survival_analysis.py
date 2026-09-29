"""
survival_analysis.py — log-rank + Cox PH on subtyping clusters.

Reads a clusters.csv produced by any method, writes survival.csv and cox.csv
with a provenance header naming the exact clusters file used.

Usage:
    python survival_analysis.py --cancer BRCA --method BIOS \
        --version v2_benchmark_600ep_2026-09-15 --endpoint OS
"""
import argparse, os, sys
from datetime import datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import subtyping_dir, survival_dir, survival as survival_table
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--cancer", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--endpoint", default="OS", choices=["OS", "PFI", "DSS", "DFI"])
ap.add_argument("--clusters", default=None,
                help="override path to clusters.csv (default: derived from cancer/method/version)")
a = ap.parse_args()

pred_fp = a.clusters or f"{subtyping_dir(a.cancer, a.method, a.version)}/clusters.csv"
outdir = survival_dir(a.cancer, a.method, a.version)
os.makedirs(outdir, exist_ok=True)

def norm(s):
    return str(s).strip().upper().replace('_', '-')[:12]

# ---- clusters (comment lines starting with # are skipped) ----
pred = pd.read_csv(pred_fp, comment='#')
pred.columns = [c.lower() for c in pred.columns]
pred['patient'] = pred[pred.columns[0]].map(norm)
pred = pred.drop_duplicates('patient')[['patient', 'cluster']]

# ---- survival endpoints from TCGA-CDR ----
clin = pd.read_csv(survival_table(a.cancer), comment='#')
bc = 'bcr_patient_barcode' if 'bcr_patient_barcode' in clin.columns else clin.columns[0]
clin = clin[[bc, a.endpoint, f'{a.endpoint}.time']].copy()
clin.columns = ['patient', 'event', 'time']
clin['patient'] = clin['patient'].map(norm)
clin['event'] = pd.to_numeric(clin['event'], errors='coerce')
clin['time'] = pd.to_numeric(clin['time'], errors='coerce')
clin = clin.dropna(subset=['event', 'time'])
clin = clin[clin['time'] >= 0]

df = pred.merge(clin, on='patient', how='inner')
counts = df['cluster'].value_counts()
df = df[df['cluster'].isin(counts[counts >= 2].index)]

# ---- log-rank ----
lr = multivariate_logrank_test(df['time'], df['cluster'], df['event'])
neglog10p = -np.log10(max(lr.p_value, 1e-300))

# ---- Cox PH ----
design = pd.get_dummies(df[['time', 'event', 'cluster']],
                        columns=['cluster'], drop_first=True).astype(float)
cindex, cox_rows = np.nan, []
try:
    cph = CoxPHFitter().fit(design, 'time', 'event')
    cindex = cph.concordance_index_
    for cov, row in cph.summary.iterrows():
        cox_rows.append({'covariate': cov, 'coef': row['coef'],
                         'HR': row['exp(coef)'], 'p': row['p']})
except Exception as e:
    print(f"Cox skipped: {e}")

# ---- write survival.csv with provenance header ----
rel_src = os.path.relpath(os.path.abspath(pred_fp),
                          "/gpfs/research/fangroup/ug25b/BIOS/BIOS_v2")
header = (f"# source: {rel_src}\n"
          f"# generated: {datetime.now():%Y-%m-%d %H:%M} by survival_analysis.py\n")

summary = pd.DataFrame([{
    'cancer': a.cancer, 'method': a.method, 'version': a.version,
    'endpoint': a.endpoint, 'n': len(df), 'events': int(df['event'].sum()),
    'clusters': int(df['cluster'].nunique()),
    'logrank_chi2': round(lr.test_statistic, 4),
    'logrank_p': lr.p_value,
    'neglog10_p': round(neglog10p, 4),
    'c_index': round(cindex, 4) if not np.isnan(cindex) else '',
}])

out_fp = f"{outdir}/survival_{a.endpoint}.csv"
with open(out_fp, 'w') as fh:
    fh.write(header)
    summary.to_csv(fh, index=False)

if cox_rows:
    cox_fp = f"{outdir}/cox_{a.endpoint}.csv"
    with open(cox_fp, 'w') as fh:
        fh.write(header)
        pd.DataFrame(cox_rows).to_csv(fh, index=False)

# ---- KM plot ----
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 5))
for cl, sub in df.groupby('cluster'):
    kmf.fit(sub['time'] / 365.25, sub['event'], label=f'cluster {cl} (n={len(sub)})')
    kmf.plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel('Years'); ax.set_ylabel(f'{a.endpoint} probability')
ax.set_title(f'{a.cancer} {a.method} {a.endpoint} (log-rank p={lr.p_value:.2e})')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{outdir}/KM_{a.endpoint}.png", dpi=150)

print(summary.to_string(index=False))
print(f"\nsaved: {out_fp}")