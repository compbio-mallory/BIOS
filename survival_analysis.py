#!/usr/bin/env python3
"""KM + multivariate log-rank + Cox PH for BIOS clusters vs TCGA-CDR survival."""
import argparse, pandas as pd, numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

def norm_patient(s): return str(s).strip().upper().replace('_','-')[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)       # sample_id,cluster
    ap.add_argument('--clinical', required=True)   # bcr_patient_barcode,OS,OS.time,PFI,PFI.time
    ap.add_argument('--cancer', default='')
    ap.add_argument('--endpoint', default='PFI', choices=['OS','PFI','DSS','DFI'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred)
    pred.columns = [c.lower() for c in pred.columns]
    pred['patient'] = pred[pred.columns[0]].map(norm_patient)
    pred = pred.drop_duplicates('patient')

    clin = pd.read_csv(a.clinical)
    ev, tm = a.endpoint, f'{a.endpoint}.time'
    bc = 'bcr_patient_barcode' if 'bcr_patient_barcode' in clin.columns else clin.columns[0]
    clin = clin[[bc, ev, tm]].copy(); clin.columns = ['patient','event','time']
    clin['patient'] = clin['patient'].map(norm_patient)
    clin['event'] = pd.to_numeric(clin['event'], errors='coerce')
    clin['time']  = pd.to_numeric(clin['time'],  errors='coerce')
    clin = clin.dropna(subset=['event','time'])
    clin = clin[clin['time'] >= 0]

    df = pred.merge(clin, on='patient', how='inner')
    counts = df['cluster'].value_counts()
    df = df[df['cluster'].isin(counts[counts>=2].index)]
    print(f'[{a.cancer}] endpoint={a.endpoint}  patients={len(df)}  clusters={df["cluster"].nunique()}  events={int(df["event"].sum())}\n')

    lr = multivariate_logrank_test(df['time'], df['cluster'], df['event'])
    print(f'Log-rank (global): chi2={lr.test_statistic:.3f}  p={lr.p_value:.3e}\n')

    design = pd.get_dummies(df[['time','event','cluster']], columns=['cluster'], drop_first=True).astype(float)
    try:
        cph = CoxPHFitter(); cph.fit(design, 'time', 'event')
        print('Cox PH (ref = first cluster):')
        print(cph.summary[['coef','exp(coef)','p']].rename(columns={'exp(coef)':'HR'}).to_string(float_format=lambda x:f'{x:.3f}'))
        print(f'\nConcordance (C-index): {cph.concordance_index_:.3f}')
    except Exception as e:
        print('Cox skipped:', e)

    if a.out:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        kmf = KaplanMeierFitter(); fig, ax = plt.subplots(figsize=(7,5))
        for cl, sub in df.groupby('cluster'):
            kmf.fit(sub['time']/365.25, sub['event'], label=f'cluster {cl} (n={len(sub)})')
            kmf.plot_survival_function(ax=ax, ci_show=False)
        ax.set_xlabel('Years'); ax.set_ylabel(f'{a.endpoint} probability')
        ax.set_title(f'{a.cancer} {a.endpoint}  (log-rank p={lr.p_value:.2e})'); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.out, dpi=150); print(f'\nKM plot -> {a.out}')

if __name__ == '__main__':
    main()
