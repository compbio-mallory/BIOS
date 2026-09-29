#!/usr/bin/env python3
"""Per-cluster pathway enrichment with significance (label-free interpretability).
Uses BIOS clusters + H50 GSVA anchors. Reports mean-diff + t-test p (BH-corrected)."""
import sys, numpy as np, pandas as pd
from scipy import stats
from gsva_anchors import CONFIGS

def bh_correct(pvals):
    p=np.asarray(pvals); n=len(p); order=np.argsort(p)
    q=np.empty(n); prev=1.0
    for rank,idx in enumerate(order[::-1]):
        r=n-rank
        val=min(prev, p[idx]*n/r); q[idx]=val; prev=val
    return q

CANCER   = sys.argv[1] if len(sys.argv)>1 else 'UCEC'
PRED     = sys.argv[2] if len(sys.argv)>2 else 'data/labels/ucec_pred_matched.csv'
ANCHORS  = sys.argv[3] if len(sys.argv)>3 else 'data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv'
names=[n[9:] for n in CONFIGS['H50']]

pred=pd.read_csv(PRED); pred.columns=[c.lower() for c in pred.columns]
idcol='sample_id' if 'sample_id' in pred.columns else pred.columns[0]
pred['p']=pred[idcol].astype(str).str.upper().str[:12]
a=pd.read_csv(ANCHORS); a['p']=a['patient_id'].astype(str).str.upper().str[:12]
m=pred.merge(a,on='p')
cols=[c for c in a.columns if c.startswith('HALLMARK')]
X=m[cols].values; C=m['cluster'].values

rows=[]
for cl in sorted(set(C)):
    inm=C==cl; out=~inm
    for j,pw in enumerate(cols):
        diff=X[inm,j].mean()-X[out,j].mean()
        t,p=stats.ttest_ind(X[inm,j],X[out,j],equal_var=False)
        rows.append((cl,names[j],diff,p))
df=pd.DataFrame(rows,columns=['cluster','pathway','enrichment','p'])
# BH correction within each cluster
df['q']=np.nan
for cl in df['cluster'].unique():
    mask=df['cluster']==cl
    df.loc[mask,'q']=bh_correct(df.loc[mask,'p'].values)
df.to_csv(f'results/{CANCER}_cluster_enrichment.csv',index=False)

print(f'=== {CANCER}: per-cluster enriched pathways (q<0.05, sorted by enrichment) ===\n')
for cl in sorted(set(C)):
    sub=df[(df['cluster']==cl)&(df['q']<0.05)&(df['enrichment']>0)].sort_values('enrichment',ascending=False)
    print(f'Cluster {cl} (n={(C==cl).sum()}) — {len(sub)} significantly enriched:')
    for _,r in sub.head(6).iterrows():
        print(f'    {r.enrichment:+.3f}  q={r.q:.1e}  {r.pathway[:35]}')
    print()
print(f'saved results/{CANCER}_cluster_enrichment.csv')
