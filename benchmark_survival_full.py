#!/usr/bin/env python3
"""Full metric panel per method (UCEC): log-rank p, C-index, HR, silhouette,
# significant clinical params (chi2/Kruskal-Wallis), running time.
Uses CORRECT matched-anchor BIOS clusters."""
import time, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, kruskal
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

CANCER, K = 'UCEC', 4
base = f'../subtype_file/fea/{CANCER}'
fea = [pd.read_csv(f'{base}/{o}.fea', index_col=0) for o in ['CN','meth','miRNA','rna']]
X = MinMaxScaler().fit_transform(np.concatenate(fea, axis=0).T)
patients = fea[0].columns.tolist()
Xp = PCA(n_components=50, random_state=0).fit_transform(X)

# survival (PFI)
clin = pd.read_csv('data/labels/ucec_survival_cdr.csv')
clin['patient'] = clin['bcr_patient_barcode'].astype(str).str.upper().str[:12]
clin['event'] = pd.to_numeric(clin['PFI'], errors='coerce')
clin['time']  = pd.to_numeric(clin['PFI.time'], errors='coerce')
clin = clin.dropna(subset=['event','time']); clin = clin[clin['time']>=0][['patient','event','time']]

# clinical params for enrichment
cdr = pd.read_excel('data/labels/TCGA-CDR.xlsx', sheet_name='TCGA-CDR')
cdr['patient'] = cdr['bcr_patient_barcode'].astype(str).str.upper().str[:12]
CONT = ['age_at_initial_pathologic_diagnosis']          # Kruskal-Wallis
DISC = ['gender','ajcc_pathologic_tumor_stage','histological_grade']  # chi2
cdr_sub = cdr[['patient']+CONT+DISC].copy()

def clinical_enrichment(df_lab):
    """count clinical params significantly associated with clusters (p<0.05)"""
    m = df_lab.merge(cdr_sub, on='patient', how='inner')
    nsig, details = 0, []
    for col in CONT:
        sub = m[['cluster',col]].dropna()
        sub[col] = pd.to_numeric(sub[col], errors='coerce'); sub=sub.dropna()
        groups=[g[col].values for _,g in sub.groupby('cluster') if len(g)>1]
        if len(groups)>=2:
            p = kruskal(*groups).pvalue
            details.append(f'{col}:{p:.1e}'); nsig += p<0.05
    for col in DISC:
        sub = m[['cluster',col]].dropna()
        sub = sub[~sub[col].astype(str).str.contains('Not|nan|NaN|\\[', na=True)]
        if sub[col].nunique()>1 and sub['cluster'].nunique()>1:
            ct = pd.crosstab(sub['cluster'], sub[col])
            if ct.shape[0]>1 and ct.shape[1]>1:
                p = chi2_contingency(ct)[1]
                details.append(f'{col}:{p:.1e}'); nsig += p<0.05
    return nsig, details

def metrics(labels, runtime):
    df = pd.DataFrame({'patient':patients,'cluster':labels})
    sil = silhouette_score(X, labels) if len(set(labels))>1 else float('nan')
    s = df.merge(clin, on='patient', how='inner')
    vc = s['cluster'].value_counts(); s = s[s['cluster'].isin(vc[vc>=2].index)]
    lr = multivariate_logrank_test(s['time'], s['cluster'], s['event'])
    try:
        d = pd.get_dummies(s[['time','event','cluster']], columns=['cluster'], drop_first=True).astype(float)
        cph = CoxPHFitter().fit(d,'time','event'); cidx=cph.concordance_index_
        _s = cph.summary
        _best = _s['p'].idxmin()          # cluster with strongest (smallest-p) effect
        hr = _s.loc[_best, 'exp(coef)']    # its hazard ratio (meaningful, not the null max)
        hr_p = _s.loc[_best, 'p']
    except Exception:
        cidx, hr, hr_p = float('nan'), float('nan'), float('nan')
    nsig, det = clinical_enrichment(df)
    return dict(logrank_p=lr.p_value, C_index=cidx, max_HR=hr, HR_p=hr_p, silhouette=sil,
                n_sig_clinical=nsig, runtime_s=runtime, n=len(s), clin_detail=';'.join(det))

methods = {}
for name, fn in [
    ('KMeans',        lambda: KMeans(K,n_init=10,random_state=0).fit_predict(X)),
    ('PCA+KMeans',    lambda: KMeans(K,n_init=10,random_state=0).fit_predict(Xp)),
    ('Spectral',      lambda: SpectralClustering(K,affinity='nearest_neighbors',random_state=0).fit_predict(Xp)),
    ('Agglomerative', lambda: AgglomerativeClustering(K).fit_predict(Xp)),
]:
    t=time.time(); lab=fn(); methods[name]=(lab, time.time()-t)
try:
    import snf
    t=time.time()
    aff = snf.make_affinity([f.T.values for f in fea], metric='euclidean', K=20)
    lab = SpectralClustering(K,affinity='precomputed',random_state=0).fit_predict(snf.snf(aff,K=20))
    methods['SNF']=(lab, time.time()-t)
except Exception as e: print('(SNF skipped:',e,')')
try:
    from nemo_port import nemo_clustering
    t=time.time(); lab=nemo_clustering([f.values for f in fea],K); methods['NEMO']=(lab,time.time()-t)
except Exception as e: print('(NEMO skipped:',e,')')

# BIOS — CORRECT matched clusters
bios = pd.read_csv('data/labels/ucec_pred_matched.csv')
bios.columns=[c.lower() for c in bios.columns]
methods['BIOS'] = (bios.set_index('sample_id').loc[patients,'cluster'].values, float('nan'))

rows=[]
for name,(lab,rt) in methods.items():
    m = metrics(np.asarray(lab), rt)
    rows.append({'method':name, **m})
    print(f'{name:14s} logrank_p={m["logrank_p"]:.2e}  C-idx={m["C_index"]:.3f}  '
          f'HR={m["max_HR"]:.2f}  sil={m["silhouette"]:.3f}  sigClin={m["n_sig_clinical"]}  t={m["runtime_s"]:.1f}s')

out = pd.DataFrame(rows).sort_values('logrank_p')
print('\n=== UCEC FULL METRIC PANEL (matched anchors, PFI, k=4) ===')
print(out[['method','logrank_p','C_index','max_HR','silhouette','n_sig_clinical','runtime_s','n']].to_string(index=False))
out.to_csv('results/UCEC_full_metrics_panel.csv', index=False)
print('\nsaved results/UCEC_full_metrics_panel.csv')
