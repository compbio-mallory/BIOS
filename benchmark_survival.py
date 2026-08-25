#!/usr/bin/env python3
"""Run each baseline's clusters through PFI survival, compare to BIOS."""
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

CANCER, K = 'UCEC', 4
base = f'../subtype_file/fea/{CANCER}'
fea = [pd.read_csv(f'{base}/{o}.fea', index_col=0) for o in ['CN','meth','miRNA','rna']]
X = MinMaxScaler().fit_transform(np.concatenate(fea, axis=0).T)
patients = fea[0].columns.tolist()
Xp = PCA(n_components=50, random_state=0).fit_transform(X)

# clinical (PFI)
clin = pd.read_csv('data/labels/ucec_survival_cdr.csv')
clin['patient'] = clin['bcr_patient_barcode'].astype(str).str.upper().str[:12]
clin['event'] = pd.to_numeric(clin['PFI'], errors='coerce')
clin['time']  = pd.to_numeric(clin['PFI.time'], errors='coerce')
clin = clin.dropna(subset=['event','time'])
clin = clin[clin['time'] >= 0][['patient','event','time']]

def survival_stats(labels):
    df = pd.DataFrame({'patient':patients,'cluster':labels}).merge(clin,on='patient',how='inner')
    vc = df['cluster'].value_counts(); df = df[df['cluster'].isin(vc[vc>=2].index)]
    lr = multivariate_logrank_test(df['time'], df['cluster'], df['event'])
    try:
        d = pd.get_dummies(df[['time','event','cluster']], columns=['cluster'], drop_first=True).astype(float)
        cph = CoxPHFitter().fit(d,'time','event'); cidx = cph.concordance_index_
    except Exception:
        cidx = float('nan')
    return lr.p_value, cidx, len(df)

methods = {
  'KMeans':        KMeans(K,n_init=10,random_state=0).fit_predict(X),
  'PCA+KMeans':    KMeans(K,n_init=10,random_state=0).fit_predict(Xp),
  'Spectral':      SpectralClustering(K,affinity='nearest_neighbors',random_state=0).fit_predict(Xp),
  'Agglomerative': AgglomerativeClustering(K).fit_predict(Xp),
}
try:
    import snf
    aff = snf.make_affinity([f.T.values for f in fea], metric='euclidean', K=20)
    methods['SNF'] = SpectralClustering(K,affinity='precomputed',random_state=0).fit_predict(snf.snf(aff,K=20))
except Exception as e:
    print('(SNF skipped:', e, ')')

# BIOS clusters (already saved)
bios = pd.read_csv('data/labels/ucec_pred.csv')
methods['BIOS (ours)'] = bios.set_index('sample_id').loc[patients,'cluster'].values

rows=[]
for name, lab in methods.items():
    p, c, n = survival_stats(lab)
    rows.append((name, p, c, n))
    print(f'{name:14s} logrank_p={p:.2e}  C-index={c:.3f}  n={n}')

df = pd.DataFrame(rows, columns=['method','PFI_logrank_p','C_index','n']).sort_values('PFI_logrank_p')
print('\n=== PFI SURVIVAL COMPARISON (UCEC, k=4) ===')
print(df.to_string(index=False))
df.to_csv('results/UCEC_survival_comparison.csv', index=False)
print('\nsaved results/UCEC_survival_comparison.csv')
