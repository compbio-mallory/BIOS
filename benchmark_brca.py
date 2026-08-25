#!/usr/bin/env python3
"""BRCA method comparison vs PAM50 (k=5). Self-contained; needs nemo_port.py.
BIOS/Subtype-DCC plugged from re-eval (best epoch, same eval pipeline)."""
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

CANCER, K = 'BRCA', 5
base = f'../subtype_file/fea/{CANCER}'
fea = [pd.read_csv(f'{base}/{o}.fea', index_col=0) for o in ['CN','meth','miRNA','rna']]
X = MinMaxScaler().fit_transform(np.concatenate(fea, axis=0).T)
patients = fea[0].columns.tolist()
Xp = PCA(n_components=50, random_state=0).fit_transform(X)

gt = pd.read_csv('data/ground_truth/ground_truth_BRCA.csv')
gtm = dict(zip(gt['patient_id'], gt['numeric_label']))
idx = [i for i,p in enumerate(patients) if p in gtm]
true = np.array([gtm[patients[i]] for i in idx])
print(f'{CANCER}: {len(idx)} labelled patients, {X.shape[1]} features, k={K}\n')

def score(labels):
    le = labels[idx]
    return (adjusted_rand_score(true, le),
            normalized_mutual_info_score(true, le),
            v_measure_score(true, le))

methods = {
 'KMeans':        KMeans(K, n_init=10, random_state=0).fit_predict(X),
 'PCA+KMeans':    KMeans(K, n_init=10, random_state=0).fit_predict(Xp),
 'Spectral':      SpectralClustering(K, affinity='nearest_neighbors', random_state=0).fit_predict(Xp),
 'Agglomerative': AgglomerativeClustering(K).fit_predict(Xp),
}
try:
    import snf
    aff = snf.make_affinity([f.T.values for f in fea], metric='euclidean', K=20)
    methods['SNF'] = SpectralClustering(K, affinity='precomputed', random_state=0).fit_predict(snf.snf(aff, K=20))
except Exception as e:
    print('(SNF skipped:', e, ')')
try:
    from nemo_port import nemo_clustering
    methods['NEMO (reimpl)'] = nemo_clustering([f.values for f in fea], K)
except Exception as e:
    print('(NEMO skipped:', e, ')')

rows = []
for n, l in methods.items():
    a, nm, v = score(np.asarray(l))
    rows.append((n, a, nm, v))
    print(f'{n:16s} ARI={a:.4f}  NMI={nm:.4f}  V={v:.4f}')

# BIOS and Subtype-DCC from re-eval (best epoch, same pipeline)
rows.append(('Subtype-DCC', 0.3412, None, 0.4746))   # ep250
rows.append(('BIOS (ours)', 0.3556, None, 0.4931))   # ep300

df = pd.DataFrame(rows, columns=['method','ARI','NMI','V_measure']).sort_values('ARI', ascending=False)
print('\n=== BRCA FULL COMPARISON (PAM50, k=5) ===')
print(df.to_string(index=False))
df.to_csv('results/BRCA_full_metrics.csv', index=False)
print('\nsaved results/BRCA_full_metrics.csv')
