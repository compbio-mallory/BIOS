#!/usr/bin/env python3
"""Run classical clustering baselines on UCEC omics, score vs ground truth.
Same feature loading as dataloader.py, same ARI/NMI as evaluate_clustering.py."""
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

CANCER, K = 'UCEC', 4
base = f'../subtype_file/fea/{CANCER}'
fea = [pd.read_csv(f'{base}/{o}.fea', index_col=0) for o in ['CN','meth','miRNA','rna']]
X = np.concatenate(fea, axis=0).T
X = MinMaxScaler().fit_transform(X)
patients = fea[0].columns.tolist()

gt = pd.read_csv('data/ground_truth/ground_truth_UCEC.csv')
gt_map = dict(zip(gt['patient_id'], gt['true_cluster']))
idx = [i for i,p in enumerate(patients) if p in gt_map]
Xe = X[idx]; true = np.array([gt_map[patients[i]] for i in idx])
print(f'{CANCER}: {Xe.shape[0]} patients, {Xe.shape[1]} features, k={K}\n')

Xp = PCA(n_components=50, random_state=0).fit_transform(Xe)  # for spectral/agglom speed

methods = {
  'KMeans':        lambda: KMeans(K, n_init=10, random_state=0).fit_predict(Xe),
  'PCA+KMeans':    lambda: KMeans(K, n_init=10, random_state=0).fit_predict(Xp),
  'Spectral':      lambda: SpectralClustering(K, affinity='nearest_neighbors', random_state=0).fit_predict(Xp),
  'Agglomerative': lambda: AgglomerativeClustering(K).fit_predict(Xp),
}
try:
    import snf
    def run_snf():
        aff = snf.make_affinity([f.T.values[idx] for f in fea], metric='euclidean', K=20)
        fused = snf.snf(aff, K=20)
        return SpectralClustering(K, affinity='precomputed', random_state=0).fit_predict(fused)
    methods['SNF'] = run_snf
except Exception:
    print('(snfpy not available — skipping SNF)\n')

rows=[]
for name, fn in methods.items():
    try:
        p = fn()
        rows.append((name, adjusted_rand_score(true,p), normalized_mutual_info_score(true,p), v_measure_score(true,p)))
        print(f'{name:14s} ARI={rows[-1][1]:.4f}  NMI={rows[-1][2]:.4f}')
    except Exception as e:
        print(f'{name:14s} FAILED: {e}')

rows.append(('BIOS (ours)', 0.2972, 0.3104, 0.3104))  # epoch 210
df = pd.DataFrame(rows, columns=['method','ARI','NMI','V_measure']).sort_values('ARI', ascending=False)
print('\n=== COMPARISON (UCEC, MOLECULAR labels, k=4) ===')
print(df.to_string(index=False))
df.to_csv('results/UCEC_baseline_comparison.csv', index=False)
print('\nsaved results/UCEC_baseline_comparison.csv')
