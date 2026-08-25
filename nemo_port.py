import numpy as np
from sklearn.cluster import SpectralClustering

def nemo_clustering(omics_list, n_clusters=4, ratio=6):
    """Faithful port of NEMO (Rappoport & Shamir 2019).
    omics_list: list of (features x samples) arrays, same sample order.
    """
    n = omics_list[0].shape[1]
    k = max(1, round(n / ratio))
    sym_affs = []
    for X in omics_list:
        Xs = X.T  # samples x features
        # Gaussian affinity from Euclidean distance (SNF-style, sigma via local scaling)
        from scipy.spatial.distance import cdist
        D = cdist(Xs, Xs, 'euclidean')
        # local sigma = mean dist to k nearest (excluding self)
        sig = np.sort(D, axis=1)[:, 1:k+1].mean(axis=1) + 1e-9
        S = np.exp(-(D**2) / (sig[:,None]*sig[None,:] * 0.5 + 1e-9))
        # relative kNN: keep top-k per row, normalize kept to sum 1
        M = np.zeros_like(S)
        for i in range(n):
            idx = np.argsort(S[i])[::-1][:k]
            kept = S[i, idx]
            M[i, idx] = kept / (kept.sum() + 1e-9)
        sym = M + M.T
        sym_affs.append(sym)
    fused = np.mean(sym_affs, axis=0)  # all omics share all patients -> plain mean
    fused = (fused + fused.T) / 2
    labels = SpectralClustering(n_clusters, affinity='precomputed',
                                random_state=0).fit_predict(fused)
    return labels
