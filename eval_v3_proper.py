"""
eval_v3_proper.py — enrichment recomputed on today's clusters, then Spearman.
Also tests |enrichment| in case the gate tracks discriminativeness, not direction.
"""
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
import argparse

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))

fea_CN = pd.read_csv('../subtype_file/fea/UCEC/CN.fea', header=0, index_col=0, sep=',')
patient_ids = fea_CN.columns.tolist()

ckpt = torch.load('save/model_ucec_v3_H40_hinge/checkpoint_400.tar',
                  map_location='cpu', weights_only=False)
ae = AE(hid_dim=cfg.feature_dim, bio_dim=ckpt['bio_dim'])
model = network.Network(ae=ae, feature_dim=cfg.feature_dim,
                        class_num=ckpt['n_clusters'], bio_dim=ckpt['bio_dim'],
                        n_anchors=ckpt['n_anchors'], head_type=ckpt['head_type'])
model.load_state_dict(ckpt['net'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

DL = get_feature('UCEC', 64, False,
                 bio_anchor_file='data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv')
hards = []
with torch.no_grad():
    for batch in DL:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        c, h, z_bio = model.forward_cluster(x)
        hards.append(c.detach().cpu().view(-1))
C = pd.Series(torch.cat(hards).numpy(), index=patient_ids)

# GSVA anchors, aligned to the same order
gsva = pd.read_csv('data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv')
gsva = gsva.set_index('patient_id').loc[patient_ids]
gsva.columns = [c.replace('HALLMARK_', '') for c in gsva.columns]

gates = pd.read_csv('results/v3_hinge_cluster_gates.csv', index_col=0)
gates.columns = [c.replace('HALLMARK_', '') for c in gates.columns]

k = ckpt['n_clusters']
print("cluster sizes:", C.value_counts().sort_index().to_dict())

rows, rhos_signed, rhos_abs = [], [], []
for j in range(k):
    mask = (C == j).values
    enr = {}
    for pw in gsva.columns:
        a, b = gsva.loc[mask, pw], gsva.loc[~mask, pw]
        stat = a.mean() - b.mean()
        try:
            p = mannwhitneyu(a, b).pvalue
        except ValueError:
            p = 1.0
        enr[pw] = (stat, -np.log10(max(p, 1e-300)))
    E = pd.DataFrame(enr, index=['diff', 'neglog10p']).T
    g = gates.loc[j] if j in gates.index else gates.iloc[j]

    top_enr = E.nlargest(3, 'neglog10p').index.tolist()
    top_gate = g.nlargest(3).index.tolist()
    print(f"\ncluster {j}: top enriched (by p): {top_enr}")
    print(f"           top gate:            {top_gate}")

    # signed: gate vs positive-enrichment significance
    sig_pos = E['neglog10p'] * np.sign(E['diff'])
    r1, p1 = spearmanr(g.values, sig_pos.loc[g.index].values)
    # unsigned: gate vs discriminativeness regardless of direction
    r2, p2 = spearmanr(g.values, E.loc[g.index, 'neglog10p'].values)
    rhos_signed.append(r1); rhos_abs.append(r2)
    print(f"           rho_signed={r1:.3f} (p={p1:.3g})   rho_|discrim|={r2:.3f} (p={p2:.3g})")

print(f"\nmean rho_signed = {np.mean(rhos_signed):.3f}  ->  "
      f"{'PASS' if np.mean(rhos_signed) > 0.4 else 'FAIL'} (pre-registered)")
print(f"mean rho_|discrim| = {np.mean(rhos_abs):.3f}  (exploratory — gate as "
      f"discriminative-pathway selector)")