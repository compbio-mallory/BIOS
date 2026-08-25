"""
eval_v3_final.py — ID-aligned clustering metrics + pre-registered Spearman test.
"""
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import v_measure_score, adjusted_rand_score

sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
import argparse

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))

# ── recompute assignments WITH patient order from the anchor file ─
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
C = torch.cat(hards).numpy()
assert len(C) == len(patient_ids), f"{len(C)} preds vs {len(patient_ids)} ids"
pred = pd.Series(C, index=patient_ids, name='pred')

# ── 1. ID-aligned V-measure / ARI ─────────────────────────────────
gt = pd.read_csv('data/ground_truth/ground_truth_UCEC.csv').set_index('patient_id')
common = gt.index.intersection(pred.index)
print(f"[1] aligned {len(common)}/{len(gt)} labeled patients")
y_true = gt.loc[common, 'label'].values
y_pred = pred.loc[common].values
print(f"    V-measure = {v_measure_score(y_true, y_pred):.4f}")
print(f"    ARI       = {adjusted_rand_score(y_true, y_pred):.4f}")

# ── 2. label composition per predicted cluster ────────────────────
print("\n[2] cluster composition:")
comp = pd.crosstab(pred.loc[common], gt.loc[common, 'label'])
print(comp.to_string())
dominant = comp.idxmax(1)

# ── 3. Spearman ρ: gate ranking vs enrichment ranking ────────────
gates = pd.read_csv('results/v3_hinge_cluster_gates.csv', index_col=0)  # (k, 40)
enr = pd.read_csv('results/UCEC_cluster_enrichment.csv')
print("\n[3] enrichment columns:", enr.columns.tolist())

# ── 3. Spearman ρ: gate ranking vs enrichment ranking ────────────
gates = pd.read_csv('results/v3_hinge_cluster_gates.csv', index_col=0)  # (k, 40)
gates.columns = [c.replace('HALLMARK_', '') for c in gates.columns]

enr = pd.read_csv('results/UCEC_cluster_enrichment.csv')
enr['neglog10_q'] = -np.log10(enr['q'].clip(lower=1e-300))

# Map July's enrichment clusters to subtypes via their signature pathways,
# then match today's clusters by dominant GT label.
# July enrichment: E2F/G2M -> serous/CN_HIGH, ESTROGEN_RESPONSE -> endometrioid
# (CN_LOW), PI3K/MTORC1 -> MSI.
print("\n[3] July enrichment cluster signatures (top-3 by -log10 q):")
enr_top = {}
for jc in sorted(enr['cluster'].unique()):
    sub = enr[enr['cluster'] == jc].nlargest(3, 'neglog10_q')
    enr_top[jc] = sub
    print(f"  enr cluster {jc}: " + ", ".join(
        f"{r.pathway}({r.neglog10_q:.0f})" for r in sub.itertuples()))

# signature -> subtype rules from the July analysis
def subtype_of(top_pathways):
    tp = set(top_pathways)
    if tp & {'E2F_TARGETS', 'G2M_CHECKPOINT', 'MYC_TARGETS_V1'}:
        return 'UCEC_CN_HIGH'
    if tp & {'ESTROGEN_RESPONSE_EARLY', 'ESTROGEN_RESPONSE_LATE'}:
        return 'UCEC_CN_LOW'
    if tp & {'PI3K_AKT_MTOR_SIGNALING', 'MTORC1_SIGNALING'}:
        return 'UCEC_MSI'
    return None

enr_subtype = {jc: subtype_of(list(enr_top[jc]['pathway'])) for jc in enr_top}
print("  inferred:", enr_subtype)

# today's clusters -> subtype via dominant GT label (from section 2)
print("\n[4] Spearman rho, matched by subtype:")
rhos = []
for j in gates.index:
    subtype = dominant.get(int(j), None)
    match = [jc for jc, s in enr_subtype.items() if s == subtype]
    if not match:
        print(f"  cluster {j} (dom: {subtype}): no matching enrichment cluster — skipped")
        continue
    e = enr[enr['cluster'] == match[0]].set_index('pathway')['neglog10_q']
    g = gates.loc[j]
    shared = g.index.intersection(e.index)
    rho, pval = spearmanr(g[shared].values, e[shared].values)
    rhos.append(rho)
    print(f"  cluster {j} (dom: {subtype}, enr cluster {match[0]}): "
          f"rho={rho:.3f} p={pval:.3g} n={len(shared)}")

if rhos:
    m = np.mean(rhos)
    print(f"\n  mean rho = {m:.3f}  ->  "
          f"{'PASS (>0.4)' if m > 0.4 else 'FAIL (<=0.4)'} per pre-registered criterion")