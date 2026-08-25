#!/usr/bin/env python3
"""Per-CLUSTER pathway importance from the attention_query head (label-free).
Primary: group patients by BIOS's own clusters, rank the 50 pathways per cluster.
Optional: if ground-truth labels exist, map clusters->subtypes for validation."""
import sys, torch, numpy as np, pandas as pd
from modules.network import Network
from modules.ae import AE
from dataloader import get_feature
from gsva_anchors import CONFIGS

CKPT    = sys.argv[1] if len(sys.argv)>1 else 'save/model_aq_ucec_bd32/checkpoint_350.tar'
BIO_DIM = int(sys.argv[2]) if len(sys.argv)>2 else 32
CANCER, K = 'UCEC', 4
GT_FILE = 'data/ground_truth/ground_truth_UCEC.csv'   # optional, for validation only

anchor_names = [n[9:] for n in CONFIGS['H50']]   # strip HALLMARK_ prefix
ck = torch.load(CKPT, map_location='cpu')
ae = AE(hid_dim=256, bio_dim=BIO_DIM)
m = Network(ae=ae, feature_dim=256, class_num=K, bio_dim=BIO_DIM,
            n_anchors=50, head_type='attention_query')
m.load_state_dict(ck['net'], strict=False); m.eval()

dl = get_feature(CANCER, 64, training=False,
                 bio_anchor_file='data/bio_anchors/bio_anchors_UCEC_gsva_H50.csv')
cn = pd.read_csv(f'../subtype_file/fea/{CANCER}/CN.fea', index_col=0)
patients = list(cn.columns)

att_list, clu_list = [], []
with torch.no_grad():
    for batch in dl:
        x = batch[0].float()
        c, h, z_bio = m.forward_cluster(x)          # predicted clusters (label-free)
        m.bio_head(z_bio)                            # populate attention weights
        w = m.bio_head.last_attn_weights             # (B,50,bio_dim)
        att_list.append(w.sum(-1).numpy())           # mass per anchor -> (B,50)
        clu_list.append(c.numpy())
att = np.concatenate(att_list,0)                     # (N,50)
clusters = np.concatenate(clu_list,0)                # (N,)

# ── PRIMARY: per-cluster top pathways (label-free) ──
print(f"\n=== Per-CLUSTER pathway attention (label-free) — {CKPT} ===\n")
# also compute contrast: how much each cluster up-weights a pathway vs global mean
global_mean = att.mean(0)
for cl in sorted(set(clusters)):
    mask = clusters==cl
    cl_mean = att[mask].mean(0)
    contrast = cl_mean - global_mean                 # what THIS cluster emphasizes vs average
    order = np.argsort(contrast)[::-1]
    print(f"Cluster {cl} (n={mask.sum()}) — top 6 DISTINCTIVE pathways (vs global):")
    for i in order[:6]:
        print(f"    {anchor_names[i]:35s} +{contrast[i]:.4f}  (abs {cl_mean[i]:.3f})")
    print()

# ── OPTIONAL: map clusters -> known subtypes for validation (only if GT exists) ──
try:
    gt = pd.read_csv(GT_FILE)
    gt['patient'] = gt['patient_id'].astype(str).str.upper().str[:12]
    names = {0:'CN_HIGH',1:'CN_LOW',2:'MSI',3:'POLE'}
    pmap = dict(zip(gt['patient'], gt['true_cluster'].map(names)))
    pat12 = [str(p).upper()[:12] for p in patients]
    truesub = np.array([pmap.get(p,None) for p in pat12])
    print("=== VALIDATION: cluster -> subtype mapping (majority) ===")
    ct = pd.crosstab(pd.Series(clusters,name='cluster'),
                     pd.Series(truesub,name='subtype'))
    print(ct.to_string())
    print("\ndominant subtype per cluster:")
    for cl in ct.index:
        print(f"  cluster {cl} -> {ct.loc[cl].idxmax()} ({100*ct.loc[cl].max()/ct.loc[cl].sum():.0f}%)")
except Exception as e:
    print(f"(no ground-truth validation: {e})")
