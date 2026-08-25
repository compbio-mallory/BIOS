#!/usr/bin/env python3
"""Per-cluster pathway importance, v2 — reads bd16 where attention has signal.
Uses attention variance + contrast to surface subtype-distinctive pathways."""
import sys, torch, numpy as np, pandas as pd
from modules.network import Network
from modules.ae import AE
from dataloader import get_feature
from gsva_anchors import CONFIGS

CKPT    = sys.argv[1] if len(sys.argv)>1 else 'save/model_aq_ucec_bd16/checkpoint_350.tar'
BIO_DIM = int(sys.argv[2]) if len(sys.argv)>2 else 16
CANCER, K = 'UCEC', 4

anchor_names = [n[9:] for n in CONFIGS['H50']]
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
        c,h,zb = m.forward_cluster(x)
        m.bio_head(zb)
        w = m.bio_head.last_attn_weights          # (B,50,bio_dim)
        # importance per anchor = how PEAKED its attention is (max-mean), captures selectivity
        peakedness = (w.max(-1).values - w.mean(-1))   # (B,50)
        att_list.append(peakedness.numpy())
        clu_list.append(c.numpy())
att = np.concatenate(att_list,0)                  # (N,50)
clusters = np.concatenate(clu_list,0)

# also get per-cluster MEAN attention mass (the abs level)
att_mass_list=[]
with torch.no_grad():
    for batch in dl:
        x=batch[0].float(); c,h,zb=m.forward_cluster(x); m.bio_head(zb)
        att_mass_list.append(m.bio_head.last_attn_weights.sum(-1).numpy())
mass=np.concatenate(att_mass_list,0)

gmean = mass.mean(0)
print(f"\n=== bd{BIO_DIM} per-cluster DISTINCTIVE pathways (contrast vs global) — {CKPT} ===\n")
for cl in sorted(set(clusters)):
    m_ = clusters==cl
    contrast = mass[m_].mean(0) - gmean
    order = np.argsort(contrast)[::-1]
    print(f"Cluster {cl} (n={m_.sum()}):")
    for i in order[:5]:
        print(f"    +{contrast[i]:+.4f}  {anchor_names[i]}")
    print()

# cluster->subtype validation
gt=pd.read_csv('data/ground_truth/ground_truth_UCEC.csv')
gt['patient']=gt['patient_id'].astype(str).str.upper().str[:12]
names={0:'CN_HIGH',1:'CN_LOW',2:'MSI',3:'POLE'}
pmap=dict(zip(gt['patient'],gt['true_cluster'].map(names)))
truesub=np.array([pmap.get(str(p).upper()[:12],None) for p in patients])
ct=pd.crosstab(pd.Series(clusters,name='cluster'),pd.Series(truesub,name='subtype'))
print("cluster -> dominant subtype:")
for cl in ct.index:
    print(f"  cluster {cl} -> {ct.loc[cl].idxmax()} ({100*ct.loc[cl].max()/ct.loc[cl].sum():.0f}%)")
