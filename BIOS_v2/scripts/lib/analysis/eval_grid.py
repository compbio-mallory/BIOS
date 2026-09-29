"""eval_grid.py — V/ARI for the reproduction + 2x2 runs.
Handles both save-path generations (A/B: no _b suffix; D/E: _b suffix)."""
import sys, os, argparse
import pandas as pd
import torch
sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from sklearn.metrics import v_measure_score, adjusted_rand_score

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))

# (label, cancer, K, model_dir, anchor_file)
RUNS = [
  ("A mlp/M4/bd4",    "BRCA", 5, "save/model_BRCA_mlp_gsva_decorr_M4_b4",    "gsva_decorr_M4"),
 ("A mlp/M4/bd4",    "UCEC", 4, "save/model_UCEC_mlp_gsva_decorr_M4_b4",    "gsva_decorr_M4"),
 ("B lin/M4/bd4",    "BRCA", 5, "save/model_BRCA_linear_gsva_decorr_M4_b4", "gsva_decorr_M4"),
 ("B lin/M4/bd4",    "UCEC", 4, "save/model_UCEC_linear_gsva_decorr_M4_b4", "gsva_decorr_M4"),
 ("D lin/M4/bd16",   "BRCA", 5, "save/model_BRCA_linear_gsva_decorr_M4_b16","gsva_decorr_M4"),
 ("D lin/M4/bd16",   "UCEC", 4, "save/model_UCEC_linear_gsva_decorr_M4_b16","gsva_decorr_M4"),
 ("E lin/H40/bd4",   "BRCA", 5, "save/model_BRCA_linear_gsva_H40_b4",       "gsva_H40"),
 ("E lin/H40/bd4",   "UCEC", 4, "save/model_UCEC_linear_gsva_H40_b4",       "gsva_H40"),
 ("C lin/H40/bd16",  "BRCA", 5, "save/model_BRCA_linear_H40",               "gsva_H40"),
 ("C lin/H40/bd16",  "UCEC", 4, "save/model_UCEC_linear_H40",               "gsva_H40"),
]

print(f"{'run':16s} {'cancer':6s} {'V':>7s} {'ARI':>7s}   (poster ref: BRCA mlp/M4 0.493/0.356; UCEC 0.360/0.363)")
print("-" * 78)
for label, cancer, k, mdir, amode in RUNS:
    ckpt_fp = f"{mdir}/checkpoint_200.tar"
    if not os.path.exists(ckpt_fp):
        print(f"{label:16s} {cancer:6s}   MISSING {ckpt_fp}"); continue
    ckpt = torch.load(ckpt_fp, map_location='cpu', weights_only=False)
    ae = AE(hid_dim=cfg.feature_dim, bio_dim=ckpt['bio_dim'])
    m = network.Network(ae=ae, feature_dim=cfg.feature_dim, class_num=k,
                        bio_dim=ckpt['bio_dim'], n_anchors=ckpt['n_anchors'],
                        head_type=ckpt['head_type'])
    m.load_state_dict(ckpt['net']); m.eval()
    fea = pd.read_csv(f'../subtype_file/fea/{cancer}/CN.fea', header=0, index_col=0, sep=',')
    ids = fea.columns.tolist()
    DL = get_feature(cancer, 64, False,
                     bio_anchor_file=f'data/bio_anchors/bio_anchors_{cancer}_{amode}.csv')
    hs = []
    with torch.no_grad():
        for b in DL:
            x = b[0] if isinstance(b, (list, tuple)) else b
            c, h, z = m.forward_cluster(x)
            hs.append(c.view(-1))
    pred = pd.Series(torch.cat(hs).numpy(), index=ids)
    # save clusters for survival follow-up
    tag = label.replace(' ', '_').replace('/', '-')
    pd.DataFrame({'patient_id': pred.index, 'cluster': pred.values}).to_csv(
        f'results/{cancer}_{tag}_clusters.csv', index=False)
    gt = pd.read_csv(f'data/ground_truth/ground_truth_{cancer}.csv')
    id_col = 'patient_id' if 'patient_id' in gt.columns else gt.columns[0]
    gt = gt.set_index(id_col)
    lab = 'label' if 'label' in gt.columns else gt.columns[-1]
    common = gt.index.intersection(pred.index)
    y = gt.loc[common, lab]
    print(f"{label:16s} {cancer:6s} {v_measure_score(y, pred.loc[common]):7.4f} "
          f"{adjusted_rand_score(y, pred.loc[common]):7.4f}")