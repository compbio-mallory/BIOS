"""
probe_clusters_generic.py — cluster sizes + per-cluster GSVA profiles.
Usage: python probe_clusters_generic.py --cancer UVM --k 4
"""
import sys, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from scipy.stats import mannwhitneyu

p = argparse.ArgumentParser()
p.add_argument("--cancer", required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--head", default="linear")
p.add_argument("--epoch", type=int, default=200)
args = p.parse_args()

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))
anchor_file = f'data/bio_anchors/bio_anchors_{args.cancer}_gsva_H40.csv'
ckpt_path = f'save/model_{args.cancer}_{args.head}_H40/checkpoint_{args.epoch}.tar'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ae = AE(hid_dim=cfg.feature_dim, bio_dim=ckpt['bio_dim'])
model = network.Network(ae=ae, feature_dim=cfg.feature_dim,
                        class_num=ckpt['n_clusters'], bio_dim=ckpt['bio_dim'],
                        n_anchors=ckpt['n_anchors'], head_type=ckpt['head_type'])
model.load_state_dict(ckpt['net'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

fea_CN = pd.read_csv(f'../subtype_file/fea/{args.cancer}/CN.fea',
                     header=0, index_col=0, sep=',')
patient_ids = fea_CN.columns.tolist()

DL = get_feature(args.cancer, 64, False, bio_anchor_file=anchor_file)
hards = []
with torch.no_grad():
    for batch in DL:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        c, h, z_bio = model.forward_cluster(x)
        hards.append(c.detach().cpu().view(-1))
C = pd.Series(torch.cat(hards).numpy(), index=patient_ids)

gsva = pd.read_csv(anchor_file).set_index('patient_id').loc[patient_ids]
gsva.columns = [c.replace('HALLMARK_', '') for c in gsva.columns]

sizes = C.value_counts().sort_index()
print(f"\n{args.cancer}: cluster sizes {sizes.to_dict()}")
frac = sizes.max() / len(C)
print(f"largest cluster fraction: {frac:.2f} "
      f"{'(DEGENERATE — one cluster dominates)' if frac > 0.85 else '(ok)'}")

print(f"\nper-cluster top-enriched pathways (mean GSVA diff, p<0.01 by MWU):")
for j in sorted(C.unique()):
    mask = (C == j).values
    rows = []
    for pw in gsva.columns:
        a, b = gsva.loc[mask, pw], gsva.loc[~mask, pw]
        try: pv = mannwhitneyu(a, b).pvalue
        except ValueError: pv = 1.0
        rows.append((pw, a.mean() - b.mean(), pv))
    df = pd.DataFrame(rows, columns=['pw', 'diff', 'p'])
    top = df[df.p < 0.01].nlargest(3, 'diff')
    desc = ", ".join(f"{r.pw}(+{r.diff:.2f})" for r in top.itertuples()) or "NONE significant"
    print(f"  cluster {j} (n={sizes[j]}): {desc}")

# save assignments for survival analysis
out = pd.DataFrame({'patient_id': C.index, 'cluster': C.values})
out.to_csv(f'results/{args.cancer}_{args.head}_H40_clusters.csv', index=False)
print(f"\nsaved: results/{args.cancer}_{args.head}_H40_clusters.csv")