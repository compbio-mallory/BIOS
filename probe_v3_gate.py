"""
probe_v3_gate.py — verdict on the hinge run.
Loads checkpoint_400, forwards the UCEC cohort, reports:
  raw mean gate entropy, cross-cluster gate divergence,
  top-5 pathways per cluster, V-measure/ARI vs ground truth.
"""
import os, sys, math, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from sklearn.metrics import v_measure_score, adjusted_rand_score

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="save/model_ucec_v3_H40_hinge/checkpoint_400.tar")
p.add_argument("--cancer_type", default="UCEC")
p.add_argument("--anchor_file", default="data/bio_anchors/bio_anchors_UCEC_gsva_H40.csv")
p.add_argument("--feature_dim", type=int, default=None, help="override if config differs")
args = p.parse_args()

# ── config (same source as training) ─────────────────────────────
cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))
feature_dim = args.feature_dim or cfg.feature_dim

# ── checkpoint metadata drives model construction ────────────────
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
bio_dim, n_anchors = ckpt["bio_dim"], ckpt["n_anchors"]
k, head_type = ckpt["n_clusters"], ckpt["head_type"]
print(f"ckpt: epoch={ckpt['epoch']} head={head_type} bio_dim={bio_dim} "
      f"anchors={n_anchors} k={k}")

ae = AE(hid_dim=feature_dim, bio_dim=bio_dim)
model = network.Network(ae=ae, feature_dim=feature_dim, class_num=k,
                        bio_dim=bio_dim, n_anchors=n_anchors, head_type=head_type)
model.load_state_dict(ckpt["net"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# ── data (same loader as training) ───────────────────────────────
DL = get_feature(args.cancer_type, 64, True, bio_anchor_file=args.anchor_file)
pathways = list(pd.read_csv(args.anchor_file, index_col=0).columns)
assert len(pathways) == n_anchors, f"{len(pathways)} names vs {n_anchors} anchors"

gates, hards = [], []
with torch.no_grad():
    for batch in DL:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        c, h, z_bio = model.forward_cluster(x)
        _, aux = model.bio_predict(h, z_bio)
        gates.append(aux["gate"].cpu())
        hards.append(c.detach().cpu().view(-1))

G = torch.cat(gates)            # (N, 40)
C = torch.cat(hards)            # (N,)
N = G.shape[0]
print(f"\ncohort: {N} patients")

# ── 1. RAW mean gate entropy — the verdict ───────────────────────
H = -(G * (G + 1e-8).log()).sum(1)
print(f"\n[1] raw gate entropy  mean={H.mean():.3f}  (uniform={math.log(n_anchors):.3f})")
print(f"    min={H.min():.3f} median={H.median():.3f} max={H.max():.3f}")
print(f"    VERDICT: {'COLLAPSED' if H.mean() < 0.5 else 'HELD' if H.mean() > 1.0 else 'MARGINAL'}")

# ── 2. per-cluster mean gates + divergence ───────────────────────
cluster_gates = torch.stack([G[C == j].mean(0) for j in range(k)])  # (k, 40)
sim = torch.nn.functional.cosine_similarity(
    cluster_gates.unsqueeze(1), cluster_gates.unsqueeze(0), dim=-1)
off = sim[~torch.eye(k, dtype=bool)]
print(f"\n[2] cross-cluster gate cosine  mean={off.mean():.3f}  "
      f"({'DIFFERENTIATED' if off.mean() < 0.7 else 'CONVERGED — clusters share gates'})")

# ── 3. top-5 pathways per cluster ────────────────────────────────
print("\n[3] top-5 pathways per cluster:")
for j in range(k):
    n_j = int((C == j).sum())
    top = cluster_gates[j].topk(5)
    names = [f"{pathways[i]}({w:.3f})" for i, w in zip(top.indices.tolist(),
                                                        top.values.tolist())]
    print(f"  cluster {j} (n={n_j}): " + ", ".join(names))

# ── 4. clustering quality ────────────────────────────────────────
gt_path = f"data/ground_truth/ground_truth_{args.cancer_type}.csv"
if os.path.exists(gt_path):
    gt = pd.read_csv(gt_path)
    y = gt.iloc[:, -1].values
    if len(y) == N:
        print(f"\n[4] V-measure={v_measure_score(y, C.numpy()):.4f}  "
              f"ARI={adjusted_rand_score(y, C.numpy()):.4f}")
    else:
        print(f"\n[4] gt length {len(y)} != cohort {N} — skipped (check label alignment)")
else:
    print(f"\n[4] no ground truth at {gt_path}")

# save per-cluster gates for the Spearman step
out = pd.DataFrame(cluster_gates.numpy(), columns=pathways)
out.to_csv("results/v3_hinge_cluster_gates.csv")
print("\nsaved: results/v3_hinge_cluster_gates.csv")