"""V-measure and ARI at every saved checkpoint — reconstructs the May
best-checkpoint selection and shows where clustering quality plateaus."""
import sys, glob, re, argparse
import pandas as pd, torch
sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from sklearn.metrics import v_measure_score, adjusted_rand_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--cancer", default="BRCA")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--anchors", required=True)
a = ap.parse_args()

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))
fea = pd.read_csv(f'../subtype_file/fea/{a.cancer}/CN.fea', header=0, index_col=0, sep=',')
ids = fea.columns.tolist()
gt = pd.read_csv(f'data/ground_truth/ground_truth_{a.cancer}.csv')
gt = gt.set_index('patient_id' if 'patient_id' in gt.columns else gt.columns[0])
lab = 'label' if 'label' in gt.columns else gt.columns[-1]

eps, vs, aris = [], [], []
files = sorted(glob.glob(f"{a.model_dir}/checkpoint_*.tar"),
               key=lambda f: int(re.search(r'_(\d+)\.tar', f).group(1)))
for f in files:
    ep = int(re.search(r'_(\d+)\.tar', f).group(1))
    ck = torch.load(f, map_location='cpu', weights_only=False)
    ae = AE(hid_dim=cfg.feature_dim, bio_dim=ck['bio_dim'])
    m = network.Network(ae=ae, feature_dim=cfg.feature_dim, class_num=a.k,
                        bio_dim=ck['bio_dim'],
                        n_anchors=ck.get('n_anchors') or ck['bio_dim'],
                        head_type=ck['head_type'])
    m.load_state_dict(ck['net']); m.eval()
    DL = get_feature(a.cancer, 64, False, bio_anchor_file=a.anchors)
    hs = []
    with torch.no_grad():
        for b in DL:
            x = b[0] if isinstance(b, (list, tuple)) else b
            c, h, z = m.forward_cluster(x); hs.append(c.view(-1))
    pred = pd.Series(torch.cat(hs).numpy(), index=ids)
    common = gt.index.intersection(pred.index)
    v = v_measure_score(gt.loc[common, lab], pred.loc[common])
    ar = adjusted_rand_score(gt.loc[common, lab], pred.loc[common])
    eps.append(ep); vs.append(v); aris.append(ar)
    print(f"ep {ep:4d}  V={v:.4f}  ARI={ar:.4f}")
    del m, ck

best = eps[vs.index(max(vs))]
print(f"\nbest V={max(vs):.4f} at epoch {best}")
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(eps, vs, marker='o', ms=3, lw=1.2, label='V-measure')
ax.plot(eps, aris, marker='s', ms=3, lw=1.2, label='ARI')
ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax.grid(which='major', alpha=0.45)
ax.grid(which='minor', alpha=0.20, ls=':')
ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
ax.set_title(f"{a.cancer} {a.model_dir.split('/')[-1]} — metric vs epoch")
ax.legend()
plt.tight_layout()
plt.savefig(f"results/sweep_{a.cancer}_{a.model_dir.split('/')[-1]}.png", dpi=200)