"""Log-rank -log10 p vs epoch across saved checkpoints (BIOS clusters, TCGA-CDR)."""
import sys, glob, re, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from lifelines.statistics import multivariate_logrank_test
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--cancer", default="BRCA")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--anchors", required=True)
ap.add_argument("--clinical", required=True)      # data/labels/brca_survival_cdr.csv
ap.add_argument("--endpoint", default="OS", choices=["OS", "PFI"])
ap.add_argument("--every", type=int, default=2,   # every 2nd checkpoint = every 20 epochs
                help="stride over checkpoints to save time")
a = ap.parse_args()

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))
fea = pd.read_csv(f'../subtype_file/fea/{a.cancer}/CN.fea', header=0, index_col=0, sep=',')
ids = fea.columns.tolist()

clin = pd.read_csv(a.clinical)
bc = 'bcr_patient_barcode' if 'bcr_patient_barcode' in clin.columns else clin.columns[0]
clin = clin[[bc, a.endpoint, f'{a.endpoint}.time']].copy()
clin.columns = ['patient', 'event', 'time']
clin['patient'] = clin['patient'].astype(str).str.upper().str.replace('_', '-').str[:12]
clin['event'] = pd.to_numeric(clin['event'], errors='coerce')
clin['time'] = pd.to_numeric(clin['time'], errors='coerce')
clin = clin.dropna(subset=['event', 'time'])
clin = clin[clin['time'] >= 0]

files = sorted(glob.glob(f"{a.model_dir}/checkpoint_*.tar"),
               key=lambda f: int(re.search(r'_(\d+)\.tar', f).group(1)))[::a.every]
eps, negl = [], []
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
    pred = pd.DataFrame({'patient': ids, 'cluster': torch.cat(hs).numpy()})
    df = pred.merge(clin, on='patient', how='inner')
    counts = df['cluster'].value_counts()
    df = df[df['cluster'].isin(counts[counts >= 2].index)]
    lr = multivariate_logrank_test(df['time'], df['cluster'], df['event'])
    v = -np.log10(max(lr.p_value, 1e-300))
    eps.append(ep); negl.append(v)
    print(f"ep {ep:4d}  -log10p={v:6.3f}  (p={lr.p_value:.2e}, n={len(df)}, "
          f"clusters={df['cluster'].nunique()})")
    del m, ck

valid = [(e, v) for e, v in zip(eps, negl) if np.isfinite(v)]
best_ep, best_v = max(valid, key=lambda t: t[1])
print(f"\nbest -log10p={best_v:.3f} at epoch {best_ep}")
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(eps, negl, marker='o', ms=3, lw=1.2)
ax.axhline(-np.log10(0.05), ls=':', c='red', lw=1.2, label='p = 0.05')
ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax.grid(which='major', alpha=0.45)
ax.grid(which='minor', alpha=0.20, ls=':')
ax.set_xlabel("Epoch"); ax.set_ylabel(f"-log10 p ({a.endpoint} log-rank)")
ax.set_title(f"{a.cancer} — survival separation vs epoch ({a.model_dir.split('/')[-1]})")
ax.legend()
plt.tight_layout()
plt.savefig(f"results/sweep_survival_{a.cancer}_{a.endpoint}.png", dpi=200)