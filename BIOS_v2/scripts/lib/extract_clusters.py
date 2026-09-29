"""
extract_clusters.py — load a trained BIOS checkpoint, write clusters.csv.

Usage:
    python extract_clusters.py --cancer BRCA --version v2_... --config <path>
"""
import argparse, os, sys
from datetime import datetime
import pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import omics, subtyping_dir, config_template
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature

ap = argparse.ArgumentParser()
ap.add_argument("--cancer", required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--method", default="BIOS")
a = ap.parse_args()

with open(a.config) as fh:
    cfg_run = yaml.safe_load(fh)
cfg = argparse.Namespace(**yaml_config_hook(config_template()))

ckpt_fp = f"{cfg_run['model_path']}/checkpoint_{cfg_run['epochs']}.tar"
ck = torch.load(ckpt_fp, map_location='cpu', weights_only=False)

ae = AE(hid_dim=cfg.feature_dim, bio_dim=ck['bio_dim'])
mdl = network.Network(ae=ae, feature_dim=cfg.feature_dim,
                      class_num=cfg_run['cluster_number'], bio_dim=ck['bio_dim'],
                      n_anchors=ck['n_anchors'], head_type=ck['head_type'])
mdl.load_state_dict(ck['net']); mdl.eval()

ids = pd.read_csv(omics(a.cancer, "CN"), header=0, index_col=0, sep=',').columns.tolist()
DL = get_feature(a.cancer, cfg_run['batch_size'], False,
                 bio_anchor_file=cfg_run['bio_anchor_file'])
hs = []
with torch.no_grad():
    for b in DL:
        x = b[0] if isinstance(b, (list, tuple)) else b
        c, h, z = mdl.forward_cluster(x)
        hs.append(c.view(-1))
pred = torch.cat(hs).numpy()
if len(pred) != len(ids):
    print(f"WARNING: {len(pred)} predictions vs {len(ids)} patients in CN.fea")
    raise SystemExit("cohort/anchor mismatch — run align_cohort.py first")

outdir = subtyping_dir(a.cancer, a.method, a.version)
os.makedirs(outdir, exist_ok=True)
out_fp = f"{outdir}/clusters.csv"
with open(out_fp, 'w') as fh:
    fh.write(f"# method: {a.method}\n")
    fh.write(f"# cancer: {a.cancer}\n")
    fh.write(f"# version: {a.version}\n")
    fh.write(f"# config: {a.config}\n")
    fh.write(f"# checkpoint: {ckpt_fp}\n")
    fh.write(f"# generated: {datetime.now():%Y-%m-%d %H:%M}\n")
    pd.DataFrame({'patient_id': ids, 'cluster': pred}).to_csv(fh, index=False)

print(f"saved {out_fp}  ({len(ids)} patients, "
      f"{len(set(pred))} clusters, sizes {pd.Series(pred).value_counts().sort_index().to_dict()})")