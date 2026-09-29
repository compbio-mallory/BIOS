"""Fill the 12x5 BRCA ablation table: V-measure + survival from final checkpoints."""
import sys, os, re, glob, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from modules import network
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature
from sklearn.metrics import v_measure_score, adjusted_rand_score
from lifelines.statistics import multivariate_logrank_test

cfg = argparse.Namespace(**yaml_config_hook("config/config.yaml"))
CANCER, K = 'BRCA', 5

fea = pd.read_csv(f'../subtype_file/fea/{CANCER}/CN.fea', header=0, index_col=0, sep=',')
ids = fea.columns.tolist()

gt = pd.read_csv(f'data/ground_truth/ground_truth_{CANCER}.csv')
gt = gt.set_index('patient_id' if 'patient_id' in gt.columns else gt.columns[0])
lab = 'numeric_label' if 'numeric_label' in gt.columns else gt.columns[-1]

clin = pd.read_csv('data/labels/brca_survival_cdr.csv')
bc = 'bcr_patient_barcode' if 'bcr_patient_barcode' in clin.columns else clin.columns[0]
clin = clin[[bc, 'OS', 'OS.time']].copy()
clin.columns = ['patient', 'event', 'time']
clin['patient'] = clin['patient'].astype(str).str.upper().str.replace('_', '-').str[:12]
clin['event'] = pd.to_numeric(clin['event'], errors='coerce')
clin['time'] = pd.to_numeric(clin['time'], errors='coerce')
clin = clin.dropna(subset=['event', 'time'])
clin = clin[clin['time'] >= 0]

rows = []
for d in sorted(glob.glob('save/head_*/')):
    name = os.path.basename(d.rstrip('/'))
    m = re.match(r'head_(\w+?)_biodim(\d+)_pathways(\d+)', name)
    head, bd, pw = m.group(1), int(m.group(2)), int(m.group(3))
    anchors = ('data/bio_anchors/bio_anchors_BRCA_gsva_decorr_M4.csv' if pw == 4
               else 'data/bio_anchors/bio_anchors_BRCA_gsva_H40.csv')

    ck = torch.load(f'{d}/checkpoint_600.tar', map_location='cpu', weights_only=False)
    ae = AE(hid_dim=cfg.feature_dim, bio_dim=ck['bio_dim'])
    mdl = network.Network(ae=ae, feature_dim=cfg.feature_dim, class_num=K,
                          bio_dim=ck['bio_dim'], n_anchors=ck['n_anchors'],
                          head_type=ck['head_type'])
    mdl.load_state_dict(ck['net']); mdl.eval()

    DL = get_feature(CANCER, 64, False, bio_anchor_file=anchors)
    hs = []
    with torch.no_grad():
        for b in DL:
            x = b[0] if isinstance(b, (list, tuple)) else b
            c, h, z = mdl.forward_cluster(x); hs.append(c.view(-1))
    pred = pd.Series(torch.cat(hs).numpy(), index=ids)

    common = gt.index.intersection(pred.index)
    v = v_measure_score(gt.loc[common, lab], pred.loc[common])
    ar = adjusted_rand_score(gt.loc[common, lab], pred.loc[common])

    sp = pd.DataFrame({'patient': ids, 'cluster': pred.values}).merge(
        clin, on='patient', how='inner')
    cnt = sp['cluster'].value_counts()
    sp = sp[sp['cluster'].isin(cnt[cnt >= 2].index)]
    lr = multivariate_logrank_test(sp['time'], sp['cluster'], sp['event'])
    nl = -np.log10(max(lr.p_value, 1e-300))

    pd.DataFrame({'patient_id': pred.index, 'cluster': pred.values}).to_csv(
        f'results/{name}_clusters.csv', index=False)
    rows.append((head, bd, pw, v, ar, nl, len(sp)))
    del mdl, ck

print(f"\n{'head':10s} {'bio_dim':>7s} {'pathways':>8s} {'V':>7s} {'ARI':>7s} "
      f"{'-log10p':>8s} {'n':>5s}")
print('-' * 60)
order = {'linear': 0, 'mlp': 1, 'attention': 2}
for r in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1], r[2])):
    print(f"{r[0]:10s} {r[1]:7d} {r[2]:8d} {r[3]:7.4f} {r[4]:7.4f} {r[5]:8.3f} {r[6]:5d}")

out = pd.DataFrame(rows, columns=['head', 'bio_dim', 'pathways',
                                  'V_measure', 'ARI', 'survival_neglog10p', 'n'])
out.to_csv('results/ablation_brca_table.csv', index=False)
print("\nsaved results/ablation_brca_table.csv")