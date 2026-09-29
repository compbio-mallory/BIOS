# BIOS HPC Structure
## Tree (code only, data/omics excluded)
```
./BIOS_structure.md
./Subtype-DCC/README.md
./Subtype-DCC/archive/ae_3dim.py
./Subtype-DCC/archive/dataloader_original.py
./Subtype-DCC/archive/train.py
./Subtype-DCC/archive/train_bioanchor.py
./Subtype-DCC/archive/train_bioanchor_3dim.py
./Subtype-DCC/archive/train_original.py
./Subtype-DCC/compute_gsva_anchors.py
./Subtype-DCC/config/config.yaml
./Subtype-DCC/data/raw/BRCA_clinicalMatrix.txt
./Subtype-DCC/dataloader.py
./Subtype-DCC/generate_fig_correlation.py
./Subtype-DCC/gsva_anchors.py
./Subtype-DCC/modules/ae.py
./Subtype-DCC/modules/contrastive_loss.py
./Subtype-DCC/modules/network.py
./Subtype-DCC/modules/network_OLD.py
./Subtype-DCC/scripts/compute/compute_brca_bioanchors.py
./Subtype-DCC/scripts/compute/compute_brca_bioanchors_fixed.py
./Subtype-DCC/scripts/compute/compute_pathway_anchors.py
./Subtype-DCC/scripts/compute/compute_receptor_status.py
./Subtype-DCC/scripts/evaluate/ae.py
./Subtype-DCC/scripts/evaluate/contrastive_loss.py
./Subtype-DCC/scripts/evaluate/dataloader.py
./Subtype-DCC/scripts/evaluate/evaluate_clustering.py
./Subtype-DCC/scripts/evaluate/evaluate_clustering_OLD.py
./Subtype-DCC/scripts/evaluate/network.py
./Subtype-DCC/scripts/train/train_baseline_OLD.py
./Subtype-DCC/scripts/train/train_bioanchor.py
./Subtype-DCC/scripts/train/train_bioanchor_attention_OLD.py
./Subtype-DCC/scripts/train/train_bioanchor_linear_OLD.py
./Subtype-DCC/scripts/train/train_bioanchor_mlp_OLD.py
./Subtype-DCC/scripts/utils/check_gene_names.py
./Subtype-DCC/scripts/utils/create_synthetic_dataset.py
./Subtype-DCC/scripts/utils/download_tcga_clinical.py
./Subtype-DCC/scripts/utils/generate_dummy_bioanchors.py
./Subtype-DCC/scripts/utils/get_brca_ground_truth.py
./Subtype-DCC/slurm/job_600ep_apoptosis.slurm
./Subtype-DCC/slurm/job_600ep_hypoxia.slurm
./Subtype-DCC/slurm/job_attention_OLD.slurm
./Subtype-DCC/slurm/job_baseline.slurm
./Subtype-DCC/slurm/job_bioanchor.slurm
./Subtype-DCC/slurm/job_bioanchor_3dim.slurm
./Subtype-DCC/slurm/job_bioanchor_attention_new.slurm
./Subtype-DCC/slurm/job_bioanchor_linear.slurm
./Subtype-DCC/slurm/job_bioanchor_linear_4dim_200ep.slurm
./Subtype-DCC/slurm/job_bioanchor_linear_best.slurm
./Subtype-DCC/slurm/job_bioanchor_linear_new.slurm
./Subtype-DCC/slurm/job_bioanchor_mlp.slurm
./Subtype-DCC/slurm/job_bioanchor_mlp_4dim_200ep.slurm
./Subtype-DCC/slurm/job_bioanchor_mlp_best.slurm
./Subtype-DCC/slurm/job_bioanchor_mlp_new.slurm
./Subtype-DCC/slurm/job_combo_sweep.slurm
./Subtype-DCC/slurm/job_combo_sweep_1.slurm
./Subtype-DCC/slurm/job_combo_sweep_2.slurm
./Subtype-DCC/slurm/job_gsva_attn_M8.slurm
./Subtype-DCC/slurm/job_gsva_attn_decorr_M4.slurm
./Subtype-DCC/slurm/job_gsva_decorr_M4_attn_v2.slurm
./Subtype-DCC/slurm/job_gsva_decorr_M4_attn_v3.slurm
./Subtype-DCC/slurm/job_gsva_decorr_M4_mlp_v2.slurm
./Subtype-DCC/slurm/job_gsva_decorr_M4_mlp_v3.slurm
./Subtype-DCC/slurm/job_gsva_linear_M8.slurm
./Subtype-DCC/slurm/job_gsva_linear_decorr_M4.slurm
./Subtype-DCC/slurm/job_gsva_mlp_M8.slurm
./Subtype-DCC/slurm/job_gsva_mlp_decorr_M4.slurm
./Subtype-DCC/slurm/job_lambda_test.slurm
./Subtype-DCC/slurm/job_mlp_OLD.slurm
./Subtype-DCC/slurm/job_mlp_apoptosis_200ep.slurm
./Subtype-DCC/slurm/job_mlp_hypoxia_200ep.slurm
./Subtype-DCC/slurm/job_repro_linear_OLD.slurm
./Subtype-DCC/utils/__init__.py
./Subtype-DCC/utils/save_model.py
./Subtype-DCC/utils/yaml_config_hook.py
./Subtype-GAN/GAN_vs_VAE.py
./Subtype-GAN/README.md
./Subtype-GAN/SubtypeGAN.py
./Subtype-GAN/sgan.yml
./job.slurm
```
## Key dirs
### .
21
.claude
.git
.gitignore
BIOS_structure.md
Subtype-DCC
Subtype-GAN
bios_project.zip
job.slurm
subtype_file

### Subtype-DCC
792
.DS_Store
.git
.gitignore
LICENSE
README.md
__pycache__
archive
compute_gsva_anchors.py
config
data
dataloader.py
docs
fig_anchor_correlation.png
generate_fig_correlation.py
gsva_anchor_scores_real.csv
gsva_anchors.py
gsva_cache_BRCA_M8.pkl
gsva_cache_BRCA_decorr_M4.pkl
logs
modules
results
save
scripts
slurm
utils

### scripts

### scripts/evaluate

### slurm

### data

## --- TRAIN script (head) ---
```python
"""
train_bioanchor_attention_OLD.py
=================================
Self-attention bio head using the OLD architecture:
- network_OLD.py for encoder + projectors (unchanged)
- bio_head is a SEPARATE object with its own bio_optimizer
- Only the bio head uses self-attention instead of linear

This is the old separate-optimizer pattern but with attention
instead of linear, to isolate whether attention itself helps
independent of the shared/separate optimizer question.

Compare results against:
    train_bioanchor_linear_OLD.py  → V=0.4855  (separate optimizer, linear)
    train_bioanchor.py attention   → V=0.4447  (shared optimizer, attention)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from modules import network_OLD as network, contrastive_loss
from utils import yaml_config_hook
from torch import optim
from dataloader import get_feature
import matplotlib.pyplot as plt


# ── Self-attention bio head (defined inline, no changes to network_OLD.py) ──

class BioAnchorHeadAttention(nn.Module):
    """
    Self-attention bio anchor head.
    Treats each bio dim as its own token — attention learns
    which anchors are relevant to predicting each other.

    Used here with the OLD separate-optimizer architecture
    to isolate attention's effect independent of optimizer choice.
    """
    def __init__(self, bio_dim, d_model=32, num_heads=2):
        super().__init__()
        self.token_proj  = nn.Linear(1, d_model)
        self.attn        = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = 0.0,
            batch_first = True,
        )
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, z_bio):
        # (batch, N) → (batch, N, d_model) → (batch, N) 
        x = self.token_proj(z_bio.unsqueeze(-1))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.output_proj(x).squeeze(-1)


# ── Utilities (same as old train scripts) ──

def inference(loader, model, device):
    model.eval()
    cluster_vector, feature_vector = [], []
    for step, batch_data in enumerate(loader):
        if len(batch_data) == 2:
            x, _ = batch_data
        else:
            x = batch_data[0]
        x = x.float().to(device)
        with torch.no_grad():
            z, _, _ = model.ae(x)
            c, h = model.forward_cluster(x)
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    print("Features shape {}".format(np.array(feature_vector).shape))
    return np.array(cluster_vector), np.array(feature_vector)


def draw_fig(loss, cancer_type, epoch):
    plt.figure()
    plt.plot(range(len(loss)), loss, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('Train loss')
    plt.title('OLD arch + Self-Attention — Train loss vs. epoch')
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{cancer_type}_attention_old_loss.png')
    plt.close()


def save_model(args, model, optimizer, current_epoch):
    os.makedirs(args.model_path, exist_ok=True)
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    torch.save({
        'net':       model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     current_epoch,
    }, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--cancer_type",     "-c", type=str,   required=True)
    parser.add_argument("--batch_size",            type=int,   default=64)
    parser.add_argument("--cluster_number",        type=int,   required=True)
    parser.add_argument("--lambda_bio",            type=float, default=0.1)
    parser.add_argument("--bio_dim",               type=int,   required=True)
    parser.add_argument("--bio_anchor_file",       type=str,   required=True)
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
```
## --- EVALUATE script (head) ---
```python
"""
evaluate_clustering.py  —  evaluation script for BIOS models
=============================================================
Changes from old version
-------------------------
    OLD: from network import Network       wrong import path
         from ae import AE                 wrong import path
         bio_dim=15                         hardcoded
         network.Network(ae, feature_dim, cluster_num)  old 3-arg signature
         c, h = model.forward_cluster(x)   old 2-value unpack
         no --bio_dim, --head_type args

    NEW: from modules.network import ...   correct import
         from modules.ae import AE          correct import
         bio_dim and head_type read from checkpoint (we saved them there)
         Network() called with all required args
         c, h, z_bio = model.forward_cluster(x)  correct 3-value unpack
         --cluster_number arg added
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# OLD: from network import Network
# OLD: from ae import AE
# NEW: correct module paths
from modules.network import Network
from modules.ae import AE
from dataloader import get_feature


def load_model(model_path, feature_dim, cluster_num, device):
    """
    Load trained model from checkpoint.
    bio_dim and head_type are read from the checkpoint itself —
    no need to pass them as arguments since we saved them during training.
    """
    # load checkpoint first to read architecture params
    checkpoint = torch.load(model_path, map_location=device)

    # OLD: bio_dim=15 hardcoded
    # NEW: read from checkpoint — saved by train_bioanchor.py
    bio_dim   = checkpoint.get('bio_dim',   14)    # fallback 14 if old checkpoint
    head_type = checkpoint.get('head_type', 'mlp') # fallback mlp if old checkpoint

    ae = AE(hid_dim=feature_dim, bio_dim=bio_dim)

    # OLD: model = network.Network(ae, feature_dim, cluster_num)  ← 3 args, crashes
    # NEW: all required args passed explicitly
    model = Network(
        ae          = ae,
        feature_dim = feature_dim,
        class_num   = cluster_num,
        bio_dim     = bio_dim,
        n_anchors   = bio_dim,
        head_type   = head_type,
    )

    model.load_state_dict(checkpoint['net'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"   head_type={head_type}, bio_dim={bio_dim}")
    return model


def get_predictions(model, dataloader, device):
    """Get cluster predictions and embeddings from model."""
    all_predictions = []
    all_embeddings  = []

    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data[0]

            x = x.float().to(device)

            z, z_bio, z_novel = model.ae(x)

            # OLD: c, h = model.forward_cluster(x)   ← 2 values, crashes
            # NEW: forward_cluster returns 3 values
            c, h, z_bio_out = model.forward_cluster(x)

            all_predictions.extend(c.cpu().numpy())
            all_embeddings.extend(z.cpu().numpy())

    return np.array(all_predictions), np.array(all_embeddings)


def evaluate_clustering(pred_labels, true_labels):
    """Compute clustering metrics."""
    return {
        'ARI':       adjusted_rand_score(true_labels, pred_labels),
        'NMI':       normalized_mutual_info_score(true_labels, pred_labels),
        'V-measure': v_measure_score(true_labels, pred_labels),
    }


def plot_confusion_matrix(pred_labels, true_labels, cancer_type):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Pred {i}' for i in range(cm.shape[1])],
                yticklabels=[f'True {i}' for i in range(cm.shape[0])])
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Cluster')
    plt.title(f'Confusion Matrix - {cancer_type}')
    plt.tight_layout()
    plt.savefig(f'results/{cancer_type}_confusion_matrix.png', dpi=150)
    print(f"Saved confusion matrix to results/{cancer_type}_confusion_matrix.png")

```
## --- DATALOADER (head) ---
```python
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def get_feature(cancer_type, batch_size, training, bio_anchor_file=None):
    """
    Load multi-omics features and optionally bio-anchors.

    Parameters
    ----------
    cancer_type    : str   e.g. "BRCA", "LUAD"
    batch_size     : int
    training       : bool  shuffle if True
    bio_anchor_file: str or None
                     Path to bio-anchors CSV (e.g. "bio_anchors_BRCA_6dim.csv").
                     If None, no bio-anchors are loaded and dataloader returns
                     only omics features.
    """
    # ── Load omics features ───────────────────────────────────────────────────
    base = f'../subtype_file/fea/{cancer_type}'
    fea_CN    = pd.read_csv(f'{base}/CN.fea',    header=0, index_col=0, sep=',')
    fea_meth  = pd.read_csv(f'{base}/meth.fea',  header=0, index_col=0, sep=',')
    fea_mirna = pd.read_csv(f'{base}/miRNA.fea', header=0, index_col=0, sep=',')
    fea_rna   = pd.read_csv(f'{base}/rna.fea',   header=0, index_col=0, sep=',')

    feature = np.concatenate((fea_CN, fea_meth, fea_mirna, fea_rna), axis=0).T
    feature = MinMaxScaler().fit_transform(feature)
    feature = torch.tensor(feature, dtype=torch.float32)

    # ── Load bio-anchors (optional) ───────────────────────────────────────────
    anchor_values = None
    if bio_anchor_file is not None:
        if not os.path.exists(bio_anchor_file):
            print(f"[dataloader] Warning: bio_anchor_file not found: {bio_anchor_file}")
        else:
            bio_anchors = pd.read_csv(bio_anchor_file)
            patient_ids = fea_CN.columns.tolist()
            bio_anchors = bio_anchors.set_index('patient_id').loc[patient_ids].reset_index()
            anchor_values = torch.tensor(bio_anchors.iloc[:, 1:].values, dtype=torch.float32)
            print(f"[dataloader] Loaded bio-anchors from {bio_anchor_file}: {anchor_values.shape}")
    else:
        print(f"[dataloader] No bio_anchor_file specified — loading omics only")

    # ── Build dataset and dataloader ──────────────────────────────────────────
    dataset = TensorDataset(feature, anchor_values) if anchor_values is not None else TensorDataset(feature)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training)```
## --- one example SLURM ---
```bash
#!/bin/bash
#SBATCH --job-name=scgclust_rna_breast
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q



PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

echo "=== Done ==="
date
```
