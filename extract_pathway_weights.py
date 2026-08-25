"""
extract_pathway_weights.py — pull per-subtype pathway gate weights out of a
trained v3 checkpoint and run the two acceptance tests from V3_INTEGRATION.md.

Test 1 (weak)  : do clusters select different pathways?
Test 2 (real)  : does the gate agree with the independent per-cluster enrichment
                 result? Reported as Spearman rho between gate rank and
                 -log10(q) rank, per cluster.

Outputs:
  {out}_gate_matrix.csv     K x A per-subtype pathway weights
  {out}_top_pathways.csv    ranked top-N per cluster, with enrichment q alongside
  {out}_report.txt          differentiation + alignment verdict

ADAPTER: the loader block below is the only part that depends on your repo API.
Edit `load_model_and_data` if the signatures have drifted.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# ADAPTER — edit to match your repo
# ---------------------------------------------------------------------------
def load_model_and_data(args):
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)

    import torch
    from modules import network
    from modules.ae import AE
    from utils import yaml_config_hook
    from dataloader import get_feature

    cfg = yaml_config_hook(os.path.join(args.repo, "config/config.yaml"))

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head_type  = ck.get("head_type", "attention_query_v3")
    bio_dim    = ck["bio_dim"]
    n_clusters = ck.get("n_clusters", args.cluster_number)

    anchors_df   = pd.read_csv(args.anchors)
    anchor_names = [c.replace("HALLMARK_", "")
                    for c in anchors_df.columns if c != "patient_id"]
    n_anchors    = ck.get("n_anchors", len(anchor_names))

    ae = AE(hid_dim=cfg["feature_dim"], bio_dim=bio_dim)
    model = network.Network(
        ae=ae,
        feature_dim=cfg["feature_dim"],
        class_num=n_clusters,
        bio_dim=bio_dim,
        n_anchors=n_anchors,
        head_type=head_type,
    )
    model.load_state_dict(ck["net"])
    model.eval()

    # patient order == fea_CN column order, same as the dataloader uses
    fea_CN = pd.read_csv(f"../subtype_file/fea/{args.cancer}/CN.fea",
                         header=0, index_col=0, sep=",")
    patients = fea_CN.columns.tolist()

    # shuffle=False so batch order matches `patients`
    DL = get_feature(args.cancer, batch_size=256, training=False,
                     bio_anchor_file=args.anchors)

    z_bio_all, c_soft_all = [], []
    with torch.no_grad():
        for batch in DL:
            x = batch[0].float()
            c_soft, h, z_bio, aux = model.forward_cluster_soft(x)
            # aux['c0'] is the conditioning actually used inside the head
            c_soft_all.append(aux["c0"] if aux is not None else c_soft)
            z_bio_all.append(z_bio)

    z_bio  = torch.cat(z_bio_all, 0)
    c_soft = torch.cat(c_soft_all, 0)

    assert len(patients) == z_bio.size(0), \
        f"patient/embedding mismatch: {len(patients)} vs {z_bio.size(0)}"

    return model, z_bio, c_soft, patients, anchor_names

    # ---- replace with your real omics loader ----
    raise NotImplementedError(
        "Wire load_omics() for the cancer here, run the encoder, and return\n"
        "z_bio = z[:, :bio_dim] and c_soft = cluster_head(z).softmax(-1).detach()."
    )


# ---------------------------------------------------------------------------
def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def cosine_matrix(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn @ Mn.T


EXPECTED = {
    "UCEC": {
        "proliferative/serous/CN-high": ["E2F_TARGETS", "G2M_CHECKPOINT",
                                         "MYC_TARGETS_V1", "MITOTIC_SPINDLE"],
        "endometrioid/CN-low": ["ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE"],
        "MSI/POLE": ["PI3K_AKT_MTOR_SIGNALING", "MTORC1_SIGNALING",
                     "ALLOGRAFT_REJECTION", "INTERFERON_GAMMA_RESPONSE"],
    },
    "BRCA": {
        "basal-like": ["MYC_TARGETS_V1", "G2M_CHECKPOINT", "E2F_TARGETS"],
        "luminal": ["ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE"],
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--enrichment", required=True,
                   help="results/{CANCER}_cluster_enrichment.csv")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--repo", default="/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC")
    p.add_argument("--cluster_number", type=int, default=4)
    p.add_argument("--topn", type=int, default=8)
    a = p.parse_args()

    model, z_bio, c_soft, patients, anchor_names = load_model_and_data(a)

    with torch.no_grad():
        out = model.bio_head(z_bio, c_soft)
        gate = out["gate"]
        W = model.bio_head.cluster_gate_matrix(gate, c_soft, hard=True).numpy()

    K, A = W.shape
    uniform = 1.0 / A
    gate_np = gate.numpy()
    ent = -(gate_np * np.log(gate_np + 1e-12)).sum(1).mean()
    max_ent = np.log(A)

    Wdf = pd.DataFrame(W, index=[f"cluster_{k}" for k in range(K)],
                       columns=anchor_names)
    Wdf.to_csv(f"{a.out}_gate_matrix.csv")

    # ---- Test 1: differentiation ----
    S = cosine_matrix(W)
    off = ~np.eye(K, dtype=bool)
    mean_cos = float(S[off].mean())
    tops = {k: set(Wdf.iloc[k].nlargest(5).index) for k in range(K)}
    identical = all(tops[0] == tops[k] for k in range(K))

    # ---- Test 2: alignment with enrichment ----
    enr = pd.read_csv(a.enrichment)
    cols = {c.lower(): c for c in enr.columns}
    ccol = cols.get("cluster")
    pcol = cols.get("pathway", cols.get("geneset", cols.get("term")))
    qcol = cols.get("q", cols.get("q_value", cols.get("qval", cols.get("padj"))))
    if not all([ccol, pcol, qcol]):
        print(f"[warn] could not identify columns in {a.enrichment}; "
              f"found {list(enr.columns)}. Skipping test 2.", file=sys.stderr)
        rhos = {}
    else:
        enr[pcol] = enr[pcol].str.replace("HALLMARK_", "", regex=False)
        rhos = {}
        for k in range(K):
            sub = enr[enr[ccol].astype(str).str.contains(str(k))]
            sub = sub.set_index(pcol).reindex(anchor_names)
            score = -np.log10(sub[qcol].astype(float).clip(lower=1e-300).values)
            score = np.nan_to_num(score, nan=0.0)
            rhos[k] = spearman(W[k], score)

    # ---- top pathway table ----
    rows = []
    for k in range(K):
        for rank, (pw, wt) in enumerate(Wdf.iloc[k].nlargest(a.topn).items(), 1):
            rows.append({"cluster": k, "rank": rank, "pathway": pw,
                         "gate_weight": wt, "x_uniform": wt / uniform})
    pd.DataFrame(rows).to_csv(f"{a.out}_top_pathways.csv", index=False)

    # ---- report ----
    L = []
    L.append(f"BIOS v3 pathway gate — {a.cancer}")
    L.append(f"checkpoint: {a.checkpoint}")
    L.append(f"K={K}  A={A}  uniform weight={uniform:.4f}")
    L.append(f"mean gate entropy: {ent:.3f} / max {max_ent:.3f} "
             f"({100*ent/max_ent:.0f}% of uniform)")
    L.append("")
    L.append("TEST 1 — differentiation across subtypes (weak test)")
    L.append(f"  mean pairwise cosine between cluster gates: {mean_cos:.3f}")
    L.append(f"  top-5 sets identical across all clusters: {identical}")
    L.append(f"  verdict: {'FAIL' if (mean_cos > 0.9 or identical) else 'PASS'}")
    L.append("")
    L.append("TEST 2 — agreement with independent enrichment (the real test)")
    for k, r in rhos.items():
        L.append(f"  cluster {k}: spearman rho = {r:.3f}")
    if rhos:
        mr = float(np.nanmean(list(rhos.values())))
        L.append(f"  mean rho = {mr:.3f}")
        L.append(f"  verdict: {'PASS' if mr > 0.4 else 'FAIL'}")
    L.append("")
    L.append(f"Expected biology for {a.cancer}:")
    for name, pws in EXPECTED.get(a.cancer, {}).items():
        L.append(f"  {name}: {', '.join(pws)}")
    L.append("")
    L.append("Top pathways per cluster:")
    for k in range(K):
        top = Wdf.iloc[k].nlargest(5)
        L.append(f"  cluster {k}: " +
                 ", ".join(f"{i} ({v/uniform:.1f}x)" for i, v in top.items()))
    L.append("")
    L.append("If TEST 2 fails: attention is exhausted. Report the per-cluster")
    L.append("enrichment result instead — it recovers known biology at q=1e-50")
    L.append("in both cancers and does not depend on the attention head.")

    txt = "\n".join(L)
    with open(f"{a.out}_report.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
