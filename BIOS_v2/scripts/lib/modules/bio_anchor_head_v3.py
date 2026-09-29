"""
bio_anchor_head_v3.py — cluster-conditioned bio-anchor head for BIOS.

WHAT CHANGED FROM v2 (attention_query_v2), AND WHY
--------------------------------------------------
Two independent defects, both fixed here.

(1) SUBTYPE-BLINDNESS (the one you and Xian identified).
    v1/v2 used a fixed set of anchor queries applied identically to every
    patient. The only patient-specific input was the bio-summary slice, so the
    model could only ever learn ONE global attention strategy. v3 conditions the
    queries on the soft cluster assignment, so different subtypes attend
    differently. Cluster probs are DETACHED when used for conditioning, which
    breaks the clusters -> attention -> embedding -> clusters loop.

(2) THE SOFTMAX WAS OVER THE WRONG AXIS.
    In v2, attention was normalised over the bio_dim slots: with bio_dim=16 the
    maximum entropy is ln(16) = 2.7726, which is exactly the 2.77 reported. So
    the object being inspected was 50 separate 16-way distributions over latent
    slots — it cannot express "pathway P matters for subtype k", because
    pathways are the axis the softmax never touched. Concentrating it (2.77 ->
    1.33) therefore concentrated the wrong thing, which is consistent with the
    selected pathways coming out biologically random.

    v3 adds an explicit PATHWAY GATE: a softmax over the anchor axis, giving
    w in R^{B x A} that sums to 1 across pathways per patient. This is the
    interpretable object. Averaged within cluster it is literally
    "which pathways this subtype selects" — Xian's ask, stated directly.

    The per-anchor slot attention is retained and returned, but only as a
    diagnostic. Do not report it as pathway importance.

The gate must influence the loss or it learns nothing, so it feeds a small
residual (`fb`) back into the instance/cluster branch. See V3_INTEGRATION.md for
the two-pass forward that keeps this acyclic.

Init note: cluster_delta starts at zero, so at step 0 v3 is behaviourally
identical to a query head with no cluster conditioning. Differentiation is
learned, not imposed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioAnchorHeadV3(nn.Module):
    """
    Args
    ----
    bio_dim      : width of the bio-summary slice of the embedding (16 or 32)
    n_anchors    : number of pathway anchors (40 after filtering)
    n_clusters   : K (UCEC=4, BRCA=5)
    d_model      : internal attention width (32 is plenty)
    feedback_dim : width of the residual fed back to the instance branch;
                   set to the instance-projector input width
    """

    def __init__(self, bio_dim, n_anchors, n_clusters,
                 d_model=32, feedback_dim=None,
                 tau_slot_init=1.0, tau_path_init=1.0, alpha_init=0.5):
        super().__init__()
        self.head_type = 'attention_query_v3'
        self.bio_dim = bio_dim
        self.n_anchors = n_anchors
        self.n_clusters = n_clusters
        self.d_model = d_model

        # --- bio-summary slots -> tokens (keys/values) ---
        self.slot_emb = nn.Parameter(torch.randn(bio_dim, d_model) * 0.02)
        self.slot_bias = nn.Parameter(torch.zeros(bio_dim, d_model))
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # --- queries: one per pathway, modulated by cluster ---
        self.anchor_queries = nn.Parameter(torch.randn(n_anchors, d_model) / math.sqrt(d_model))
        self.cluster_delta = nn.Parameter(torch.randn(n_clusters, n_anchors, d_model) * 0.02)
        self.cluster_emb = nn.Parameter(torch.zeros(n_clusters, d_model))
        self.gate_norm = nn.LayerNorm(d_model)

        # --- anchor regression readout (MSE vs GSVA) ---
        self.anchor_out = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
        self.anchor_bias = nn.Parameter(torch.zeros(n_anchors))

        # --- pathway gate readout ---
        self.path_vec = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)

        # --- learnable temperatures (log-parameterised, clamped at use) ---
        self.log_tau_slot = nn.Parameter(torch.tensor(math.log(tau_slot_init)))
        self.log_tau_path = nn.Parameter(torch.tensor(math.log(tau_path_init)))

        # --- residual feedback so the gate is in the gradient path ---
        self.feedback_dim = feedback_dim
        if feedback_dim is not None:
            self.gate_proj = nn.Linear(d_model, feedback_dim)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.gate_proj = None

    # ------------------------------------------------------------------
    def forward(self, z_bio, c_soft):
        """
        z_bio  : (B, bio_dim)   bio-anchored slice of the embedding
        c_soft : (B, K)         soft cluster probs. CALLER MUST PASS DETACHED.

        Returns dict with:
          anchor_pred (B, A)  -> MSE target is the GSVA anchor matrix
          gate        (B, A)  -> softmax OVER PATHWAYS. the interpretable output.
          slot_att    (B, A, D) -> diagnostic only, softmax over bio slots
          fb          (B, feedback_dim) or None -> residual for instance branch
        """
        B = z_bio.size(0)
        d = self.d_model
        delta = torch.einsum("bk,kad->bad", c_soft, self.cluster_delta)
        ctx = c_soft @ self.cluster_emb                      # (B, d)

        if c_soft.requires_grad:
            raise RuntimeError(
                "c_soft must be detached before entering BioAnchorHeadV3. "
                "Pass c.detach() — see V3_INTEGRATION.md."
            )

        # tokens: (B, D, d)
        T = z_bio.unsqueeze(-1) * self.slot_emb.unsqueeze(0) + self.slot_bias.unsqueeze(0)
        Kt = self.Wk(T)
        Vt = self.Wv(T)

        # cluster-conditioned queries: (B, A, d)
        delta = torch.einsum("bk,kad->bad", c_soft, self.cluster_delta)
        ctx = c_soft @ self.cluster_emb                      # (B, d)
        Q = self.anchor_queries.unsqueeze(0) + delta + ctx.unsqueeze(1)

        # slot attention (diagnostic axis): (B, A, D)
        tau_s = self.log_tau_slot.exp().clamp(0.1, 10.0)
        slot_logits = torch.einsum("bad,bjd->baj", Q, Kt) / (math.sqrt(d) * tau_s)
        slot_att = slot_logits.softmax(dim=-1)
        ctx_a = torch.einsum("baj,bjd->bad", slot_att, Vt)   # (B, A, d)

        # anchor regression
        anchor_pred = (ctx_a * self.anchor_out.unsqueeze(0)).sum(-1) + self.anchor_bias

        # PATHWAY GATE — softmax over the anchor axis: (B, A)
        tau_p = self.log_tau_path.exp().clamp(0.1, 10.0)
        gate_logits = (Q * self.gate_norm(ctx_a)).sum(-1) / (math.sqrt(d) * tau_p)
        gate = gate_logits.softmax(dim=-1)

        # gate-weighted pathway summary -> residual feedback
        fb = None
        if self.gate_proj is not None:
            bio_ctx = torch.einsum("ba,ad->bd", gate * anchor_pred, self.path_vec)
            fb = self.alpha * self.gate_proj(bio_ctx)

        return {
            "anchor_pred": anchor_pred,
            "gate": gate,
            "gate_logits": gate_logits,
            "slot_att": slot_att,
            "fb": fb,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def cluster_gate_matrix(self, gate, c_soft, hard=True):
        """
        Per-subtype pathway weights: (K, A), rows sum to 1.
        hard=True  -> argmax cluster assignment (what you report)
        hard=False -> soft-prob weighted average
        """
        if hard:
            idx = c_soft.argmax(dim=1)
            W = torch.zeros(self.n_clusters, self.n_anchors, device=gate.device)
            for k in range(self.n_clusters):
                m = idx == k
                if m.any():
                    W[k] = gate[m].mean(0)
            return W
        num = c_soft.t() @ gate
        den = c_soft.sum(0).unsqueeze(1).clamp_min(1e-8)
        return num / den


# ----------------------------------------------------------------------
# Regularisers
# ----------------------------------------------------------------------
def gate_entropy_loss(gate, target_entropy=2.08, eps=1e-8):
    """Hinge: penalise entropy only ABOVE target. ln(8)=2.08 ~ 8 effective
    pathways. Unbounded minimisation collapses the gate to one-hot."""
    H = -(gate * (gate + eps).log()).sum(-1).mean()
    return torch.relu(H - target_entropy)


def gate_diversity_loss(gate, c_soft, n_clusters, eps=1e-8):
    """
    Penalise similarity between per-cluster mean gates, so subtypes select
    different pathways. Returns mean pairwise cosine similarity over k != l.

    USE A SMALL WEIGHT (see V3_INTEGRATION.md). This directly optimises the
    differentiation metric you also evaluate, so differentiation alone is NOT
    evidence the fix worked — biological alignment of the selected pathways is
    the independent check.
    """
    num = c_soft.t() @ gate                                   # (K, A)
    den = c_soft.sum(0).unsqueeze(1).clamp_min(eps)
    M = num / den
    present = (c_soft.sum(0) > 1e-3)
    if present.sum() < 2:
        return gate.new_zeros(())
    M = M[present]
    M = F.normalize(M, dim=1)
    S = M @ M.t()
    k = M.size(0)
    off = ~torch.eye(k, dtype=torch.bool, device=S.device)
    return torch.relu(S[off].mean() - 0.7)


def v3_losses(out, anchor_target, c_soft, n_clusters,
              lambda_bio=1.0, lambda_entropy=0.05, lambda_div=0.02):
    """Convenience wrapper. Add the returned 'total' to your contrastive +
    cluster losses. All components returned for logging."""
    mse = F.mse_loss(out["anchor_pred"], anchor_target)
    ent = gate_entropy_loss(out["gate"])
    div = gate_diversity_loss(out["gate"], c_soft, n_clusters)
    total = lambda_bio * mse + lambda_entropy * ent + lambda_div * div
    return {
        "total": total,
        "bio_mse": mse.detach(),
        "gate_entropy": ent.detach(),
        "gate_diversity": div.detach(),
    }
