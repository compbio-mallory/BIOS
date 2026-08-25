"""
network.py  —  BIOS model architecture
=======================================
Contains all three bio-anchor head variants in one place.

Head variants (select via --head_type argument):
    'linear'    Scenario 1 — single linear layer         (most interpretable)
    'mlp'       Scenario 2 — two-layer MLP               (balanced)
    'attention' Scenario 3 — multi-head self-attention   (most flexible)

What changed from the old network.py
--------------------------------------
    OLD: BioAnchorHead was a separate class with only a single nn.Linear
         BioAnchorHeadMLP was defined inside train_bioanchor_mlp.py (wrong place)
         Network ignored z_bio and z_novel returned by the ae
         Network.forward returned 4 values: z_i, z_j, c_i, c_j
         Network.forward_cluster returned 2 values: c, h

    NEW: All three head variants live here in one BioAnchorHead class
         Network accepts bio_dim, n_anchors, head_type in __init__
         Network.forward returns 5 values: z_i, z_j, c_i, c_j, b_hat
         Network.forward_cluster returns 3 values: c, h, z_bio
"""

import math
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from modules.bio_anchor_head_v3 import (
    BioAnchorHeadV3, gate_entropy_loss, gate_diversity_loss
)


# ══════════════════════════════════════════════════════════════════
#  Bio Anchor Head
#  Takes z_bio (the bio partition of the embedding) and predicts
#  bio-anchor values b_hat.
#
#  OLD code (was its own class, linear only):
#      class BioAnchorHead(nn.Module):
#          def __init__(self, bio_dim=15, n_anchors=15):
#              self.predictor = nn.Linear(bio_dim, n_anchors)   # <-- old, linear only
#          def forward(self, z_bio):
#              return self.predictor(z_bio)
#
#  NEW: same class name, now supports all three variants via head_type
# ══════════════════════════════════════════════════════════════════

class BioAnchorHead(nn.Module):

    def __init__(self, bio_dim, n_anchors, head_type):
        """
        Args:
            bio_dim   : number of bio dims in the split embedding (e.g. 5)
            n_anchors : number of bio anchors to predict (usually == bio_dim)
            head_type : 'linear' | 'mlp' | 'attention'
        """
        super(BioAnchorHead, self).__init__()

        self.head_type = head_type

        # ── Scenario 1: Linear ──────────────────────────────────
        # Single linear layer. Equivalent to the old BioAnchorHead.
        # Most interpretable — weight matrix directly shows which
        # bio dims contribute to each anchor prediction.
        # z_bio (batch, N) → Linear → b_hat (batch, N)
        if head_type == 'linear':
            self.predictor = nn.Linear(bio_dim, n_anchors)

        # ── Scenario 2: MLP ─────────────────────────────────────
        # Two-layer MLP with ReLU. Was previously defined inline
        # inside train_bioanchor_mlp.py — moved here where it belongs.
        # Can learn non-linear interactions between bio dims
        # e.g. how proliferation and immune scores interact.
        # z_bio (batch, N) → Linear(N,64) → ReLU → Linear(64,N) → b_hat
        elif head_type == 'mlp':
            self.predictor = nn.Sequential(
                nn.Linear(bio_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_anchors),
            )

        # ── Scenario 3: Self-Attention ───────────────────────────
        # Each bio dim is treated as its own token. Attention lets
        # the model learn which bio-anchors are relevant to each
        # other — e.g. Apoptosis <-> Proliferation inverse relation,
        # Hypoxia and EMT co-activating in aggressive subtypes.
        # The learned N×N attention weight matrix is also a free
        # interpretability signal (which anchor attends to which).
        #
        # Flow:
        #   z_bio (batch, N)
        #     unsqueeze   → (batch, N, 1)
        #     token_proj  → (batch, N, d_model)  lift each scalar to d_model dims
        #     self-attn   → (batch, N, d_model)  each token attends to all others
        #     Add & LN    → (batch, N, d_model)  residual + layer norm
        #     output_proj → (batch, N, 1)         collapse back to scalar
        #     squeeze     → (batch, N)            = b_hat
        elif head_type == 'attention':
            d_model   = 32   # token embedding dim; 32 is enough for N=5-8
            num_heads = 2    # 2 attention heads, each with d_k=16

            # lifts each scalar bio dim to a d_model-dimensional token
            self.token_proj = nn.Linear(1, d_model)

            # PyTorch built-in multi-head attention
            # batch_first=True: input shape is (batch, seq_len, dim)
            self.attn = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = num_heads,
                dropout     = 0.0,
                batch_first = True,
            )

            # layer norm for the residual connection
            self.norm = nn.LayerNorm(d_model)

            # collapses each d_model-dim token back to a single scalar
            self.output_proj = nn.Linear(d_model, 1)

        # ── Scenario 4: Query-based Attention (decoupled n_anchors from bio_dim) ──
        # Each of the n_anchors pathways is a learnable QUERY token that
        # cross-attends to the (small) bio_dim summary. Output size = n_anchors,
        # independent of bio_dim. Exposes per-pathway attention for interpretability.
        elif head_type == 'attention_query':
            d_model = 32
            num_heads = 4
            self.aq_d_model = d_model
            # project the bio summary (bio_dim scalars) into d_model tokens (keys/values)
            self.kv_proj = nn.Linear(1, d_model)          # each bio scalar -> token
            # one learnable query per anchor (the 50 "askers")
            self.anchor_queries = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=0.0, batch_first=True)
            self.aq_norm = nn.LayerNorm(d_model)
            self.aq_out = nn.Linear(d_model, 1)           # each attended query -> scalar
            self.last_attn_weights = None                 # stored for interpretability

        # ── Scenario 5: attention_query_v2 (anti-collapse) ──
        # Same query design, but: (a) temperature-sharpened attention,
        # (b) NO bypass — prediction must flow through attention,
        # (c) entropy of attention exposed so training can penalize uniformity.
        elif head_type == 'attention_query_v2':
            d_model = 32
            self.aqv2_d_model = d_model
            self.kv_proj = nn.Linear(1, d_model)
            self.anchor_queries = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
            # manual single-head attention (so we control temperature + read weights cleanly)
            self.q_scale = nn.Parameter(torch.tensor(1.0))   # learnable sharpening
            self.aqv2_out = nn.Linear(d_model, 1)
            self.last_attn_weights = None
            self.last_attn_entropy = None

        else:
            raise ValueError(
                f"Unknown head_type '{head_type}'. "
                f"Choose: 'linear', 'mlp', or 'attention'"
            )

    def forward(self, z_bio):
        """
        Args:
            z_bio : (batch, bio_dim)   — bio partition of the embedding
        Returns:
            b_hat : (batch, n_anchors) — predicted bio-anchor values
        """

        if self.head_type in ('linear', 'mlp'):
            # straight through — same as old BioAnchorHead.forward()
            return self.predictor(z_bio)

        elif self.head_type == 'attention':
            # treat each bio dim as an independent token
            x = z_bio.unsqueeze(-1)            # (batch, N, 1)
            x = self.token_proj(x)             # (batch, N, d_model)

            # self-attention: every token looks at every other token
            # attn_out contains contextualised token representations
            attn_out, _ = self.attn(x, x, x)  # (batch, N, d_model)

            # residual connection + layer norm (stabilises gradients)
            x = self.norm(x + attn_out)        # (batch, N, d_model)

            # collapse each token back to a scalar prediction
            b_hat = self.output_proj(x)        # (batch, N, 1)
            return b_hat.squeeze(-1)           # (batch, N)

        elif self.head_type == 'attention_query':
            B = z_bio.size(0)
            # bio summary -> tokens (keys/values): (B, bio_dim, d_model)
            kv = self.kv_proj(z_bio.unsqueeze(-1))
            # queries: (n_anchors, d_model) -> (B, n_anchors, d_model)
            q = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)
            attn_out, attn_w = self.cross_attn(q, kv, kv, need_weights=True)
            self.last_attn_weights = attn_w.detach()   # (B, n_anchors, bio_dim) for interpretability
            x = self.aq_norm(attn_out)
            b_hat = self.aq_out(x).squeeze(-1)         # (B, n_anchors)
            return b_hat

        elif self.head_type == 'attention_query_v2':
            import math
            B = z_bio.size(0)
            kv = self.kv_proj(z_bio.unsqueeze(-1))          # (B, bio_dim, d_model)
            q = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_anchors, d_model)
            # scaled dot-product attention with LEARNABLE temperature (sharpening)
            scale = (self.aqv2_d_model ** 0.5) / (self.q_scale.abs() + 1e-4)
            scores = torch.bmm(q, kv.transpose(1, 2)) / scale       # (B, n_anchors, bio_dim)
            attn = torch.softmax(scores, dim=-1)                    # (B, n_anchors, bio_dim)
            # entropy of attention per anchor (low = concentrated = good)
            ent = -(attn * (attn + 1e-9).log()).sum(-1)            # (B, n_anchors)
            self.last_attn_entropy = ent.mean()
            self.last_attn_weights = attn.detach()
            # prediction MUST flow through attention (no bypass): weighted sum of values
            ctx = torch.bmm(attn, kv)                               # (B, n_anchors, d_model)
            b_hat = self.aqv2_out(ctx).squeeze(-1)                 # (B, n_anchors)
            return b_hat


# ══════════════════════════════════════════════════════════════════
#  Network
#  Wraps the autoencoder and adds three heads on top:
#    1. instance_projector  → for instance-level contrastive loss
#    2. cluster_projector   → for cluster-level contrastive loss
#    3. bio_head            → for bio-anchor prediction (NEW)
#
#  OLD __init__ signature:
#      def __init__(self, ae, feature_dim, class_num):
#
#  NEW __init__ signature:
#      def __init__(self, ae, feature_dim, class_num,
#                   bio_dim, n_anchors, head_type):
# ══════════════════════════════════════════════════════════════════

class Network(nn.Module):

    def __init__(self, ae, feature_dim, class_num,
                 bio_dim, n_anchors, head_type):
        """
        Args:
            ae          : autoencoder module (from modules/ae.py)
            feature_dim : output dim of instance projector
            class_num   : number of clusters
            bio_dim     : number of bio dims in the embedding split  # NEW
            n_anchors   : number of bio anchors to predict           # NEW
            head_type   : 'linear' | 'mlp' | 'attention'            # NEW
        """
        super(Network, self).__init__()

        self.ae          = ae
        self.feature_dim = feature_dim
        self.cluster_num = class_num

        # ── instance contrastive projector (unchanged from old code) ──
        # maps full embedding h → normalised z for DCL loss
        self.instance_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim),
            nn.ReLU(),
            nn.Linear(self.ae.rep_dim, self.feature_dim),
        )

        # ── cluster projector (unchanged from old code) ──
        # maps full embedding h → soft cluster assignment c
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim),
            nn.ReLU(),
            nn.Linear(self.ae.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

        # ── bio anchor head (NEW) ──
        # maps z_bio → predicted bio-anchor values b_hat
        # head_type selects which of the three variants to use
        #self.bio_head = BioAnchorHead(bio_dim, n_anchors, head_type)

        # ── bio anchor head ──
        if head_type == 'attention_query_v3':
            # v3 needs cluster count (conditioning) and the embedding width
            # (feedback residual). rep_dim is what instance/cluster projectors read.
            self.bio_head = BioAnchorHeadV3(
                bio_dim=bio_dim,
                n_anchors=n_anchors,
                n_clusters=class_num,
                d_model=32,
                feedback_dim=self.ae.rep_dim,
            )
        else:
            self.bio_head = BioAnchorHead(bio_dim, n_anchors, head_type)

    def is_v3(self):
        return getattr(self.bio_head, 'head_type', None) == 'attention_query_v3'

    def bio_predict(self, h, z_bio):
        """
        Unified bio-head call.
        v3: computes soft cluster probs from h, DETACHES them, conditions the
            head on them. The detach is what breaks the
            clusters -> attention -> embedding -> clusters cycle.
        others: plain z_bio -> b_hat, aux is None.
        Returns (b_hat, aux_dict_or_None).
        """
        if self.is_v3():
            c0 = self.cluster_projector(h).detach()   # already softmaxed
            out = self.bio_head(z_bio, c0)
            out['c0'] = c0
            return out['anchor_pred'], out
        return self.bio_head(z_bio), None
    
    # def forward(self, x_i, x_j):
    #     """
    #     Forward pass for training.
    #     Runs both augmented views through the ae and all three heads.

    #     Args:
    #         x_i, x_j : augmented views of same batch, shape (batch, input_dim)

    #     Returns:
    #         z_i, z_j : instance projector outputs for contrastive loss
    #         c_i, c_j : cluster projector outputs for cluster loss
    #         b_hat    : bio-anchor predictions from view i   (NEW)

    #     OLD returned: z_i, z_j, c_i, c_j          (4 values)
    #     NEW returns:  z_i, z_j, c_i, c_j, b_hat   (5 values)
    #     Update your training script to unpack 5 values.
    #     """

    #     # OLD: h_i, _, _ = self.ae(x_i)  ← was discarding z_bio, z_novel
    #     # NEW: unpack all three so we can use z_bio for the bio head
    #     h_i, z_bio_i, z_novel_i = self.ae(x_i)
    #     h_j, z_bio_j, z_novel_j = self.ae(x_j)

    #     # instance contrastive head — unchanged
    #     z_i = normalize(self.instance_projector(h_i), dim=1)
    #     z_j = normalize(self.instance_projector(h_j), dim=1)

    #     # cluster head — unchanged
    #     c_i = self.cluster_projector(h_i)
    #     c_j = self.cluster_projector(h_j)

    #     # bio anchor prediction — NEW
    #     # only run on view i here; training script handles both views
    #     # by calling bio_head on z_bio_i and z_bio_j separately
    #     b_hat = self.bio_head(z_bio_i)

    #     return z_i, z_j, c_i, c_j, b_hat

    def forward(self, x_i, x_j):
        h_i, z_bio_i, z_novel_i = self.ae(x_i)
        h_j, z_bio_j, z_novel_j = self.ae(x_j)

        # bio head first — v3 feeds a residual back into the embedding
        b_hat_i, aux_i = self.bio_predict(h_i, z_bio_i)
        self.last_bio_aux_i = aux_i          # training loop reads this

        if aux_i is not None and aux_i.get('fb') is not None:
            h_i = h_i + aux_i['fb']

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j, b_hat_i

    def _embed(self, x):
        """Shared embedding path. Applies the v3 gate feedback so inference
        matches training. Returns (h_final, z_bio, aux_or_None)."""
        h, z_bio, z_novel = self.ae(x)
        aux = None
        if self.is_v3():
            c0 = self.cluster_projector(h).detach()   # pre-feedback, detached
            aux = self.bio_head(z_bio, c0)
            aux['c0'] = c0
            if aux['fb'] is not None:
                h = h + aux['fb']
        return h, z_bio, aux

    def forward_cluster(self, x):
        """Hard cluster labels. Unchanged signature: (c, h, z_bio)."""
        h, z_bio, aux = self._embed(x)
        c = torch.argmax(self.cluster_projector(h), dim=1)
        return c, h, z_bio

    def forward_cluster_soft(self, x):
        """Soft probs + aux. Returns (c_soft, h, z_bio, aux)."""
        h, z_bio, aux = self._embed(x)
        return self.cluster_projector(h), h, z_bio, aux