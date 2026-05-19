"""
fusion.py
---------
INNOVATION: Cross-Modal Bidirectional Attention Fusion
-------------------------------------------------------
Instead of naive concatenation of image and text features, we apply
bidirectional cross-modal attention:

  1. Text → Image  attention: each text vector attends over image features.
                               "What visual context matches this description?"
  2. Image → Text  attention: each image vector attends over text features.
                               "What description matches what I see?"

Both enriched representations are then combined via a gating network,
producing a richer joint embedding compared to simple concatenation.

Week 3 component: Multimodal Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class CrossModalFusion(nn.Module):
    """
    Cross-Modal Bidirectional Attention Fusion.

    For each frame step, given:
        v_feat: visual feature  [B, visual_dim]
        t_feat: text feature    [B, text_dim]

    Returns:
        fused: Tensor [B, hidden_dim]  — joint cross-modal representation
    """

    def __init__(self, visual_dim=512, text_dim=256, hidden_dim=256,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both modalities to same hidden_dim before attention
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-attention: Text queries Image (T→V)
        self.text_to_img_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Cross-attention: Image queries Text (V→T)
        self.img_to_text_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Gating network — decides how much of each enriched stream to use
        # Gate is a learned soft weighting between the two streams
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: Tensor [B, visual_dim]   or [B, K, visual_dim]
            text_feat  : Tensor [B, text_dim]     or [B, K, text_dim]
        Returns:
            fused: Tensor [B, hidden_dim]          or [B, K, hidden_dim]
        """
        squeeze = visual_feat.dim() == 2
        if squeeze:
            visual_feat = visual_feat.unsqueeze(1)   # [B, 1, visual_dim]
            text_feat = text_feat.unsqueeze(1)        # [B, 1, text_dim]

        # ── Project to common dim ────────────────────────────────────────────
        v = self.v_proj(visual_feat)    # [B, K, hidden_dim]
        t = self.t_proj(text_feat)      # [B, K, hidden_dim]

        # ── Bidirectional cross-modal attention ──────────────────────────────
        # Text attends to image  (enriched text: what image supports the text)
        t_enriched = self.text_to_img_attn(query=t, key=v, value=v)  # [B, K, H]

        # Image attends to text  (enriched image: what text describes the image)
        v_enriched = self.img_to_text_attn(query=v, key=t, value=t)  # [B, K, H]

        # ── Gating ───────────────────────────────────────────────────────────
        gate_input = torch.cat([t_enriched, v_enriched], dim=-1)   # [B, K, 2H]
        gate = self.gate(gate_input)                                # [B, K, H]

        # Soft blend: gate controls contribution of each enriched stream
        blended = gate * t_enriched + (1 - gate) * v_enriched      # [B, K, H]

        # ── Final fusion with original enriched streams ───────────────────────
        fused_input = torch.cat([blended, t_enriched + v_enriched], dim=-1)  # [B, K, 2H]
        fused = self.fusion_proj(fused_input)                                 # [B, K, H]

        if squeeze:
            fused = fused.squeeze(1)    # [B, hidden_dim]

        return fused


class BaselineConcatFusion(nn.Module):
    """
    Baseline fusion via concatenation + MLP.
    Used for ablation study to measure impact of cross-modal attention.
    """

    def __init__(self, visual_dim=512, text_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_feat, text_feat):
        squeeze = visual_feat.dim() == 2
        if squeeze:
            visual_feat = visual_feat.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
        combined = torch.cat([visual_feat, text_feat], dim=-1)
        out = self.mlp(combined)
        if squeeze:
            out = out.squeeze(1)
        return out
