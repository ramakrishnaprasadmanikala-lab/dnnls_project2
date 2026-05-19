"""
attention.py
------------
Attention mechanisms used across the model.
Week 8 component: Attention Mechanism.

Includes:
  - BahdanauAttention  : additive attention for the decoder
  - ScaledDotAttention : scaled dot-product attention (for cross-modal fusion)
  - MultiHeadAttention : multi-head wrapper (used in cross-modal fusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Bahdanau (Additive) Attention ────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    """
    Standard additive attention used in the seq2seq text decoder.
    Allows decoder to focus on relevant encoder steps at each generation step.

    score(q, k) = v^T · tanh(W_q·q + W_k·k)
    """

    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.W_query = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_key = nn.Linear(key_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query : Tensor [B, query_dim]
            keys  : Tensor [B, L, key_dim]
            mask  : BoolTensor [B, L] — True where positions are padding
        Returns:
            context : Tensor [B, key_dim]
            weights : Tensor [B, L]
        """
        # Expand query: [B, 1, attn_dim]
        q_proj = self.W_query(query).unsqueeze(1)
        k_proj = self.W_key(keys)                       # [B, L, attn_dim]
        energy = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)  # [B, L]

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)             # [B, L]
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # [B, key_dim]
        return context, weights


# ─── Scaled Dot-Product Attention ─────────────────────────────────────────────
class ScaledDotAttention(nn.Module):
    """
    Scaled dot-product attention.
    score(Q, K) = softmax(Q·K^T / sqrt(d_k)) · V
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query : Tensor [B, Lq, d_k]
            key   : Tensor [B, Lk, d_k]
            value : Tensor [B, Lk, d_v]
            mask  : BoolTensor [B, Lq, Lk]
        Returns:
            out     : Tensor [B, Lq, d_v]
            weights : Tensor [B, Lq, Lk]
        """
        d_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_k)  # [B, Lq, Lk]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.bmm(weights, value)
        return out, weights


# ─── Multi-Head Attention ──────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention over (query, key, value) triples.
    Used in the cross-modal fusion module.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        """[B, L, d_model] → [B*H, L, d_k]"""
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.d_k)
        x = x.transpose(1, 2).contiguous()
        return x.view(B * self.num_heads, L, self.d_k)

    def merge_heads(self, x, B):
        """[B*H, L, d_k] → [B, L, d_model]"""
        L = x.size(1)
        x = x.view(B, self.num_heads, L, self.d_k)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensor [B, L, d_model]
        Returns:
            out: Tensor [B, Lq, d_model]
        """
        B = query.size(0)
        residual = query

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        out, _ = self.attn(Q, K, V, mask=mask)
        out = self.merge_heads(out, B)
        out = self.W_o(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out
