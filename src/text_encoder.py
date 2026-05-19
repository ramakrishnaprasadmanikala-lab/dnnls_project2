"""
text_encoder.py
---------------
GRU-based text encoder that embeds and encodes frame narrative text.
Week 6 component: Text Encoder.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextEncoder(nn.Module):
    """
    Encodes a tokenised text sequence using an Embedding + GRU.

    Architecture:
        Embedding → GRU (packed) → final hidden state projection
        Input : [B, T] token indices + [B] lengths
        Output: [B, hidden_dim]  (last hidden state)
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)          # keep PAD embedding zero

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

        # Project final hidden state to output
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, tokens, lengths):
        """
        Args:
            tokens : Tensor [B, T]
            lengths: Tensor [B]  actual sequence lengths (no padding)
        Returns:
            hidden : Tensor [B, hidden_dim]   — sentence representation
            all_hs : Tensor [B, T, hidden_dim] — all step outputs
        """
        embedded = self.dropout(self.embedding(tokens))   # [B, T, embed_dim]

        # Pack to avoid computing on padding
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(embedded, lengths_cpu,
                                      batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        all_hs, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]

        # Take final layer hidden state
        final_hidden = hidden[-1]                         # [B, hidden_dim]
        out = self.out_proj(final_hidden)
        return out, all_hs

    def encode_sequence(self, tokens_seq, lengths_seq):
        """
        Encode a sequence of K text segments independently.

        Args:
            tokens_seq : Tensor [B, K, T]
            lengths_seq: Tensor [B, K]
        Returns:
            text_feats : Tensor [B, K, hidden_dim]
            all_hs_seq : Tensor [B, K, T, hidden_dim]
        """
        B, K, T = tokens_seq.shape
        tokens_flat = tokens_seq.view(B * K, T)
        lengths_flat = lengths_seq.view(B * K)

        feats_flat, all_hs_flat = self.forward(tokens_flat, lengths_flat)
        T_out = all_hs_flat.shape[1]

        text_feats = feats_flat.view(B, K, self.hidden_dim)
        all_hs_seq = all_hs_flat.view(B, K, T_out, self.hidden_dim)

        return text_feats, all_hs_seq
