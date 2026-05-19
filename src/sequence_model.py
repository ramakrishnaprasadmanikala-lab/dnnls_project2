"""
sequence_model.py
-----------------
GRU-based temporal sequence model.
Processes the sequence of K fused multimodal frames and outputs a
context representation used by the decoders to generate frame K+1.
Week 7 component: Sequence Model.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceModel(nn.Module):
    """
    Temporal GRU that processes K fused multimodal representations.

    The GRU captures temporal dynamics — how the story evolves over frames.
    The final hidden state encodes a summary of the entire input sequence,
    which is passed to the decoders to generate the next (K+1)-th frame.

    Architecture:
        [fused_1, fused_2, ..., fused_K]  → GRU → context
        Input : [B, K, input_dim]
        Output:
            context     : [B, hidden_dim]         final hidden state
            all_states  : [B, K, hidden_dim]      all hidden states (for attention)
            init_hidden : [num_layers, B, hidden_dim]
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2,
                 dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        # If bidirectional, project back to hidden_dim
        if bidirectional:
            self.bidir_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional encoding for the K frames (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, input_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, seq_lengths=None):
        """
        Args:
            x           : Tensor [B, K, input_dim]  fused frame representations
            seq_lengths : Tensor [B] or None         actual sequence lengths
        Returns:
            context     : Tensor [B, hidden_dim]
            all_states  : Tensor [B, K, hidden_dim]
            hidden      : Tensor [num_layers, B, hidden_dim]
        """
        B, K, _ = x.shape

        # Add positional encoding
        x = x + self.pos_embed[:, :K, :]
        x = self.dropout(x)

        if seq_lengths is not None:
            lengths_cpu = seq_lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths_cpu,
                                          batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.gru(packed)
            all_states, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            all_states, hidden = self.gru(x)

        if self.bidirectional:
            # Merge forward + backward outputs
            fwd = all_states[:, :, :self.hidden_dim]
            bwd = all_states[:, :, self.hidden_dim:]
            all_states = self.bidir_proj(torch.cat([fwd, bwd], dim=-1))

            # Merge hidden states
            hidden = hidden.view(self.num_layers, 2, B, self.hidden_dim)
            hidden = (hidden[:, 0] + hidden[:, 1]) / 2  # average directions

        # Context = final hidden state of last layer
        context = hidden[-1]                           # [B, hidden_dim]
        return context, all_states, hidden
