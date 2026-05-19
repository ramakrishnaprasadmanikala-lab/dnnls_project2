"""
decoders.py
-----------
Dual decoders for generating the next multimodal element:
  1. TextDecoder  — GRU seq2seq decoder with Bahdanau attention
                    Generates next-frame narrative text word-by-word.
  2. ImageDecoder — MLP that predicts next-frame visual feature vector.
                    Trained with cosine similarity loss against the real
                    ResNet18 features of the target frame.

Weeks 9–10 components: Dual Decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import BahdanauAttention
from .data_loader import SOS_IDX, EOS_IDX, PAD_IDX


# ─── Text Decoder ─────────────────────────────────────────────────────────────
class TextDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention over sequence-model states.
    Generates the next-frame narrative text token by token.

    At each step t:
        context_t = Attention(hidden_{t-1}, encoder_states)
        input_t   = Embedding(y_{t-1})
        hidden_t  = GRU([input_t ; context_t], hidden_{t-1})
        logits_t  = Linear(hidden_t)
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512,
                 encoder_dim=512, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[PAD_IDX].fill_(0)

        self.attention = BahdanauAttention(
            query_dim=hidden_dim,
            key_dim=encoder_dim,
            attn_dim=hidden_dim // 2,
        )

        # GRU input = embedded token + context vector
        self.gru = nn.GRU(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Output projection to vocabulary
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoder_dim + embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward_step(self, prev_token, hidden, encoder_states):
        """
        Single decoder step.

        Args:
            prev_token    : Tensor [B]        previous ground-truth/predicted token
            hidden        : Tensor [1, B, H]  previous GRU hidden state
            encoder_states: Tensor [B, K, H]  temporal encoder outputs

        Returns:
            logits : Tensor [B, vocab_size]
            hidden : Tensor [1, B, H]
        """
        embedded = self.dropout(self.embedding(prev_token))   # [B, embed_dim]
        query = hidden[-1]                                     # [B, H]
        context, _ = self.attention(query, encoder_states)     # [B, H]

        gru_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # [B, 1, embed+H]
        out, hidden = self.gru(gru_input, hidden)             # out: [B, 1, H]
        out = out.squeeze(1)                                   # [B, H]

        logit_input = torch.cat([out, context, embedded], dim=-1)  # [B, H+H+embed]
        logits = self.out_proj(logit_input)                        # [B, vocab]
        return logits, hidden

    def forward(self, encoder_states, init_hidden, target_tokens=None,
                max_len=80, teacher_forcing_ratio=0.5):
        """
        Args:
            encoder_states      : Tensor [B, K, H]
            init_hidden         : Tensor [num_layers, B, H]  from sequence model
            target_tokens       : Tensor [B, T]  (None at inference)
            max_len             : max generation length
            teacher_forcing_ratio: prob of using ground truth token

        Returns:
            all_logits : Tensor [B, T, vocab_size]
        """
        B = encoder_states.size(0)
        device = encoder_states.device

        # Use last-layer hidden as decoder init
        hidden = init_hidden[-1:].contiguous()  # [1, B, H]

        prev_token = torch.full((B,), SOS_IDX, dtype=torch.long, device=device)
        T = target_tokens.size(1) if target_tokens is not None else max_len

        all_logits = []
        for t in range(T):
            logits, hidden = self.forward_step(prev_token, hidden, encoder_states)
            all_logits.append(logits)

            # Teacher forcing: use real token or predicted token
            use_teacher = (target_tokens is not None and
                           torch.rand(1).item() < teacher_forcing_ratio and
                           self.training)
            if use_teacher:
                prev_token = target_tokens[:, t]
            else:
                prev_token = logits.argmax(dim=-1)

        return torch.stack(all_logits, dim=1)   # [B, T, vocab_size]

    @torch.no_grad()
    def greedy_decode(self, encoder_states, init_hidden, max_len=80):
        """Greedy decoding for inference (no teacher forcing)."""
        return self.forward(encoder_states, init_hidden,
                            target_tokens=None, max_len=max_len,
                            teacher_forcing_ratio=0.0)


# ─── Image Feature Decoder ────────────────────────────────────────────────────
class ImageDecoder(nn.Module):
    """
    Predicts the visual feature vector of the next frame.
    Trained with cosine embedding loss: predicted features should be
    directionally close to real ResNet18 features of the target image.

    This is more tractable than pixel-level image generation and still
    captures the visual semantic content of the predicted frame.

    Architecture:
        context [B, context_dim] → MLP → predicted_features [B, output_dim]
    """

    def __init__(self, context_dim=512, output_dim=512, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, output_dim),
        )

    def forward(self, context):
        """
        Args:
            context: Tensor [B, context_dim]
        Returns:
            pred_feat: Tensor [B, output_dim]  (NOT normalised)
        """
        return self.mlp(context)

    def cosine_loss(self, pred, target):
        """
        Cosine embedding loss between predicted and target visual features.
        Drives predicted features to be directionally similar to real features.
        """
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        # Loss = 1 - cosine_similarity (want similarity → 1, loss → 0)
        return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
