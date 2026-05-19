"""
model.py
--------
Multimodal sequence model for visual story continuation.
Uses pre-cached visual features — no CNN in the training forward pass.

Architecture:
    input_feats [B,K,512] ──────────────────────────────────► v_feats
                                                                   │
    input_texts [B,K,T]  ──► TextEncoder (GRU)          ──► t_feats
                                                                   │
                             CrossModalFusion (INNOVATION)  ──► fused [B,K,128]
                             (Bidirectional cross-attention
                              + adaptive gating network)
                                                                   │
                             SequenceModel (GRU)            ──► context [B,256]
                                                                   │
                             TextDecoder (GRU + Attention)  ──► text logits
"""

import torch
import torch.nn as nn
import yaml

from .text_encoder import TextEncoder
from .fusion import CrossModalFusion, BaselineConcatFusion
from .sequence_model import SequenceModel
from .decoders import TextDecoder


class MultimodalModel(nn.Module):
    """
    End-to-end multimodal sequence model for visual story continuation.
    Given K (image features, text) pairs, generates the (K+1)-th text.
    """

    def __init__(self, cfg, vocab_size):
        super().__init__()
        c_txt = cfg["model"]["text_encoder"]
        c_fus = cfg["model"]["fusion"]
        c_seq = cfg["model"]["sequence_model"]
        c_dec = cfg["model"]["text_decoder"]

        vis_dim = cfg["model"]["visual_encoder"]["feature_dim"]
        txt_dim = c_txt["hidden_dim"]
        fus_dim = c_fus["hidden_dim"]
        seq_dim = c_seq["hidden_dim"]

        # ── Text Encoder ──────────────────────────────────────────────────────
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=c_txt["embed_dim"],
            hidden_dim=txt_dim,
            num_layers=c_txt["num_layers"],
            dropout=c_txt["dropout"],
        )

        # ── Fusion: Cross-Modal Bidirectional Attention (INNOVATION) ──────────
        if c_fus["type"] == "cross_modal_attention":
            self.fusion = CrossModalFusion(
                visual_dim=vis_dim,
                text_dim=txt_dim,
                hidden_dim=fus_dim,
                num_heads=c_fus["num_heads"],
                dropout=c_fus["dropout"],
            )
        else:
            self.fusion = BaselineConcatFusion(
                visual_dim=vis_dim,
                text_dim=txt_dim,
                hidden_dim=fus_dim,
            )

        # ── Sequence Model ────────────────────────────────────────────────────
        self.sequence_model = SequenceModel(
            input_dim=fus_dim,
            hidden_dim=seq_dim,
            num_layers=c_seq["num_layers"],
            dropout=c_seq["dropout"],
        )

        # ── Text Decoder ──────────────────────────────────────────────────────
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            embed_dim=c_txt["embed_dim"],
            hidden_dim=c_dec["hidden_dim"],
            encoder_dim=seq_dim,
            dropout=c_txt["dropout"],
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_feats, input_texts, input_lens,
                target_texts=None, teacher_forcing_ratio=0.7):
        """
        Args:
            input_feats  : Tensor [B, K, 512]   pre-cached visual features
            input_texts  : Tensor [B, K, T]
            input_lens   : Tensor [B, K]
            target_texts : Tensor [B, T'] or None
        Returns:
            text_logits  : Tensor [B, T', vocab_size]
        """
        t_feats, _ = self.text_encoder.encode_sequence(input_texts, input_lens)
        fused = self.fusion(input_feats, t_feats)
        context, all_states, hidden = self.sequence_model(fused)
        text_logits = self.text_decoder(
            encoder_states=all_states,
            init_hidden=hidden,
            target_tokens=target_texts,
            teacher_forcing_ratio=teacher_forcing_ratio if self.training else 0.0,
        )
        return text_logits

    @torch.no_grad()
    def generate(self, input_feats, input_texts, input_lens, max_len=50):
        self.eval()
        t_feats, _ = self.text_encoder.encode_sequence(input_texts, input_lens)
        fused = self.fusion(input_feats, t_feats)
        context, all_states, hidden = self.sequence_model(fused)
        logits = self.text_decoder.greedy_decode(all_states, hidden, max_len=max_len)
        return logits.argmax(dim=-1)


def build_model(cfg_path, vocab_size):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return MultimodalModel(cfg, vocab_size), cfg
