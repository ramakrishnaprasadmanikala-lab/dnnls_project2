"""
visual_encoder.py
-----------------
CNN-based visual encoder using a pretrained ResNet18 backbone.
Produces a fixed-size feature vector per image frame.
Week 4 component: Visual Encoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    """
    Encodes an image into a dense feature vector using ResNet18.

    Architecture:
        ResNet18 (pretrained, backbone frozen) → GlobalAvgPool → Linear → ReLU → Dropout
        Input : [B, 3, 224, 224]
        Output: [B, feature_dim]
    """

    def __init__(self, feature_dim=512, pretrained=True, freeze_backbone=True,
                 dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Load pretrained ResNet18 ─────────────────────────────────────────
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Remove final classification head — keep up to AdaptiveAvgPool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 512, 1, 1]
        backbone_out_dim = 512  # ResNet18 final channels

        # Optionally freeze backbone to speed up training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Projection head ──────────────────────────────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, H, W]
        Returns:
            features: Tensor [B, feature_dim]
        """
        with torch.set_grad_enabled(not self._backbone_frozen()):
            feats = self.backbone(x)                   # [B, 512, 1, 1]
        feats = feats.flatten(1)                       # [B, 512]
        out = self.projector(feats)                    # [B, feature_dim]
        return out

    def encode_sequence(self, images):
        """
        Encode a sequence of images independently.

        Args:
            images: Tensor [B, K, 3, H, W]
        Returns:
            feats: Tensor [B, K, feature_dim]
        """
        B, K, C, H, W = images.shape
        imgs_flat = images.view(B * K, C, H, W)
        feats_flat = self.forward(imgs_flat)           # [B*K, feature_dim]
        return feats_flat.view(B, K, self.feature_dim)  # [B, K, feature_dim]

    def _backbone_frozen(self):
        return not next(self.backbone.parameters()).requires_grad
