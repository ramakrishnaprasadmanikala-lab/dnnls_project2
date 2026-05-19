"""
utils.py
--------
Shared utility functions: seeding, metrics tracking, parameter counting, etc.
"""

import os
import random
import time
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format seconds into a readable string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def plot_training_curves(history, save_path="results/training_curves.png"):
    """Save a plot of train/val loss over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "r-s", label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss",  fontsize=13)
    ax.set_title("Training Curves — Multimodal Story Continuation", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Utils] Training curves saved to {save_path}")


def plot_ablation(results, save_path="results/ablation_bar.png"):
    """Bar chart comparing cross-modal attention vs baseline."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = list(results.keys())
    metrics = ["bleu_1", "bleu_2", "cosine_sim"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(labels):
        vals = [results[label][m] for m in metrics]
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(["BLEU-1", "BLEU-2", "Cosine Similarity"], fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation: Cross-Modal Attention vs Concatenation", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Utils] Ablation chart saved to {save_path}")
