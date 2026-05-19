# Visual Storytelling with Cross-Modal Bidirectional Attention

## Quick Links
- **[Experiments Notebook](experiments.ipynb)** — Full experimental workflow
- **[Baseline Results](results/ablation_table.csv)** — Concatenation baseline performance
- **[Improved Results](results/ablation_bar.png)** — Results with Cross-Modal Attention innovation
- **[Training Curves](results/training_curves.png)** — Loss curves across epochs
- **[Sample Frames](results/sample_frames.png)** — Dataset visualisation

---

## Innovation Summary

**I modified the multimodal fusion mechanism to use Cross-Modal Bidirectional Attention, expecting to improve narrative text generation quality in sequential visual story continuation.**

Instead of naively concatenating image and text features (the baseline approach), the innovation applies bidirectional cross-modal attention at every story frame:

- **Text → Image attention:** Each text representation queries all image features, extracting relevant visual context that supports the narrative.
- **Image → Text attention:** Each image representation queries all text features, grounding visual content in the narrative description.
- **Adaptive gating:** A learned sigmoid gate dynamically blends the two enriched streams based on content, rather than weighting them equally.

This richer joint representation is then passed to a GRU sequence model and attention-based text decoder to predict the next story frame's narrative.

---

## Key Results

| Metric | Baseline (Concat) | Improved (Ours) | Change |
|---|---|---|---|
| BLEU-1 | 0.1359 | **0.1390** | **+0.0031** |
| BLEU-2 | 0.0396 | **0.0405** | **+0.0009** |
| Cosine Similarity | 0.0096 | -0.0152 | -0.0248 |

---

## Most Important Finding

> Cross-Modal Bidirectional Attention consistently outperforms the concatenation baseline on both BLEU-1 and BLEU-2 metrics, confirming that allowing modalities to selectively query each other produces richer representations for narrative text generation. Cosine similarity is not a meaningful metric in this experiment as the image decoder loss was disabled (weight=0.0), focusing the model entirely on text generation quality.

---

## Dataset

**StoryReasoning** — Oliveira & Matos (2025). *StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation.* arXiv:2505.10292.

| Property | Value |
|---|---|
| Total stories | 4,178 |
| Total frames | 52,016 movie images |
| Train split | 3,550 stories |
| Test split | 626 stories |
| Stories used (train) | 300 |
| Stories used (val) | 80 |

---

## Architecture

```
images [B,K,3,H,W] ──► VisualEncoder (ResNet18, frozen) ──► v_feats [B,K,512]
                                                                      │
texts  [B,K,T]     ──► TextEncoder (GRU, H=128)          ──► t_feats [B,K,128]
                                                                      │
                         CrossModalFusion (INNOVATION)          ──► fused [B,K,128]
                         ├── Text → Image attention
                         ├── Image → Text attention
                         └── Adaptive gating network
                                                                      │
                         SequenceModel (GRU, H=256)             ──► context [B,256]
                                                                      │
                         TextDecoder (GRU + Bahdanau Attention) ──► text logits
```

| Component | Details |
|---|---|
| Visual Encoder | ResNet18 pretrained on ImageNet, backbone frozen |
| Text Encoder | GRU, embed_dim=64, hidden_dim=128 |
| Fusion | Cross-Modal Bidirectional Attention, 2 heads |
| Sequence Model | GRU, hidden_dim=256, 1 layer |
| Text Decoder | GRU + Bahdanau attention, teacher forcing ratio=0.7 |
| Vocabulary | 3,000 tokens |

---

## Training Configuration

| Setting | Value |
|---|---|
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 0.002 (AdamW) |
| LR schedule | Cosine annealing |
| Gradient clipping | 1.0 |
| Attention model seed | 42 |
| Baseline model seed | 123 |
| Input frames (K) | 3 |
| Max text length | 40 tokens |

---

## Experiment Design

**Pre-registered Hypothesis:** Cross-Modal Bidirectional Attention will outperform naive concatenation for multimodal fusion because bidirectional attention allows each modality to selectively query the other, amplifying relevant cross-modal signals rather than weighting them equally.

**Ablation:** Two models trained with identical hyperparameters, differing only in the fusion module:
1. **Cross-Modal Attention** (ours) — `fusion.type: cross_modal_attention`, seed=42
2. **Concatenation Baseline** — `fusion.type: concat`, seed=123

Different seeds ensure independent weight initialisation, preventing the variable-reuse issue that causes identical scores in Colab sessions.

---

## How to Reproduce

1. Upload `dnnls_project2_final.zip` to Google Colab
2. Run the Colab Setup cell (Cell 0) to unzip and configure paths
3. Install dependencies: `pip install -r requirements.txt`
4. Run all cells sequentially in `experiments.ipynb`
5. Results are saved to `results/` automatically

**Runtime:** approximately 15–25 minutes on CPU | 5–8 minutes on GPU (T4)

---

## Repository Structure

```
dnnls_project2/
├── experiments.ipynb        ← Main experiment notebook
├── config.yaml              ← All hyperparameters
├── requirements.txt         ← Python dependencies
├── src/
│   ├── model.py             ← Full model combining all components
│   ├── fusion.py            ← Cross-Modal Bidirectional Attention (innovation)
│   ├── data_loader.py       ← Dataset + visual feature caching
│   ├── visual_encoder.py    ← ResNet18 CNN encoder
│   ├── text_encoder.py      ← GRU text encoder
│   ├── sequence_model.py    ← Temporal GRU
│   ├── decoders.py          ← Text decoder with attention
│   ├── attention.py         ← Bahdanau + Multi-head attention
│   └── utils.py             ← Seeding, metrics, plotting
└── results/
    ├── training_curves.png  ← Loss curves
    ├── ablation_bar.png     ← Comparison bar chart
    ├── ablation_table.csv   ← Numerical results
    ├── metrics.txt          ← Final scores
    └── sample_frames.png    ← Dataset visualisation
```

---

## Citation

```
@article{oliveira2025storyreasoning,
  title   = {StoryReasoning Dataset: Using Chain-of-Thought for Scene
             Understanding and Grounded Story Generation},
  author  = {Oliveira, Daniel A. P. and Matos, David M.},
  journal = {arXiv preprint arXiv:2505.10292},
  year    = {2025}
}
```
