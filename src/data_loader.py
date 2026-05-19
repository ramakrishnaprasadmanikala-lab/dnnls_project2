"""
data_loader.py
--------------
Loads and preprocesses the StoryReasoning dataset from Hugging Face.
Each story contains sequential movie frames + per-frame narrative text.
Task: given K frames, predict frame K+1 (multimodal continuation).

Visual Feature Caching
-----------------------
All visual features are extracted once at startup and cached to disk.
Subsequent runs load cached tensors — no CNN cost per training batch.
"""

import os
import re
import pickle
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# ─── Tokens ───────────────────────────────────────────────────────────────────
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.word_freq = Counter()

    def build_from_texts(self, texts):
        for text in texts:
            for token in self._tok(text):
                self.word_freq[token] += 1
        for word, _ in self.word_freq.most_common(self.max_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"[Vocab] {len(self.word2idx)} tokens")

    def encode(self, text, max_len=None):
        ids = [self.word2idx.get(t, UNK_IDX) for t in self._tok(text)]
        return ids[:max_len] if max_len else ids

    def decode(self, ids):
        words = []
        for i in ids:
            if i in (EOS_IDX, PAD_IDX):
                break
            if i != SOS_IDX:
                words.append(self.idx2word.get(i, UNK_TOKEN))
        return " ".join(words)

    def _tok(self, text):
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── Text Parsing ─────────────────────────────────────────────────────────────
def extract_frame_texts(story):
    """Extract per-frame narrative segments from story string."""
    matches = re.findall(r"<gdi image\d+>(.*?)</gdi>", story, re.DOTALL | re.IGNORECASE)
    out = []
    for m in matches:
        text = re.sub(r"<[^>]+>", " ", m)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


# ─── Image Transform ──────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pil_to_tensor(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    return IMG_TRANSFORM(img.convert("RGB"))


# ─── Visual Feature Extraction & Caching ──────────────────────────────────────
def extract_and_cache_features(hf_subset, visual_encoder, cache_path, device):
    """
    Pre-extract all visual features from a dataset split and cache to disk.
    On subsequent runs, features are loaded from cache instantly.
    """
    if os.path.exists(cache_path):
        print(f"[Cache] Loading features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"[Cache] Extracting visual features...")
    visual_encoder.eval()
    all_story_feats = []

    with torch.no_grad():
        for item in tqdm(hf_subset, desc="Extracting features"):
            frame_feats = []
            for img in item["images"]:
                t = pil_to_tensor(img).unsqueeze(0).to(device)
                feat = visual_encoder.backbone(t).flatten(1)
                feat = visual_encoder.projector(feat)
                frame_feats.append(feat.squeeze(0).cpu())
            all_story_feats.append(torch.stack(frame_feats))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(all_story_feats, cache_path)
    print(f"[Cache] Saved to {cache_path}")
    return all_story_feats


# ─── Dataset ──────────────────────────────────────────────────────────────────
class StoryDataset(Dataset):
    """
    Dataset using pre-cached visual features.
    Returns pre-computed tensors — no CNN cost per batch.
    """

    def __init__(self, hf_subset, cached_feats, vocab, split="train",
                 max_seq_len=3, max_text_len=40):
        self.data = hf_subset
        self.cached_feats = cached_feats
        self.vocab = vocab
        self.K = max_seq_len
        self.T = max_text_len
        self.split = split
        self._valid = self._filter()

    def _filter(self):
        valid = []
        for i, item in enumerate(self.data):
            texts = extract_frame_texts(item["story"])
            n_frames = len(self.cached_feats[i])
            if n_frames >= self.K + 1 and len(texts) >= self.K + 1:
                valid.append(i)
        print(f"[Dataset/{self.split}] {len(valid)}/{len(self.data)} stories usable")
        return valid

    def __len__(self):
        return len(self._valid)

    def __getitem__(self, idx):
        real_idx = self._valid[idx]
        item = self.data[real_idx]
        feats = self.cached_feats[real_idx]
        texts = extract_frame_texts(item["story"])

        max_start = len(feats) - self.K - 1
        start = (np.random.randint(0, max(1, max_start + 1))
                 if self.split == "train" else 0)

        input_feats = feats[start: start + self.K]
        target_feat = feats[start + self.K]

        in_tok, in_len = [], []
        for txt in texts[start: start + self.K]:
            enc = self.vocab.encode(txt, max_len=self.T)
            in_len.append(len(enc))
            enc = enc + [PAD_IDX] * (self.T - len(enc))
            in_tok.append(torch.tensor(enc, dtype=torch.long))
        input_texts = torch.stack(in_tok)
        input_lens  = torch.tensor(in_len, dtype=torch.long)

        tgt_enc = ([SOS_IDX]
                   + self.vocab.encode(texts[start + self.K], max_len=self.T - 2)
                   + [EOS_IDX])
        tgt_len = len(tgt_enc)
        tgt_enc = tgt_enc + [PAD_IDX] * (self.T - len(tgt_enc))
        target_text = torch.tensor(tgt_enc, dtype=torch.long)

        return {
            "input_feats":  input_feats,
            "input_texts":  input_texts,
            "input_lens":   input_lens,
            "target_feat":  target_feat,
            "target_text":  target_text,
            "target_len":   tgt_len,
        }


# ─── Builder ──────────────────────────────────────────────────────────────────
def build_loaders(cfg, train_hf, test_hf, visual_encoder, device):
    """Build vocabulary, cache features, return DataLoaders."""
    c         = cfg["training"]
    K, T      = c["max_seq_len"], c["max_text_len"]
    cache_dir = cfg["paths"]["feature_cache_dir"]
    ckpt_dir  = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    n_tr = min(c["train_subset"], len(train_hf))
    n_va = min(c["val_subset"],   len(test_hf))
    train_sub = train_hf.select(range(n_tr))
    val_sub   = test_hf.select(range(n_va))

    vocab_path = os.path.join(ckpt_dir, "vocab.pkl")
    if os.path.exists(vocab_path):
        print("[Vocab] Loading cached vocabulary")
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary(max_size=cfg["model"]["text_encoder"]["vocab_size"])
        all_texts = []
        for item in train_sub:
            all_texts.extend(extract_frame_texts(item["story"]))
        vocab.build_from_texts(all_texts)
        vocab.save(vocab_path)

    tr_cache = os.path.join(cache_dir, "train_feats.pt")
    va_cache = os.path.join(cache_dir, "val_feats.pt")
    train_feats = extract_and_cache_features(train_sub, visual_encoder, tr_cache, device)
    val_feats   = extract_and_cache_features(val_sub,   visual_encoder, va_cache, device)

    train_ds = StoryDataset(train_sub, train_feats, vocab, "train", K, T)
    val_ds   = StoryDataset(val_sub,   val_feats,   vocab, "val",   K, T)

    train_loader = DataLoader(train_ds, batch_size=c["batch_size"],
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"],
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, vocab
