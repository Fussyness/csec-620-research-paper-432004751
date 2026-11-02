"""Classifiers. Each one turns a byteplot into a fixed representation
(`represent`, cached per dataset by src.train), then fits and predicts on
stacks of those representations.

  gist_knn      GIST [3] + Euclidean k-NN — the reproduction of [1]
  byte_hist_lr  256-bin byte histogram + logistic regression — an "easy benchmark" probe
  cnn           ImageNet-pretrained ResNet [9], fine-tuned — the stronger baseline ([4], [7])
"""
from __future__ import annotations
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import features

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _full_proba(classes, proba, n_classes) -> np.ndarray:
    """Expand predict_proba columns to all n_classes (a fold may miss a family)."""
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, classes] = proba
    return out


class GistKNN:
    name = "gist_knn"

    def __init__(self, cfg):
        self.gcfg, self.k = cfg["features"]["gist"], cfg["models"]["gist_knn"]["k"]

    def rep_key(self):
        return "gist_" + json.dumps(self.gcfg, sort_keys=True)

    def represent(self, img):
        return features.gist(img, self.gcfg)

    def fit(self, R, y, n_classes, seed):
        # No feature scaling: [1] uses plain Euclidean distance on GIST vectors.
        self.n_classes = n_classes
        self.clf = KNeighborsClassifier(n_neighbors=self.k, metric="euclidean").fit(R, y)

    def predict_proba(self, R):
        return _full_proba(self.clf.classes_, self.clf.predict_proba(R), self.n_classes)


class ByteHistLR:
    name = "byte_hist_lr"

    def __init__(self, cfg):
        self.mcfg = cfg["models"]["byte_hist_lr"]

    def rep_key(self):
        return "hist256"

    def represent(self, img):
        return features.byte_histogram(img)

    def fit(self, R, y, n_classes, seed):
        self.n_classes = n_classes
        self.scaler = StandardScaler().fit(R)          # train fold only
        self.clf = LogisticRegression(C=self.mcfg["C"], max_iter=self.mcfg["max_iter"],
                                      random_state=seed).fit(self.scaler.transform(R), y)

    def predict_proba(self, R):
        p = self.clf.predict_proba(self.scaler.transform(R))
        return _full_proba(self.clf.classes_, p, self.n_classes)


class CNNClassifier:
    """ResNet on grayscale byteplots replicated to 3 channels. Augmentation is
    off (augmentation changes results [7]); the loss is class-weighted for imbalance."""
    name = "cnn"

    def __init__(self, cfg):
        self.c = cfg["models"]["cnn"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def rep_key(self):
        return f"cnn{self.c['input_size']}"

    def _resize(self, x):
        s = self.c["input_size"]
        return F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)

    def represent(self, img):
        x = torch.from_numpy(img).float()[None, None] / 255.0
        return (self._resize(x)[0, 0].clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    def _build(self, n_classes):
        import torchvision
        ctor = getattr(torchvision.models, self.c["backbone"])
        weights = "DEFAULT" if self.c["pretrained"] else None
        net = ctor(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        if self.c["freeze_backbone"]:
            for n, p in net.named_parameters():
                p.requires_grad = n.startswith("fc.")
        return net.to(self.device)

    def _normalise(self, x):
        """x: (B, 1, S, S) in [0, 1] -> ImageNet-normalised 3-channel batch."""
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return (x.expand(-1, 3, -1, -1) - mean) / std

    def _batch(self, R, idx):
        x = torch.from_numpy(R[idx]).to(self.device, non_blocking=True)
        return self._normalise(x.float()[:, None] / 255.0)

    def logits_from_raw(self, x):
        """Differentiable path for white-box attacks: x is (B, 1, H, W) in [0, 1]
        at the byteplot's native size."""
        return self.net(self._normalise(self._resize(x)))

    def fit(self, R, y, n_classes, seed):
        torch.manual_seed(seed)
        self.net = self._build(n_classes)
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(w, device=self.device) if self.c["class_weighted"] else None)
        opt = torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],
                                lr=self.c["lr"], weight_decay=self.c["weight_decay"])
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        gen = np.random.default_rng(seed)
        yt = torch.from_numpy(y).to(self.device)
        bs = self.c["batch_size"]
        self.net.train()
        for _ in range(self.c["epochs"]):
            perm = gen.permutation(len(y))
            for i in range(0, len(y), bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(self.device, enabled=use_amp):
                    loss = loss_fn(self.net(self._batch(R, idx)), yt[idx])
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
        self.net.eval()

    @torch.no_grad()
    def predict_proba(self, R):
        out, bs = [], self.c["batch_size"] * 2
        for i in range(0, len(R), bs):
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                logits = self.net(self._batch(R, np.arange(i, min(i + bs, len(R)))))
            out.append(torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(out)


MODELS = {m.name: m for m in (GistKNN, ByteHistLR, CNNClassifier)}


def build(name: str, cfg: dict):
    return MODELS[name](cfg)
