"""E4: functionality-preserving append perturbations (threat model of [11], [12]).

Written by the student (see the AI-use policy in README.md). Until implemented,
these raise NotImplementedError and src.train skips E4.

Common conventions for every function here:
  * `img` is a clean byteplot (2-D uint8). Recover its bytes with
    `src.data.image_bytes(img)`.
  * `fraction` is the append budget relative to the original byte count:
    append round(fraction * img.size) bytes (0.01 = 1%). A budget of 0 must
    return the image unchanged.
  * Append the new bytes *after* the original ones, then re-render with
    `src.data.byteplot(raw, width_table)`. This can move the file into a
    different width bin of [1]'s table, which is the interaction the paper
    analyses, so do not keep the old width by force.
  * Return a new 2-D uint8 array; never modify `img` in place.
"""
from __future__ import annotations
import numpy as np
from src.data import byteplot, image_bytes
import torch
import torch.nn.functional as F


def append_bytes(img: np.ndarray, fraction: float, kind: str, rng: np.random.Generator,
                 width_table, donor: np.ndarray | None = None) -> np.ndarray:
    """Black-box append attack.

    kind:
      "random"        uniform random bytes from `rng`
      "zero"          zero padding
      "other_family"  bytes taken from `donor`, a byteplot of a sample from a
                      different family (the caller picks it; take a contiguous
                      run starting at a random offset drawn from `rng`)
    """
    if fraction == 0:
        return img.copy()

    raw = image_bytes(img)
    n = img.size
    k = int(round(fraction * n))

    if kind == "zero":
        extra = bytes(k)
    elif kind == "random":
        extra = rng.integers(0, 256, size=k, dtype=np.uint8).tobytes()
    elif kind == "other_family":
        d = image_bytes(donor)
        copies = -(-k // donor.size)
        d = d * copies
        start = rng.integers(0, len(d) - k + 1)
        extra = d[start:start+k]
    else:
        raise ValueError(f"unknown append kind: {kind}")

    return byteplot(raw+extra, width_table)


def fgsm_append(cnn, img: np.ndarray, label: int, fraction: float, eps: float,
                width_table) -> np.ndarray:
    """White-box FGSM restricted to the appended rows (CNN only).

    Steps: append `fraction` of extra bytes (e.g. zeros) and re-render; turn the
    image into a float tensor in [0, 1] of shape (1, 1, H, W) with
    requires_grad; compute the loss of `cnn.logits_from_raw(x)` against `label`;
    step by eps * sign(grad) *only on the appended pixels* (mask everything
    else); clip to [0, 1]; quantise back to uint8 bytes.

    `cnn` is a src.model.CNNClassifier; `logits_from_raw` applies the same
    resize and normalisation as training, and gradients flow through it.
    """
    if fraction == 0:
        return img.copy()
    padded = append_bytes(img, fraction, "zero", None, width_table)

    n = img.size
    k = int(round(fraction * n))
    falseRow = np.zeros(padded.size, dtype=bool)
    falseRow[n:n+k] = True
    fold = falseRow.reshape(padded.shape)

    x = (torch.from_numpy(padded).float() / 255)[None,None].to(cnn.device).requires_grad_(True)
    mask = torch.from_numpy(fold)[None,None].to(cnn.device)

    logits = cnn.logits_from_raw(x)
    loss = F.cross_entropy(logits, torch.tensor([label], device=cnn.device))
    grad = torch.autograd.grad(loss, x)[0]

    adv = (x.detach() + eps * grad.sign() * mask).clamp(0, 1)

    return (adv[0, 0] * 255).round().to(torch.uint8).cpu().numpy()