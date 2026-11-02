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
    raise NotImplementedError("append_bytes not implemented yet (src/attack.py)")


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
    raise NotImplementedError("fgsm_append not implemented yet (src/attack.py)")
