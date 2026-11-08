"""Image descriptors: GIST [3] (as used by [1]), a byte histogram, and a thumbnail.

`gist` is the from-scratch part of the reproduction and is written by the
student (see the AI-use policy in README.md). Until it is implemented it raises
NotImplementedError, and src.train skips every model and step that needs it,
recording the skip in results/metrics.json.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a byteplot to size x size with bilinear interpolation (float32 in [0, 1])."""
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

def _frequency_grid(size):
    freqList = np.fft.fftfreq(size)
    x, y = np.meshgrid(freqList, freqList)
    freqRadius = np.hypot(x, y)
    angleGrid = np.arctan2(y, x)
    return freqRadius, angleGrid

def _filter_bank(size, orientation):
    radius, angle = _frequency_grid(size)
    petals = []
    for i, n in enumerate(orientation):
        fc = 0.3/(1.85**i)
        angleWidth = (np.pi / n) / 2.355
        for j in range(n):
            theta = np.pi * j / n
            difference = angle - theta
            angleDiff = np.angle(np.exp(1j * difference))
            distance = radius / fc - 1
            radialBell = np.exp(-(distance**2) / (2 * 0.2**2))
            angleBell = (np.exp(-(angleDiff**2) / (2 * angleWidth**2)))
            combineBells = radialBell * angleBell
            petals.append(combineBells)
    return np.stack(petals)

def _prefilter(img):
    pad = 5
    mult = img * 255
    logged = np.log(mult + 1)
    padded = np.pad(logged, pad, mode="symmetric")
    size = padded.shape[0]
    grid, _ = _frequency_grid(size)
    cycles = grid * size
    s1 = 4 / np.sqrt(np.log(2))
    mask = np.exp(-(cycles)**2 / s1**2)

    def blur(a):
        transformer = np.fft.fft2(a)
        transMask = mask * transformer
        invTrans = np.fft.ifft2(transMask)
        real = np.real(invTrans)
        return real

    white = padded - blur(padded)


    loud = np.sqrt(np.abs(blur((white**2))))
    norm = white / (0.2 + loud)

    slicedNorm = norm[pad:-pad, pad:-pad]
    return slicedNorm

def _block_mean(response, blocks):
    tile = response.shape[0] // blocks
    means = []
    for r in range(blocks):
        for c in range(blocks):
            rows = r * tile
            columns = c * tile
            block = response[rows:rows+tile, columns:columns+tile]
            means.append(block.mean())
    return np.array(means)
    

def gist(img: np.ndarray, gcfg: dict) -> np.ndarray:
    """GIST descriptor of one byteplot, following Oliva & Torralba [3] with the
    parameters Nataraj et al. [1] report.

    Args:
        img:  2-D uint8 byteplot of any shape.
        gcfg: config["features"]["gist"], with
              image_size   side length the byteplot is resized to first
              orientations orientations per scale, finest first; its length
                           is the number of scales (e.g. [8, 4, 4, 4])
              blocks       the filter-response maps are averaged over a
                           blocks x blocks grid

    Returns:
        1-D float32 vector of length sum(orientations) * blocks**2
        (320 for [8, 4, 4, 4] and a 4 x 4 grid). Deterministic: the same image
        must give the same vector.

    Suggested steps (see [3], Sec. 4, and [1], Sec. 4):
        1. `resize(img, image_size)` (provided above).
        2. Prefilter: whiten / normalise local contrast.
        3. Build a Gabor filter bank in the frequency domain, one filter per
           (scale, orientation).
        4. For each filter: FFT the image, multiply, inverse FFT, take |.|.
        5. Average each response map over the blocks x blocks grid.
        6. Concatenate.

    Record every value [1] does not state (prefilter width, Gabor bandwidth,
    padding) as an explicit assumption in the paper.
    """
    
    size = gcfg["image_size"]
    orientations = gcfg["orientations"]
    blocks = gcfg["blocks"]

    resized = resize(img, size)
    prefiltered = _prefilter(resized)

    padding = 32
    pad = np.pad(prefiltered, padding, mode="symmetric")
    paddedSize = pad.shape[0]

    transformer = np.fft.fft2(pad)
    bank = _filter_bank(paddedSize, orientations)

    features = []
    for petal in bank:
        filtered = np.fft.ifft2(petal * transformer)
        magnitude = np.abs(filtered)
        crop = magnitude[padding:-padding, padding:-padding]
        pooled = _block_mean(crop, blocks)
        features.append(pooled)

    return np.concatenate(features).astype(np.float32)


def gist_dim(gcfg: dict) -> int:
    return sum(gcfg["orientations"]) * gcfg["blocks"] ** 2


def byte_histogram(img: np.ndarray) -> np.ndarray:
    """Normalised 256-bin histogram of byte values (the 'is the benchmark easy?' baseline)."""
    h = np.bincount(img.ravel(), minlength=256).astype(np.float32)
    return h / max(h.sum(), 1.0)


def thumbnail(img: np.ndarray, size: int) -> np.ndarray:
    """Flattened size x size resize; a cheap alternative near-duplicate descriptor."""
    return resize(img, size).ravel()
