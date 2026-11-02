"""Datasets, byteplot rendering, deduplication, and cross-validation folds.

Every dataset is exposed as an `ImageSet`: parallel lists of names and integer
labels plus `load(i)`, which returns the i-th byteplot as a 2-D uint8 array
(one byte per pixel, rows of `width` bytes, as in Nataraj et al. [1]).

Sources
  malimg    pre-rendered PNGs, one folder per family (data/README.md)
  big2015   Microsoft BIG 2015 `.bytes` hex dumps, rendered once with [1]'s
            width table and cached as PNGs
  synthetic small fake byteplots with injected duplicates, for CI and tests
"""
from __future__ import annotations
import csv, hashlib, os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- byteplots
def width_for_size(n_bytes: int, width_table) -> int:
    """Byteplot width for a file of `n_bytes`, from [1]'s size→width table.

    `width_table` rows are [upper_bound_kb, width]; bounds are exclusive and a
    null bound marks the open-ended last row. 1 KB = 1024 bytes (assumption:
    [1] does not say which convention it uses).
    """
    kb = n_bytes / 1024
    for bound, width in width_table:
        if bound is None or kb < bound:
            return int(width)
    raise ValueError("width_table needs a final row with a null bound")


def byteplot(raw: bytes, width_table) -> np.ndarray:
    """Render raw bytes as a grayscale image of shape (rows, width).

    The last partial row is zero-padded (assumption; [1] does not specify).
    """
    buf = np.frombuffer(raw, dtype=np.uint8)
    width = width_for_size(len(buf), width_table)
    rows = -(-len(buf) // width)
    out = np.zeros(rows * width, dtype=np.uint8)
    out[: len(buf)] = buf
    return out.reshape(rows, width)


def image_bytes(img: np.ndarray) -> bytes:
    """Inverse of `byteplot` up to padding: the image's pixels in file order."""
    return np.ascontiguousarray(img, dtype=np.uint8).tobytes()


def pixel_sha256(img: np.ndarray) -> str:
    """Exact-duplicate key: hash of the shape and the pixel array."""
    h = hashlib.sha256(str(img.shape).encode())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- datasets
@dataclass
class ImageSet:
    name: str
    families: list[str]                 # label index -> family name
    names: list[str]                    # sample ids
    labels: np.ndarray                  # int64, shape (n,)
    _loader: Callable[[int], np.ndarray] = field(repr=False)

    def __len__(self):
        return len(self.names)

    def load(self, i: int) -> np.ndarray:
        return self._loader(i)


def _load_png(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def load_malimg(root: str) -> ImageSet:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Malimg not found at {root}. Download it as described in data/README.md "
            "or set data.malimg_dir in the config.")
    families = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for li, fam in enumerate(families):
        for p in sorted((root / fam).glob("*.png")):
            paths.append(p); labels.append(li)
    return ImageSet("malimg", families, [f"{p.parent.name}/{p.name}" for p in paths],
                    np.array(labels, dtype=np.int64), lambda i: _load_png(paths[i]))


_ADDRESS = re.compile(rb"^[0-9A-Fa-f]+[ \t]", re.MULTILINE)


def parse_bytes_file(path) -> bytes:
    """BIG 2015 `.bytes` line: an address followed by 16 hex bytes; `??` = unknown, skipped."""
    with open(path, "rb") as f:
        text = _ADDRESS.sub(b"", f.read()).replace(b"??", b"")
    return bytes.fromhex(text.decode("ascii"))     # fromhex skips the whitespace


def load_big2015(root: str, cache_dir: str, width_table, ids_file: str | None,
                 subset: int | None, seed: int) -> ImageSet:
    """BIG 2015 training set: `<root>/train/<Id>.bytes` and `<root>/trainLabels.csv`.

    Each file is rendered once and cached at `<cache_dir>/big2015_png/<Id>.png`.
    """
    root, png_dir = Path(root), Path(cache_dir) / "big2015_png"
    label_csv = root / "trainLabels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(
            f"BIG 2015 not found at {root}. See data/README.md or set data.big2015_dir.")
    with open(label_csv) as f:
        rows = [(r["Id"], int(r["Class"])) for r in csv.DictReader(f)]
    present = {p.stem for p in (root / "train").glob("*.bytes")}
    if ids_file:
        # The committed id list defines the subset, so it can be recreated exactly.
        with open(ids_file) as f:
            wanted = {Path(line.strip().replace("\\", "/")).stem for line in f if line.strip()}
        missing = wanted - present
        if missing:
            raise FileNotFoundError(f"{len(missing)} ids in {ids_file} have no .bytes file in "
                                    f"{root / 'train'} (e.g. {sorted(missing)[:3]})")
        present = wanted
    else:
        print(f"  [big2015] using the {len(present)} extracted of {len(rows)} labelled samples")
    rows = [r for r in rows if r[0] in present]
    ids = np.array([r[0] for r in rows])
    labels = np.array([r[1] - 1 for r in rows], dtype=np.int64)   # classes are 1..9
    if subset and subset < len(ids):
        keep, _ = train_test_split(np.arange(len(ids)), train_size=subset,
                                   stratify=labels, random_state=seed)
        keep = np.sort(keep)
        ids, labels = ids[keep], labels[keep]
    png_dir.mkdir(parents=True, exist_ok=True)

    def loader(i):
        png = png_dir / f"{ids[i]}.png"
        if not png.exists():
            img = byteplot(parse_bytes_file(root / "train" / f"{ids[i]}.bytes"), width_table)
            Image.fromarray(img).save(png)
            return img
        return _load_png(png)

    # Class names as published with the Kaggle challenge (Class 1..9)
    families = ["Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda", "Tracur",
                "Kelihos_ver1", "Obfuscator.ACY", "Gatak"]
    return ImageSet("big2015", families, list(ids), labels, loader)


def make_synthetic(cfg_syn: dict, width_table, seed: int) -> ImageSet:
    """Fake byteplots: each family is a fixed byte 'template' with random
    mutations and size jitter; a fraction of samples are exact copies of an
    earlier sample of the same family (so dedup has something to find)."""
    rng = np.random.default_rng(seed)
    nf, per = cfg_syn["n_families"], cfg_syn["per_family"]
    lo, hi = cfg_syn["min_kb"] * 1024, cfg_syn["max_kb"] * 1024
    images, labels = [], []
    for fam in range(nf):
        base_len = int(rng.integers(lo, hi))
        template = rng.integers(0, 256, base_len, dtype=np.uint8)
        template[: base_len // 3] = rng.integers(0, 32 * (fam + 1) % 256 + 1)  # family-specific "header" texture
        for k in range(per):
            if k > 0 and rng.random() < cfg_syn["dup_rate"]:
                images.append(images[len(images) - int(rng.integers(1, k + 1))])
            else:
                n = int(base_len * rng.uniform(0.8, 1.2))
                raw = np.resize(template, n).copy()
                flip = rng.random(n) < 0.05
                raw[flip] = rng.integers(0, 256, int(flip.sum()), dtype=np.uint8)
                images.append(byteplot(raw.tobytes(), width_table))
            labels.append(fam)
    return ImageSet("synthetic", [f"fam{f}" for f in range(nf)],
                    [f"syn{i:05d}" for i in range(len(images))],
                    np.array(labels, dtype=np.int64), lambda i: images[i])


def load_dataset(name: str, cfg: dict) -> ImageSet:
    """`name` is the experiment's dataset; data.source=synthetic overrides it (CI)."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        return make_synthetic(d["synthetic"], d["width_table"], seed=0)
    if name == "malimg":
        return load_malimg(d["malimg_dir"])
    if name == "big2015":
        return load_big2015(d["big2015_dir"], d["cache_dir"], d["width_table"], d.get("big2015_ids"),
                            d.get("big2015_subset"), seed=0)
    raise ValueError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------- dedup
def _union_find_groups(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def exact_duplicate_groups(hashes: list[str]) -> np.ndarray:
    """Group id per sample; samples with identical pixel hashes share a group."""
    _, groups = np.unique(np.asarray(hashes), return_inverse=True)
    return groups


def duplicate_groups(hashes: list[str], feats: np.ndarray | None,
                     threshold: float | None) -> np.ndarray:
    """Exact duplicates, plus (if `threshold` is set) every pair of samples whose
    descriptor distance is below `threshold`, merged transitively."""
    exact = exact_duplicate_groups(hashes)
    if not threshold:
        return exact
    n = len(hashes)
    pairs = [(i, i0) for i, i0 in enumerate(_first_index(exact)) if i != i0]
    nn = NearestNeighbors(radius=threshold).fit(feats)
    _, idx = nn.radius_neighbors(feats)
    pairs += [(i, j) for i, row in enumerate(idx) for j in row if j > i]
    return _union_find_groups(n, pairs)


def _first_index(groups: np.ndarray) -> np.ndarray:
    first = {}
    return np.array([first.setdefault(g, i) for i, g in enumerate(groups)])


def nearest_neighbor_distances(feats: np.ndarray) -> np.ndarray:
    """Distance from each sample to its nearest *other* sample (for tuning the threshold)."""
    d, _ = NearestNeighbors(n_neighbors=2).fit(feats).kneighbors(feats)
    return d[:, 1]


def duplicate_rate_by_family(groups: np.ndarray, labels: np.ndarray, families) -> dict:
    """Fraction of each family's samples that share their group with another sample."""
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    in_dup = counts[inv] > 1
    return {fam: float(in_dup[labels == li].mean()) if (labels == li).any() else 0.0
            for li, fam in enumerate(families)}


# ---------------------------------------------------------------- folds
def make_folds(labels: np.ndarray, n_folds: int, seed: int,
               groups: np.ndarray | None = None) -> np.ndarray:
    """Fold index per sample. Stratified k-fold (E1, [1]'s protocol), or
    stratified *grouped* k-fold when `groups` is given (E2), which keeps each
    duplicate group entirely inside one fold."""
    fold = np.empty(len(labels), dtype=np.int64)
    dummy = np.zeros(len(labels))
    if groups is None:
        splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(dummy, labels)
    else:
        splits = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(
            dummy, labels, groups)
    for k, (_, te) in enumerate(splits):
        fold[te] = k
    return fold


def save_folds(path, ds: ImageSet, fold: np.ndarray, groups: np.ndarray | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "family", "group", "fold"])
        for i, name in enumerate(ds.names):
            w.writerow([name, ds.families[ds.labels[i]],
                        "" if groups is None else int(groups[i]), int(fold[i])])
