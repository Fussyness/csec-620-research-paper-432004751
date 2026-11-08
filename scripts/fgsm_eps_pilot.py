"""Pilot for choosing the FGSM step size (assumption A22; config attack.fgsm_eps).

Trains a small pilot CNN (ResNet-18, 3 epochs, 128 px) on a random subset of
Malimg, then attacks other Malimg images at one append budget with several step
sizes. For each step it prints how often FGSM raises the loss compared with plain
zero-appending, and the accuracy after the attack. The pilot model is separate
from the models evaluated in E4.

Run from the repo root (needs the Malimg data; a GPU is strongly recommended):
    python scripts/fgsm_eps_pilot.py [--train 3000] [--test 100] [--budget 0.05] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attack import append_bytes, fgsm_append  # noqa: E402
from src.data import load_malimg  # noqa: E402
from src.model import CNNClassifier  # noqa: E402
from src.train import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05, 0.2, 0.5, 1.0])
    a = ap.parse_args()

    cfg = load_config(a.config)
    wt = cfg["data"]["width_table"]
    cfg["models"]["cnn"].update(backbone="resnet18", input_size=128, epochs=3)
    ds = load_malimg(cfg["data"]["malimg_dir"])
    y = ds.labels
    idx = np.random.default_rng(a.seed).permutation(len(ds))
    tr, te = idx[:a.train], idx[a.train:a.train + a.test]

    cnn = CNNClassifier(cfg)
    cnn.fit(np.stack([cnn.represent(ds.load(int(i))) for i in tr]), y[tr], len(ds.families), a.seed)

    def loss_pred(img, label):
        x = (torch.from_numpy(img).float() / 255)[None, None].to(cnn.device)
        with torch.no_grad():
            lg = cnn.logits_from_raw(x)
        return F.cross_entropy(lg, torch.tensor([label], device=cnn.device)).item(), int(lg.argmax())

    ims = [(ds.load(int(j)), int(y[j])) for j in te]
    clean = np.mean([loss_pred(im, l)[1] == l for im, l in ims])
    base = [loss_pred(append_bytes(im, a.budget, "zero", None, wt), l) for im, l in ims]
    base_acc = np.mean([p == l for (_, p), (_, l) in zip(base, ims)])
    print(f"pilot: {a.train} train / {a.test} attacked images, budget {a.budget}, seed {a.seed}")
    print(f"clean accuracy {clean:.2f}, zero-append accuracy {base_acc:.2f}")
    print("eps    | loss up vs zero-append | accuracy after FGSM")
    for eps in a.eps:
        r = [loss_pred(fgsm_append(cnn, im, l, a.budget, eps, wt), l) for im, l in ims]
        up = np.mean([lf >= lz - 1e-6 for (lf, _), (lz, _) in zip(r, base)])
        acc = np.mean([p == l for (_, p), (_, l) in zip(r, ims)])
        print(f"{eps:<6} | {up:>21.0%} | {acc:.2f}")


if __name__ == "__main__":
    main()
