"""Compare ways to detect "this is not a close-up skin photo" for the 8-class detection model.

Every method gets its threshold from the validation set (keep 99% of real photos) and is then
scored on: test lesions/healthy (should pass), the external ISIC set (should pass), and
non-skin images (should be rejected).
Usage: python src/ood_experiment.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

import export_single_model as E
from models import build_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    infos = [json.loads(p.read_text()) for p in E.RUNS8.glob("efficientnet_b0_seed*/info.json")]
    best = max(infos, key=lambda i: i["best_val_macro_f1"])
    state = torch.load(E.RUNS8 / f"efficientnet_b0_seed{best['seed']}" / "best.pt", map_location="cpu", weights_only=True)
    model = build_model("efficientnet_b0", pretrained=False, num_classes=8)
    model.load_state_dict({k: v.half().float() if v.is_floating_point() else v for k, v in state.items()})
    device = torch.device("cuda")
    model.eval().to(device)

    data = {s: (np.load(E.DATA8 / f"{s}_x.npy", mmap_mode="r"), np.load(E.DATA8 / f"{s}_y.npy")) for s in ["train", "val", "test"]}
    data["external"] = (np.load(ROOT / "data/processed/external_x.npy", mmap_mode="r"), None)
    walls = sorted(p for p in Path("C:/Windows/Web").rglob("*.jpg") if p.stat().st_size > 50_000)
    data["non_skin"] = (E.load_images(walls), None)
    out = {s: E.run(model, x, device) for s, (x, _) in data.items()}
    feats = {s: f for s, (_, f) in out.items()}
    imgs = {s: x for s, (x, _) in data.items()}
    ytr = data["train"][1]

    def normed(f):
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    rng = np.random.default_rng(0)
    bank = normed(feats["train"][rng.choice(len(ytr), 3000, replace=False)])
    pca64 = E.OOD().fit(feats["train"], ytr)
    lw_full = LedoitWolf().fit(feats["train"])
    pca_res = PCA(128, random_state=0).fit(feats["train"])

    def skin_fraction(x):
        """Share of pixels in a broad skin-colour range (YCbCr), computed on 64x64 thumbnails."""
        x = np.asarray(x)[:, ::4, ::4].astype(np.float32)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return ((cb > 77) & (cb < 135) & (cr > 133) & (cr < 180)).mean((1, 2))

    methods = {
        "pca64 mahalanobis (current)": lambda s: pca64.score(feats[s]),
        "full mahalanobis": lambda s: lw_full.mahalanobis(feats[s]),
        "pca128 residual": lambda s: np.linalg.norm(feats[s] - pca_res.inverse_transform(pca_res.transform(feats[s])), axis=1),
        "knn cosine (k=10)": lambda s: -np.sort(normed(feats[s]) @ bank.T, axis=1)[:, -10:].mean(1),
        "low skin colour": lambda s: -skin_fraction(imgs[s]),
        "max softmax (neg)": lambda s: -out[s][0].max(1),
    }
    print(f"{'method':30s} {'test':>6s} {'extern':>7s} {'non-skin':>9s}   (share rejected)")
    for name, fn in methods.items():
        t = np.quantile(fn("val"), 0.99)
        r = {s: (fn(s) > t).mean() for s in ["test", "external", "non_skin"]}
        print(f"{name:30s} {r['test']:6.3f} {r['external']:7.3f} {r['non_skin']:9.3f}")
    # Combination: reject if either the best feature score or the colour check says so (each at 99.5%)
    f1, f2 = methods["knn cosine (k=10)"], methods["low skin colour"]
    t1, t2 = np.quantile(f1("val"), 0.995), np.quantile(f2("val"), 0.995)
    r = {s: ((f1(s) > t1) | (f2(s) > t2)).mean() for s in ["test", "external", "non_skin"]}
    print(f"{'knn OR colour (99.5% each)':30s} {r['test']:6.3f} {r['external']:7.3f} {r['non_skin']:9.3f}")


if __name__ == "__main__":
    main()
