"""Same test as ood_experiment.py, but with features from the general ImageNet-pretrained
EfficientNet-B0 (it knows everyday objects and scenes), not the skin-trained model."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
import numpy as np, timm, torch
from sklearn.covariance import LedoitWolf
import export_single_model as E
import train as T

@torch.no_grad()
def feats_of(model, x_uint8, device, batch=128):
    out = []
    for i in range(0, len(x_uint8), batch):
        x = T.normalize(T.resize_eval(T.to_input(torch.as_tensor(np.asarray(x_uint8[i:i + batch])), device)))
        out.append(model(x).cpu())
    return torch.cat(out).numpy()

device = torch.device("cuda")
gen = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=0).eval().to(device)  # pooled features only
data = {s: np.load(E.DATA8 / f"{s}_x.npy", mmap_mode="r") for s in ["train", "val", "test"]}
data["external"] = np.load(ROOT / "data/processed/external_x.npy", mmap_mode="r")
walls = sorted(p for p in Path("C:/Windows/Web").rglob("*.jpg") if p.stat().st_size > 50_000)
data["non_skin"] = E.load_images(walls)
F = {s: feats_of(gen, x, device) for s, x in data.items()}
n = lambda f: f / np.linalg.norm(f, axis=1, keepdims=True)
rng = np.random.default_rng(0)
bank = n(F["train"][rng.choice(len(F["train"]), 3000, replace=False)])
lw = LedoitWolf().fit(F["train"])
methods = {
    "general knn cosine (k=10)": lambda s: -np.sort(n(F[s]) @ bank.T, axis=1)[:, -10:].mean(1),
    "general mahalanobis": lambda s: lw.mahalanobis(F[s]),
}
print(f"{'method':30s} {'test':>6s} {'extern':>7s} {'non-skin':>9s}")
for name, fn in methods.items():
    for keep in [0.99, 0.995]:
        t = np.quantile(fn("val"), keep)
        r = {s: (fn(s) > t).mean() for s in ["test", "external", "non_skin"]}
        print(f"{name + f' @{keep}':30s} {r['test']:6.3f} {r['external']:7.3f} {r['non_skin']:9.3f}")
sc = methods["general mahalanobis"]("non_skin"); t = np.quantile(methods["general mahalanobis"]("val"), 0.99)
print("non-skin images that still pass:", [walls[i].name for i in np.where(sc <= t)[0]])
