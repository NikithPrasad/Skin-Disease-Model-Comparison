"""Train one model on the HAM10000 splits and evaluate it on the internal and external test sets.

Every model gets the same data, augmentation, loss, epochs and selection rule, so the
only thing that changes between runs is the architecture (and its learning rate:
pretrained models are fine-tuned gently, the from-scratch CNN needs a higher rate).

Usage: python src/train.py --model resnet50 --seed 0
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))  # keep downloads in the project

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

from models import MODELS, NUM_CLASSES, build_model

DATA = ROOT / "data" / "processed"
RUNS = ROOT / "results" / "runs"
IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load_split(split, data_dir=DATA):
    x = torch.from_numpy(np.load(data_dir / f"{split}_x.npy"))  # N,H,W,3 uint8, kept on CPU
    y = torch.from_numpy(np.load(data_dir / f"{split}_y.npy")).long()
    return x, y


def to_input(x_uint8, device):
    return x_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255).contiguous()


def augment(x):
    """GPU augmentation. Dermoscopy images have no natural orientation, so use full
    rotations and flips, plus mild zoom/shift and colour jitter."""
    n, dev = x.shape[0], x.device
    angle = torch.rand(n, device=dev) * 2 * math.pi
    scale = torch.empty(n, device=dev).uniform_(0.75, 1.0)  # <1 zooms in (crops)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    shift = torch.empty(n, 2, device=dev).uniform_(-0.1, 0.1)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, IMG, IMG), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    s = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    x = x * b
    x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) * c + x.mean(dim=(1, 2, 3), keepdim=True)
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * s + gray
    return x.clamp_(0, 1)


def resize_eval(x):
    return F.interpolate(x, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, x_all, device, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(x_all), batch_size):
        x = normalize(resize_eval(to_input(x_all[i:i + batch_size], device)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs.append(logits.float().softmax(1).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def count_gflops(model):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = torch.randn(1, 3, IMG, IMG, device=next(model.parameters()).device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def measure_inference(model, device):
    """Median latency for a single image on GPU and CPU, plus GPU throughput at batch 64."""
    out = {}
    model.eval()
    for dev_name, dev, reps in [("gpu", device, 100), ("cpu", torch.device("cpu"), 30)]:
        m = model.to(dev).float()
        x = torch.randn(1, 3, IMG, IMG, device=dev)
        for _ in range(10):
            m(x)
        times = []
        for _ in range(reps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out[f"latency_ms_{dev_name}_batch1"] = float(np.median(times) * 1000)
    model.to(device)
    x = torch.randn(64, 3, IMG, IMG, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
    out["throughput_img_per_s_gpu_batch64"] = float(64 * 20 / (time.perf_counter() - t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-dir", type=Path, default=DATA, help="folder with train/val/test arrays")
    ap.add_argument("--runs-dir", type=Path, default=RUNS, help="where to save the run")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="8 for the dataset with healthy skin")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    run_dir = args.runs_dir / f"{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split("train", args.data_dir)
    x_val, y_val = load_split("val", args.data_dir)

    # Class-weighted loss: rarer classes count more (inverse square-root frequency, mean 1)
    counts = torch.bincount(y_train, minlength=args.num_classes).float()
    weights = counts.pow(-0.5)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    model = build_model(args.model, num_classes=args.num_classes).to(device)
    lr = MODELS[args.model][1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps_per_epoch = len(x_train) // args.batch_size
    total_steps, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    history, best_f1 = [], -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        loss_sum = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x = normalize(augment(to_input(x_train[idx], device)))
            y = y_train[idx].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()

        probs = predict(model, x_val, device)
        pred = probs.argmax(1)
        val_loss = F.cross_entropy(torch.from_numpy(np.log(probs + 1e-8)), y_val).item()
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / steps_per_epoch,
            "val_loss": val_loss,
            "val_acc": float((pred == y_val.numpy()).mean()),
            "val_macro_f1": float(f1_score(y_val, pred, average="macro")),
            "val_balanced_acc": float(balanced_accuracy_score(y_val, pred)),
            "elapsed_s": time.time() - start,
        }
        history.append(rec)
        marker = ""
        if rec["val_macro_f1"] > best_f1:  # keep the epoch with the best validation macro F1
            best_f1 = rec["val_macro_f1"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            marker = "  *best"
        print(f"[{args.model} s{args.seed}] ep {epoch:2d}  train_loss {rec['train_loss']:.3f}  "
              f"val_acc {rec['val_acc']:.3f}  val_F1 {rec['val_macro_f1']:.3f}  "
              f"val_balacc {rec['val_balanced_acc']:.3f}  ({rec['elapsed_s']:.0f}s){marker}", flush=True)
    train_time = time.time() - start

    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
    best_epoch = max(history, key=lambda r: r["val_macro_f1"])["epoch"]
    for split in ["test", "external"]:
        x, y = load_split(split, args.data_dir if split == "test" else DATA)  # external set never changes
        np.save(run_dir / f"{split}_probs.npy", predict(model, x, device))
        np.save(run_dir / f"{split}_labels.npy", y.numpy())

    info = {
        "model": args.model,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "lr": lr,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_time_s": train_time,
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_mb": (run_dir / "best.pt").stat().st_size / 1e6,
        "gflops": count_gflops(model),
        **measure_inference(model, device),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    (run_dir / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
