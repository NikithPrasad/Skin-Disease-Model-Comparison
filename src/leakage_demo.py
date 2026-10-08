"""Shows why the data must be split by lesion, not by image.

Builds a naive split by image (same sizes and class balance as the real split, but photos of
one lesion can land in both train and test), trains MobileNetV3 on it with the identical
recipe, and compares against the lesion-grouped run (results/runs/mobilenetv3_seed0).

Usage: python src/leakage_demo.py      (~6 minutes on an RTX 4060)
Output: results/leakage_demo/summary.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
NAIVE = ROOT / "data" / "processed_by_image"
OUT = ROOT / "results" / "leakage_demo"
MODEL, SEED = "mobilenetv3", 0


def build_naive_split():
    meta = pd.read_csv(DATA / "splits.csv")
    parts = {s: np.load(DATA / f"{s}_x.npy", mmap_mode="r") for s in ["train", "val", "test"]}
    # Rows of each cached array are in the same order as that split's rows in splits.csv
    order = pd.concat([meta[meta.split == s] for s in ["train", "val", "test"]]).reset_index(drop=True)
    x_all = np.concatenate([parts[s] for s in ["train", "val", "test"]])
    n_test, n_val = (meta.split == "test").sum(), (meta.split == "val").sum()
    idx = np.arange(len(order))
    trval, test = train_test_split(idx, test_size=n_test, stratify=order.dx, random_state=42)
    train, val = train_test_split(trval, test_size=n_val, stratify=order.dx[trval], random_state=42)
    NAIVE.mkdir(parents=True, exist_ok=True)
    labels = order.dx.map(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"].index).to_numpy()
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        np.save(NAIVE / f"{name}_x.npy", x_all[np.sort(ids)])
        np.save(NAIVE / f"{name}_y.npy", labels[np.sort(ids)])
    train_lesions = set(order.lesion_id[train])
    shared = order.lesion_id[np.sort(test)].isin(train_lesions).mean()
    return float(shared)


def scores(run_dir, split):
    probs, y = np.load(run_dir / f"{split}_probs.npy"), np.load(run_dir / f"{split}_labels.npy")
    pred = probs.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float((pred == y).mean())}


def main():
    shared = build_naive_split()
    print(f"naive split: {shared:.1%} of test images show a lesion that is also in the training set")
    run = OUT / f"{MODEL}_seed{SEED}"
    if not (run / "info.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--model", MODEL, "--seed", str(SEED),
                        "--data-dir", str(NAIVE), "--runs-dir", str(OUT)], check=True)
    proper = ROOT / "results" / "runs" / f"{MODEL}_seed{SEED}"
    summary = {
        "model": MODEL, "seed": SEED,
        "test_images_with_lesion_seen_in_training": shared,
        "split_by_image": {"own_test_set": scores(run, "test"), "external_test_set": scores(run, "external")},
        "split_by_lesion": {"own_test_set": scores(proper, "test"), "external_test_set": scores(proper, "external")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
