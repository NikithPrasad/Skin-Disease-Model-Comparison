"""Build an 8-class dataset: the 7 HAM10000 lesion types plus "healthy skin".

Healthy-skin examples are cut from the HAM10000 photos themselves: the dataset's lesion
segmentation masks show exactly where the lesion is, so a square of skin well away from it
is lesion-free. One patch is taken per lesion and it inherits that lesion's train/val/test
split, so no lesion (or its surrounding skin) appears on both sides of the split.

Output: data/processed_8class/{train,val,test}_{x,y}.npy  (label 7 = healthy skin)
Usage:  python src/healthy_patches.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SRC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed_8class"
MASKS = RAW / "HAM10000_segmentations" / "HAM10000_segmentations_lesion_tschandl"
HEALTHY = 7
SIZE = 256
rng = np.random.default_rng(0)


def window_sums(a, k):
    """Sum of every k x k window of a 2-D array, via an integral image: O(H*W)."""
    s = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


def healthy_patch(image_id):
    img = np.asarray(Image.open(RAW / "HAM10000_images" / f"{image_id}.jpg").convert("RGB"))
    mask = np.asarray(Image.open(MASKS / f"{image_id}_segmentation.png").convert("L")) > 127
    h, w = mask.shape
    # Keep a margin around the lesion, and avoid the dark vignette corners of dermoscopes
    unsafe = binary_dilation(mask, iterations=max(4, h // 40)) | (img.mean(2) < 40)
    k = int(min(h, w) * rng.uniform(0.33, 0.5))
    bad = window_sums(unsafe.astype(np.int32), k)
    ys, xs = np.nonzero(bad == 0)
    if len(ys) == 0:
        return None  # the lesion fills too much of the photo
    i = rng.integers(len(ys))
    y, x = ys[i], xs[i]
    return np.asarray(Image.fromarray(img[y:y + k, x:x + k]).resize((SIZE, SIZE), Image.BICUBIC))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SRC / "splits.csv")
    one_per_lesion = meta.drop_duplicates("lesion_id")
    patches = {"train": [], "val": [], "test": []}
    skipped = 0
    for image_id, split in zip(one_per_lesion.image_id, one_per_lesion.split):
        p = healthy_patch(image_id)
        if p is None:
            skipped += 1
        else:
            patches[split].append(p)
    for split, items in patches.items():
        x = np.concatenate([np.load(SRC / f"{split}_x.npy"), np.stack(items)])
        y = np.concatenate([np.load(SRC / f"{split}_y.npy"), np.full(len(items), HEALTHY)])
        np.save(OUT / f"{split}_x.npy", x)
        np.save(OUT / f"{split}_y.npy", y)
        print(f"{split}: {len(items)} healthy patches added -> {len(y)} images")
    print(f"skipped {skipped} lesions with no lesion-free area big enough")


if __name__ == "__main__":
    main()
