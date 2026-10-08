"""Build lesion-grouped train/val/test splits and cache resized images as .npy.

HAM10000 contains several photos of the same lesion (10,015 images, 7,470 lesions).
Splitting by image would put photos of one lesion in both train and test and inflate
scores, so every split here is grouped by lesion_id.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CACHE_SIZE = 256  # images are cached at 256x256; models train on 224x224 crops
SEED = 42


def load_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC))


def cache_images(paths, out_file):
    with ThreadPoolExecutor(max_workers=16) as pool:
        arr = np.stack(list(pool.map(load_image, paths)))
    np.save(out_file, arr)
    print(f"  saved {out_file.name}: {arr.shape}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(RAW / "HAM10000_metadata")
    meta["label"] = meta["dx"].map(CLASSES.index)
    meta["path"] = [str(RAW / "HAM10000_images" / f"{i}.jpg") for i in meta["image_id"]]

    # ~20% test, then ~12.5% of the rest as validation -> roughly 70 / 10 / 20
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(sgkf.split(meta, meta["label"], meta["lesion_id"]))
    trainval = meta.iloc[trainval_idx]
    sgkf = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(trainval, trainval["label"], trainval["lesion_id"]))

    meta["split"] = "test"
    meta.loc[trainval.index[train_idx], "split"] = "train"
    meta.loc[trainval.index[val_idx], "split"] = "val"

    # Sanity check: no lesion appears in more than one split
    assert (meta.groupby("lesion_id")["split"].nunique() == 1).all()
    meta.drop(columns="path").to_csv(OUT / "splits.csv", index=False)

    print(pd.crosstab(meta["dx"], meta["split"], margins=True))
    for split in ["train", "val", "test"]:
        part = meta[meta["split"] == split]
        cache_images(part["path"], OUT / f"{split}_x.npy")
        np.save(OUT / f"{split}_y.npy", part["label"].to_numpy())

    # External test set: official ISIC 2018 Task 3 test images (different source)
    ext = pd.read_csv(RAW / "ISIC2018_Task3_Test_GroundTruth.csv")
    ext["label"] = ext["dx"].map(CLASSES.index)
    img_dir = RAW / "ISIC2018_test" / "ISIC2018_Task3_Test_Images"
    exists = ext["image_id"].map(lambda i: (img_dir / f"{i}.jpg").exists())
    if (~exists).any():
        print("external test: skipping images missing from the download:", list(ext.loc[~exists, "image_id"]))
    ext = ext[exists]
    cache_images([img_dir / f"{i}.jpg" for i in ext["image_id"]], OUT / "external_x.npy")
    np.save(OUT / "external_y.npy", ext["label"].to_numpy())
    print("external test:", len(ext), "images")


if __name__ == "__main__":
    main()
