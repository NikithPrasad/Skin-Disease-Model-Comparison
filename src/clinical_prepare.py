"""Build the clinical-photo dataset (ordinary camera photos, not dermoscopy).

Our teacher's examples are phone photos of many everyday conditions (acne, leprosy, ringworm,
shingles, vitiligo, nail and hair problems, moles, ...), which HAM10000 does not cover. This
script merges five public datasets into 24 condition groups (23 conditions + healthy skin):

  DermNet (Kaggle shubhamgoel27/dermnet)             19.5k photos, 23 categories
  Fitzpatrick17k (Groh et al. 2021)                   16.6k photos, 114 conditions
  "31 classes of skin disease" (Kaggle kelixo25)      4.9k photos, incl. 3 types of leprosy
  Skin-Disease-Dataset (Kaggle subirbiswas19)         1.2k photos of skin infections
  MSLD v2.0 original images (Kaggle joydippaul)       healthy skin + viral rashes
  Oily/Dry/Normal skin types (Kaggle shakyadissanayake) photos of normal facial skin

The same photo often appears in several of these (and twice inside DermNet), and many patients
were photographed several times. Every image is described by an ImageNet-trained EfficientNet-B0
(1280 numbers, unit length) and compared with every other image (cosine similarity, on the GPU):
  >= 0.93  the same photo (re-saved, re-cropped or watermarked): keep one copy; if the copies
           carry different condition groups the photo is ambiguous and dropped
  >= 0.88  the same patient or photo series: always placed in the same split, so the test set
           never contains a near-copy of a training photo
(Thresholds picked by looking at sample pairs; a simple image hash chained thousands of
unrelated plain-skin photos together, so it was not used.) Splits are stratified by class:
70% train / 15% validation / 15% test.

Usage: python src/clinical_prepare.py --datasets E:/Nikith_Projects/datasets
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
OUT = ROOT / "data" / "processed_clinical"
SIZE = 224
SAME_PHOTO, SAME_SERIES = 0.93, 0.88  # cosine similarity of ImageNet features, see above

# key -> display name. Order defines the class index.
CLASSES = {
    "acne": "Acne, rosacea and clogged pores",
    "bacterial": "Bacterial skin infection",
    "blistering": "Blistering skin disease",
    "eczema": "Eczema and dermatitis",
    "fungal": "Fungal infection (ringworm)",
    "granuloma": "Granulomatous skin disease",
    "hair": "Hair loss",
    "healthy": "Healthy skin",
    "herpes": "Herpes, shingles and chickenpox",
    "leprosy": "Leprosy (Hansen's disease)",
    "pigment": "Sun-related and pigment changes",
    "lupus": "Lupus and connective-tissue disease",
    "melanoma": "Melanoma",
    "mole": "Mole or birthmark",
    "nail": "Nail condition",
    "psoriasis": "Psoriasis, lichen planus and similar",
    "infestation": "Scabies, bites and infestations",
    "cancer": "Skin cancer or pre-cancer",
    "benign": "Benign growth, cyst or scar",
    "systemic": "Nutritional or systemic disease signs",
    "urticaria": "Hives (urticaria)",
    "vascular": "Vascular lesion",
    "rash": "Drug reaction, viral rash or vasculitis",
    "warts": "Warts and molluscum",
}
KEYS = list(CLASSES)

DERMNET = {
    "Acne and Rosacea Photos": "acne",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "cancer",
    "Atopic Dermatitis Photos": "eczema",
    "Bullous Disease Photos": "blistering",
    "Cellulitis Impetigo and other Bacterial Infections": "bacterial",
    "Eczema Photos": "eczema",
    "Exanthems and Drug Eruptions": "rash",
    "Hair Loss Photos Alopecia and other Hair Diseases": "hair",
    "Herpes HPV and other STDs Photos": "herpes",
    "Light Diseases and Disorders of Pigmentation": "pigment",
    "Lupus and other Connective Tissue diseases": "lupus",
    # "Melanoma Skin Cancer Nevi and Moles" mixes melanoma with harmless moles: left out
    "Nail Fungus and other Nail Disease": "nail",
    "Poison Ivy Photos and other Contact Dermatitis": "eczema",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
    "Scabies Lyme Disease and other Infestations and Bites": "infestation",
    "Seborrheic Keratoses and other Benign Tumors": "benign",
    "Systemic Disease": "systemic",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",
    "Urticaria Hives": "urticaria",
    "Vascular Tumors": "vascular",
    "Vasculitis Photos": "rash",
    "Warts Molluscum and other Viral Infections": "warts",
}

FITZ = {
    "acne": "acne", "acne vulgaris": "acne", "rosacea": "acne", "rhinophyma": "acne",
    "folliculitis": "bacterial",
    "epidermolysis bullosa": "blistering", "acquired autoimmune bullous diseaseherpes gestationis": "blistering",
    "hailey hailey disease": "blistering",
    "eczema": "eczema", "allergic contact dermatitis": "eczema", "dyshidrotic eczema": "eczema",
    "seborrheic dermatitis": "eczema", "stasis edema": "eczema", "lichen simplex": "eczema",
    "neurodermatitis": "eczema", "perioral dermatitis": "eczema", "prurigo nodularis": "eczema",
    "sarcoidosis": "granuloma", "granuloma annulare": "granuloma", "necrobiosis lipoidica": "granuloma",
    "vitiligo": "pigment", "photodermatoses": "pigment", "drug induced pigmentary changes": "pigment",
    "sun damaged skin": "pigment", "xeroderma pigmentosum": "pigment",
    "lupus erythematosus": "lupus", "lupus subacute": "lupus", "scleroderma": "lupus", "dermatomyositis": "lupus",
    "melanoma": "melanoma", "malignant melanoma": "melanoma", "superficial spreading melanoma ssm": "melanoma",
    "lentigo maligna": "melanoma",
    "nevocytic nevus": "mole", "congenital nevus": "mole", "halo nevus": "mole", "becker nevus": "mole",
    "epidermal nevus": "mole", "nevus sebaceous of jadassohn": "mole",
    "paronychia": "nail",
    "psoriasis": "psoriasis", "pustular psoriasis": "psoriasis", "lichen planus": "psoriasis",
    "pityriasis rubra pilaris": "psoriasis", "pityriasis rosea": "psoriasis",
    "pityriasis lichenoides chronica": "psoriasis",
    "scabies": "infestation", "tick bite": "infestation", "nematode infection": "infestation",
    "tungiasis": "infestation", "myiasis": "infestation", "pediculosis lids": "infestation",
    "lyme disease": "infestation",
    "basal cell carcinoma": "cancer", "solid cystic basal cell carcinoma": "cancer",
    "basal cell carcinoma morpheiform": "cancer", "squamous cell carcinoma": "cancer",
    "actinic keratosis": "cancer", "kaposi sarcoma": "cancer", "mycosis fungoides": "cancer",
    "seborrheic keratosis": "benign", "syringoma": "benign", "milia": "benign", "pilar cyst": "benign",
    "mucous cyst": "benign", "dermatofibroma": "benign", "pilomatricoma": "benign", "xanthomas": "benign",
    "keloid": "benign", "juvenile xanthogranuloma": "benign", "neurofibromatosis": "benign",
    "acrodermatitis enteropathica": "systemic", "porphyria": "systemic", "acanthosis nigricans": "systemic",
    "calcinosis cutis": "systemic",
    "urticaria": "urticaria",
    "pyogenic granuloma": "vascular", "granuloma pyogenic": "vascular", "port wine stain": "vascular",
    "lymphangioma": "vascular", "telangiectases": "vascular",
    "drug eruption": "rash", "fixed eruptions": "rash", "erythema multiforme": "rash",
    "stevens johnson syndrome": "rash", "livedo reticularis": "rash",
}

ATLAS = {
    "Actinic Keratosis": "cancer", "Actinic keratosis": "cancer", "Basal Cell Carcinoma": "cancer",
    "Squamous cell carcinoma": "cancer", "Mycosis Fungoides": "cancer",
    "Dermatofibroma": "benign", "Pigmented Benign Keratosis": "benign", "Seborrheic keratosis": "benign",
    "Neurofibromatosis": "benign",
    "Epidermolysis Bullosa Pruriginosa": "blistering", "Hailey-Hailey Disease": "blistering",
    "Herpes Simplex": "herpes", "Impetigo": "bacterial",
    "Larva Migrans": "infestation", "Pediculosis Capitis": "infestation", "Tungiasis": "infestation",
    "Leprosy Borderline": "leprosy", "Leprosy Lepromatous": "leprosy", "Leprosy Tuberculoid": "leprosy",
    "Lichen Planus": "psoriasis", "Psoriasis": "psoriasis", "Pityriasis Rosea": "psoriasis",
    "Lupus Erythematosus Chronicus Discoides": "lupus",
    "Melanoma": "melanoma", "Nevus": "mole", "Molluscum Contagiosum": "warts",
    "Tinea Corporis": "fungal", "Tinea Nigra": "fungal", "Vascular lesion": "vascular",
}

INFECTIONS = {
    "BA- cellulitis": "bacterial", "BA-impetigo": "bacterial", "FU-athlete-foot": "fungal",
    "FU-nail-fungus": "nail", "FU-ringworm": "fungal", "PA-cutaneous-larva-migrans": "infestation",
    "VI-chickenpox": "herpes", "VI-shingles": "herpes",
}

MSLD = {"Healthy": "healthy", "Chickenpox": "herpes", "Measles": "rash", "HFMD": "rash"}

IMG_EXT = (".jpg", ".jpeg", ".png")


def collect(ds):
    rows = []

    def add(paths, key, source, detail):
        for p in paths:
            rows.append({"path": str(p), "key": key, "source": source, "detail": detail})

    for split in ["train", "test"]:
        for folder, key in DERMNET.items():
            d = ds / "dermnet" / split / folder
            add(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), key, "dermnet", folder)

    fz = ds / "fitzpatrick17k-original"
    meta = pd.read_csv(fz / "fitzpatrick17k.csv")
    for h, label in zip(meta.md5hash, meta.label):
        p = fz / "finalfitz17k" / f"{h}.jpg"
        if label in FITZ and p.exists():
            add([p], FITZ[label], "fitzpatrick17k", label)

    atlas = ds / "31-classes-of-skin-disease" / "Atlas dan ISIC2019 (31 classes)"
    for split in ["train", "test"]:
        for d in sorted((atlas / split).iterdir()):
            if d.name in ATLAS:
                add(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), ATLAS[d.name], "atlas31", d.name)

    inf = ds / "skin-disease-dataset" / "skin-disease-datasaet"
    for split in ["train_set", "test_set"]:
        for folder, key in INFECTIONS.items():
            d = inf / split / folder
            add(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), key, "infections8", folder)

    # MSLD: every fold is a partition of the same originals, so fold1's three parts cover all of them
    msld = ds / "mpox-skin-lesion-dataset-version-20-msld-v20" / "Original Images" / "Original Images" / "FOLDS" / "fold1"
    for part in ["Train", "Valid", "Test"]:
        for folder, key in MSLD.items():
            d = msld / part / folder
            add(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), key, "msld", folder)

    skin_types = ds / "oily-dry-and-normal-skin-types-dataset" / "Oily-Dry-Skin-Types"
    for part in ["train", "valid", "test"]:
        for folder in ["normal", "oily", "dry"]:
            d = skin_types / part / folder
            add(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT), "healthy", "skintypes", folder)
    return pd.DataFrame(rows)


def load_one(args):
    """Decode once: returns a 224x224 uint8 image, or None if unreadable."""
    path, crop_black = args
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    if crop_black:  # stock photos rotated onto a black canvas: cut the canvas away
        a = np.asarray(im)
        mask = a.max(2) > 25
        rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        if len(rows) and len(cols):
            im = im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
        w, h = im.size  # keep the central 70%, safely inside any rotated edge
        im = im.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    im = im.crop((l, t, l + SIZE, t + SIZE))
    return np.asarray(im, dtype=np.uint8)


@torch.no_grad()
def embed(images, batch=256):
    """Unit-length ImageNet features (EfficientNet-B0, no classifier) for every image."""
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=0).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(F.normalize(model((x - mean) / std).float(), dim=1).half())
    return torch.cat(out)


def similar_groups(feats, thresholds, block=2048):
    """For each threshold: union-find over all pairs with cosine similarity >= threshold.
    The n x n similarity matrix is never held at once: it is computed block by block on the GPU
    (O(n^2 d) multiply-adds, a few seconds); union-find with path halving is ~O(pairs)."""
    n = len(feats)
    parents = {t: np.arange(n) for t in thresholds}

    def find(par, i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for s in range(0, n, block):
        sim = feats[s:s + block] @ feats.T
        for t in thresholds:
            ii, jj = torch.nonzero(sim >= t, as_tuple=True)
            keep = jj > ii + s
            par = parents[t]
            for i, j in zip((ii[keep] + s).tolist(), jj[keep].tolist()):
                ri, rj = find(par, i), find(par, j)
                if ri != rj:
                    par[max(ri, rj)] = min(ri, rj)
    return {t: np.array([find(par, i) for i in range(n)]) for t, par in parents.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=ROOT.parent / "datasets")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = collect(args.datasets)
    print(f"{len(df)} images listed", flush=True)
    jobs = [(p, s == "skintypes") for p, s in zip(df.path, df.source)]
    images = np.empty((len(df), SIZE, SIZE, 3), np.uint8)  # filled in place: one copy in RAM
    ok = np.zeros(len(df), bool)
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, r in enumerate(ex.map(load_one, jobs, chunksize=64)):
            if r is not None:
                images[i], ok[i] = r, True
    df, images = df[ok].reset_index(drop=True), images[ok]
    print(f"{len(df)} decoded ({(~ok).sum()} unreadable)", flush=True)

    groups = similar_groups(embed(images), [SAME_PHOTO, SAME_SERIES])
    df["photo"], df["group"] = groups[SAME_PHOTO], groups[SAME_SERIES]
    # A photo whose copies carry different condition groups is ambiguous: drop it
    conflict = df.groupby("photo").key.transform("nunique") > 1
    first = ~df.duplicated("photo")  # keep one copy of each photo
    keep = (~conflict & first).to_numpy()
    print(f"{df.photo.nunique()} distinct photos in {df.group.nunique()} patient/series groups; "
          f"dropped {(~first).sum()} extra copies and {conflict.sum()} images with conflicting labels", flush=True)
    df, images = df[keep].reset_index(drop=True), images[keep]

    df["y"] = df.key.map(KEYS.index)
    # 70/15/15, whole patient/series groups at a time
    idx = np.arange(len(df))
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=0)
    fold = np.empty(len(df), int)
    for f, (_, te) in enumerate(sgkf.split(idx, df.y, df.group)):
        fold[te] = f
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))

    for split in ["train", "val", "test"]:
        m = (df.split == split).to_numpy()
        np.save(OUT / f"{split}_x.npy", images[m])
        np.save(OUT / f"{split}_y.npy", df.y[m].to_numpy().astype(np.int64))
    df.to_csv(OUT / "manifest.csv", index=False)
    (OUT / "classes.json").write_text(json.dumps({"keys": KEYS, "names": CLASSES}, indent=1))

    table = pd.crosstab(df.key, df.split).reindex(KEYS)[["train", "val", "test"]]
    table["total"] = table.sum(axis=1)
    print(table.to_string())
    print(pd.crosstab(df.key, df.source).reindex(KEYS).to_string())


if __name__ == "__main__":
    main()
