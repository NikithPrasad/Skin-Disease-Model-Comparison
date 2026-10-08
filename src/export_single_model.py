"""Export the 8-class EfficientNet-B0 (7 lesion types + healthy skin) as the Skin-Disease-Detection site.

Writes into <site>/app/:
  models/model.pt     weights of the seed with the best validation macro F1 (float16)
  models/general.pt   general ImageNet-pretrained EfficientNet-B0 (float16), used only by the photo check
  models/ood.npz      the "is this a close-up skin photo?" check (see below)
  models/meta.json    scores, thresholds and per-class results, all measured with the shipped weights
  static/figures/     confusion matrix (light + dark)
  static/examples/healthy.jpg  a healthy-skin sample from the test set

Decisions are made on the VALIDATION set and only reported on the test sets:
  * melanoma warning: the highest melanoma-probability cut-off that still flags >= 80% of
    validation melanomas;
  * photo check: each photo is described by a GENERAL ImageNet-pretrained EfficientNet-B0 (which
    knows everyday objects and scenes; the skin model's own features could not tell skin from
    non-skin, see src/ood_experiment*.py). Score = Mahalanobis distance of those features from the
    training photos. Photos above the 99.5th percentile of validation photos are rejected as
    "not a close-up skin photo", so ~0.5% of real photos are turned away.

Usage: python src/export_single_model.py [--site ../Skin-Disease-Detection]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))  # the general model is cached here

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

import report
import train as T
from models import CLASSES, build_model

ROOT = Path(__file__).resolve().parent.parent
DATA8 = ROOT / "data" / "processed_8class"
RUNS8 = ROOT / "results" / "runs_8class"
LABELS = CLASSES + ["healthy"]
MEL, HEALTHY = LABELS.index("mel"), LABELS.index("healthy")
TARGET_MEL_RECALL, OOD_KEEP = 0.80, 0.995
GENERAL_ID = "efficientnet_b0.ra_in1k"


@torch.no_grad()
def run(model, x_uint8, device, batch=128):
    """Probabilities and pooled features, float32 (same numbers as the CPU website)."""
    probs, feats = [], []
    for i in range(0, len(x_uint8), batch):
        x = T.normalize(T.resize_eval(T.to_input(torch.as_tensor(np.asarray(x_uint8[i:i + batch])), device)))
        f = model.forward_head(model.forward_features(x), pre_logits=True)
        probs.append(model.get_classifier()(f).softmax(1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def load_images(paths):
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)) for p in paths])


@torch.no_grad()
def general_features(model, x_uint8, device, batch=128):
    out = []
    for i in range(0, len(x_uint8), batch):
        x = T.normalize(T.resize_eval(T.to_input(torch.as_tensor(np.asarray(x_uint8[i:i + batch])), device)))
        out.append(model(x).cpu())
    return torch.cat(out).numpy()


def metrics(p, y, mel_t, ood=None, ood_t=None):
    pred, labels = p.argmax(1), sorted(set(y))
    out = {
        "macro_f1": f1_score(y, pred, average="macro", labels=labels),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_auc": roc_auc_score(y, p[:, labels] / p[:, labels].sum(1, keepdims=True), multi_class="ovr", labels=labels),
        "melanoma_recall_top_answer": recall_score(y, pred, labels=[MEL], average="macro"),
        "melanoma_recall_with_warning": (p[y == MEL, MEL] >= mel_t).mean(),
        "warning_on_non_melanoma": (p[y != MEL, MEL] >= mel_t).mean(),
        "lesions_called_healthy": (pred[y != HEALTHY] == HEALTHY).mean(),
        "melanomas_called_healthy": (pred[y == MEL] == HEALTHY).mean(),
        "per_class_recall": {LABELS[c]: recall_score(y, pred, labels=[c], average="macro") for c in labels},
    }
    if ood is not None:
        out["rejected_by_photo_check"] = (ood > ood_t).mean()
    return out


def confusion_figures(p, y, out_dir):
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(LABELS))).astype(float)
    norm = cm / cm.sum(1, keepdims=True)
    for theme in report.THEMES:
        report.set_theme(theme)
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        ax.imshow(norm, cmap=report.BLUES, vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8, color=report.text_on(norm[i, j]))
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.savefig(out_dir / f"confusion{'' if theme == 'light' else '_dark'}.png")
        plt.close(fig)


def write_readme_numbers(readme, m):
    """Rewrite the README's accuracy section (between markers) from meta.json, so it never goes stale."""
    t, e, pc = m["test"], m["external"], m["photo_check"]
    pct = lambda v: f"{v * 100:.1f}%" if 0 < v < 0.1 else f"{round(v * 100)}%"  # small rates keep a decimal
    names = {"akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis", "df": "Dermatofibroma",
             "mel": "Melanoma", "nv": "Melanocytic nevus (mole)", "vasc": "Vascular lesion", "healthy": "Healthy skin"}
    rows = "\n".join(f"| {names[c]} | {pct(t['per_class_recall'][c])} | "
                     f"{pct(e['per_class_recall'][c]) if c in e['per_class_recall'] else 'not in this set'} |" for c in names)
    block = f"""Measured on photos the model never saw in training, using exactly the weights shipped here
(seed {m['seed']} of {m['seeds_trained']} training runs, chosen by validation score).

| | HAM10000 test (lesions + healthy-skin patches, split by lesion) | ISIC 2018 test (1,511 lesion photos, separate collection) |
|---|---|---|
| Macro F1 | {t['macro_f1']:.3f} | {e['macro_f1']:.3f} |
| Balanced accuracy | {t['balanced_acc']:.3f} | {e['balanced_acc']:.3f} |
| Accuracy | {t['accuracy']:.3f} | {e['accuracy']:.3f} |
| Lesions wrongly called healthy skin | {pct(t['lesions_called_healthy'])} | {pct(e['lesions_called_healthy'])} |

Correct answers by type (recall):

| Type | HAM10000 test | ISIC 2018 test |
|---|---|---|
{rows}

Healthy-skin examples are patches cut from around the lesions in HAM10000 photos (using the dataset's lesion
outlines), so they are dermoscopy close-ups too. The healthy score above is therefore optimistic for other
kinds of photos; those are handled by the photo check below.

### The melanoma warning

The top answer alone names {pct(t['melanoma_recall_top_answer'])} of melanomas. The site also warns whenever the melanoma
probability is {m['melanoma_threshold'] * 100:.1f}% or more (threshold picked on validation data to catch at least
{pct(m['target_melanoma_recall'])} of validation melanomas):

| | HAM10000 test | ISIC 2018 test |
|---|---|---|
| Melanomas caught by the top answer | {pct(t['melanoma_recall_top_answer'])} | {pct(e['melanoma_recall_top_answer'])} |
| Melanomas caught with the warning | {pct(t['melanoma_recall_with_warning'])} | {pct(e['melanoma_recall_with_warning'])} |
| Other photos that also get a warning | {pct(t['warning_on_non_melanoma'])} | {pct(e['warning_on_non_melanoma'])} |

### The photo check

Before any diagnosis, the photo is compared with the training photos using a general ImageNet-trained
EfficientNet-B0 (it knows everyday objects and scenes). Photos too unlike the training photos get
"this doesn't look like a close-up skin photo" and no diagnosis. Threshold: keep {m['ood_keep_on_validation'] * 100:.1f}% of validation photos.

| | Rejected |
|---|---|
| Real HAM10000 test lesion photos | {pct(pc['rejected_test_lesions'])} |
| Healthy-skin test patches | {pct(pc['rejected_test_healthy'])} |
| ISIC 2018 photos (separate collection) | {pct(pc['rejected_external'])} |
| Non-skin images ({pc['non_skin_images']} {pc['non_skin_note']}) | {pct(pc['rejected_non_skin'])} |

The skin model's own features could not tell skin from non-skin (at most 58% of the non-skin images
rejected; see `src/ood_experiment.py` in the comparison repo), which is why a general model is used.
Ordinary phone photos of skin have not been tested, because no such labelled set was available."""
    text = readme.read_text(encoding="utf-8")
    s, e2 = text.index("<!-- numbers:start -->"), text.index("<!-- numbers:end -->")
    readme.write_text(text[:s] + "<!-- numbers:start -->\n" + block + "\n" + text[e2:], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=ROOT.parent / "Skin-Disease-Detection")
    args = ap.parse_args()
    app = args.site / "app"

    infos = [json.loads(p.read_text()) for p in RUNS8.glob("efficientnet_b0_seed*/info.json")]
    best = max(infos, key=lambda i: i["best_val_macro_f1"])
    state = torch.load(RUNS8 / f"efficientnet_b0_seed{best['seed']}" / "best.pt", map_location="cpu", weights_only=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    torch.save(state, app / "models" / "model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("efficientnet_b0", pretrained=False, num_classes=len(LABELS))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.eval().to(device)

    sets = {s: (np.load(DATA8 / f"{s}_x.npy", mmap_mode="r"), np.load(DATA8 / f"{s}_y.npy")) for s in ["train", "val", "test"]}
    sets["external"] = (np.load(ROOT / "data/processed/external_x.npy", mmap_mode="r"), np.load(ROOT / "data/processed/external_y.npy"))
    wallpapers = sorted(p for p in Path("C:/Windows/Web").rglob("*.jpg") if p.stat().st_size > 50_000)
    out = {s: run(model, x, device) for s, (x, _) in sets.items()}

    # Melanoma warning threshold, chosen on validation
    pv, yv = out["val"][0], sets["val"][1]
    mel_p = np.sort(pv[yv == MEL, MEL])[::-1]
    mel_t = float(next(c for c in mel_p if (pv[yv == MEL, MEL] >= c).mean() >= TARGET_MEL_RECALL))

    # Photo check, fitted on training features, threshold from validation
    # Photo check: general ImageNet features, fitted on training photos, threshold from validation
    general = timm.create_model(GENERAL_ID, pretrained=True, num_classes=0)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in general.state_dict().items()}, app / "models" / "general.pt")
    general.load_state_dict({k: v.half().float() if v.is_floating_point() else v for k, v in general.state_dict().items()})
    general.eval().to(device)
    gfeat = {s: general_features(general, x, device) for s, (x, _) in sets.items()}
    lw = LedoitWolf().fit(gfeat["train"])
    scores = {s: lw.mahalanobis(f) for s, f in gfeat.items()}
    wall_scores = lw.mahalanobis(general_features(general, load_images(wallpapers), device))
    ood_t = float(np.quantile(scores["val"], OOD_KEEP))
    np.savez(app / "models" / "ood.npz", mean=lw.location_.astype(np.float32),
             precision=lw.precision_.astype(np.float32), threshold=np.float32(ood_t))

    comparison = json.loads((ROOT / "app/models/meta.json").read_text())
    meta = {
        "id": "efficientnet_b0", "name": "EfficientNet-B0", "timm_id": T.MODELS["efficientnet_b0"][0],
        "classes": LABELS, "seed": best["seed"], "best_epoch": best["best_epoch"],
        "seeds_trained": len(infos), "val_macro_f1_by_seed": sorted(round(i["best_val_macro_f1"], 4) for i in infos),
        "file_mb": round((app / "models" / "model.pt").stat().st_size / 1e6, 1),
        "melanoma_threshold": mel_t, "target_melanoma_recall": TARGET_MEL_RECALL,
        "ood_threshold": ood_t, "ood_keep_on_validation": OOD_KEEP, "general_timm_id": GENERAL_ID,
        "test": metrics(*out["test"][:1], sets["test"][1], mel_t, scores["test"], ood_t),
        "external": metrics(out["external"][0], sets["external"][1], mel_t, scores["external"], ood_t),
        "photo_check": {
            "rejected_test_lesions": float((scores["test"][sets["test"][1] != HEALTHY] > ood_t).mean()),
            "rejected_test_healthy": float((scores["test"][sets["test"][1] == HEALTHY] > ood_t).mean()),
            "rejected_external": float((scores["external"] > ood_t).mean()),
            "rejected_non_skin": float((wall_scores > ood_t).mean()), "non_skin_images": len(wallpapers),
            "non_skin_note": "built-in Windows wallpapers and images (landscapes, abstract art)",
        },
        "examples": {**comparison.get("examples", {})},
        "trained_on": "HAM10000 (7,009 lesion photos + 2,744 healthy-skin patches, split by lesion)",
    }

    # Lesion samples: keep each existing one if this model gets it right and the photo check
    # accepts it; otherwise pick another test photo of that type that does.
    pt, yt = out["test"][0], sets["test"][1]
    test_ids = pd.read_csv(ROOT / "data/processed/splits.csv").query("split == 'test'").image_id.to_numpy()
    good = (pt.argmax(1) == yt) & (scores["test"] <= ood_t)
    for c, code in enumerate(CLASSES):
        current = meta["examples"].get(code)
        idx = np.where(test_ids == current)[0]
        if len(idx) and good[idx[0]]:
            continue
        cand = np.where(good[:len(test_ids)] & (yt[:len(test_ids)] == c))[0]
        pick = cand[np.argsort(-pt[cand, c])[len(cand) // 4]]
        Image.open(ROOT / "data/raw/HAM10000_images" / f"{test_ids[pick]}.jpg").save(app / "static/examples" / f"{code}.jpg")
        meta["examples"][code] = str(test_ids[pick])
        print(f"sample {code}: replaced with {test_ids[pick]}")

    # Healthy-skin sample: a confidently, correctly classified test patch
    ok = np.where((yt == HEALTHY) & (pt.argmax(1) == HEALTHY) & (scores["test"] <= ood_t))[0]
    pick = ok[np.argsort(-pt[ok, HEALTHY])[len(ok) // 10]]  # confident but not the single most extreme
    Image.fromarray(np.asarray(sets["test"][0][pick])).save(app / "static/examples/healthy.jpg", quality=92)
    meta["examples"]["healthy"] = f"healthy-test-{int(pick)}"

    meta = json.loads(json.dumps(meta, default=float))
    (app / "models" / "meta.json").write_text(json.dumps(meta, indent=1))
    write_readme_numbers(args.site / "README.md", meta)
    confusion_figures(out["test"][0], yt, app / "static" / "figures")
    for k in ["test", "external"]:
        print(k, {n: round(v, 3) for n, v in meta[k].items() if not isinstance(v, dict)})
    print("photo check", {n: (round(v, 3) if isinstance(v, float) else v) for n, v in meta["photo_check"].items()})
    print("seed", best["seed"], "mel_t", round(mel_t, 4), "ood_t", round(ood_t, 1))


if __name__ == "__main__":
    main()
