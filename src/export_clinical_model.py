"""Export the clinical-photo EfficientNet-B0 (23 condition groups + healthy skin) to the SkinCheck site.

Writes into <site>/app/:
  models/model.pt     weights of the seed with the best validation macro F1 (float16)
  models/general.pt   general ImageNet-pretrained EfficientNet-B0 (float16), used only by the photo check
  models/ood.npz      the "is this a photo of skin?" check
  models/meta.json    scores, thresholds and per-group results, all measured with the shipped weights
  static/figures/     confusion matrix (light + dark)
  static/examples/    one test photo per group, for the "try an example" buttons

Decisions are made on the VALIDATION set and only reported on the test set:
  * melanoma warning: the highest melanoma-probability cut-off that still flags >= 80% of
    validation melanomas;
  * photo check: a logistic regression on general ImageNet features (EfficientNet-B0, no skin
    training) that separates the training photos from 6,000 everyday non-skin photos (objects,
    animals, landscapes, streets: Kaggle "natural-images" without its person class, and "intel image
    classification"). The cut-off keeps 99.5% of validation skin photos. The first version used a
    Mahalanobis distance instead, but clinical photos vary so much (clothes, rooms, backgrounds of
    face portraits) that it let through 18% of non-skin images; see the photo-check table in the
    SkinCheck README. Tested on held-out non-skin photos and on the built-in Windows wallpapers,
    which are never used for fitting.

Usage: python src/export_clinical_model.py [--site ../Skin-Disease-Detection] [--model efficientnet_b0]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score, top_k_accuracy_score)

import report
import train as T
from clinical_prepare import load_one
from export_single_model import general_features
from models import build_model

DATA = ROOT / "data" / "processed_clinical"
RUNS = ROOT / "results" / "runs_clinical"
CLS = json.loads((DATA / "classes.json").read_text())
LABELS, NAMES = CLS["keys"], CLS["names"]
MEL, HEALTHY = LABELS.index("melanoma"), LABELS.index("healthy")
TARGET_MEL_RECALL, OOD_KEEP = 0.80, 0.995
GENERAL_ID = "efficientnet_b0.ra_in1k"
N_NEG_FIT, N_NEG_TEST = 6000, 1500


def non_skin_paths(datasets):
    paths = [p for p in (datasets / "natural-images/data/natural_images").rglob("*.jpg") if p.parent.name != "person"]
    for part in ["seg_train/seg_train", "seg_test/seg_test"]:
        paths += sorted((datasets / "intel-image-classification" / part).rglob("*.jpg"))
    paths = sorted(paths)
    np.random.default_rng(0).shuffle(paths)
    return paths[:N_NEG_FIT], paths[N_NEG_FIT:N_NEG_FIT + N_NEG_TEST]


def load_images(paths):
    return np.stack([load_one((str(p), False)) for p in paths])  # same resize + centre crop as training


@torch.no_grad()
def run(model, x_uint8, device, batch=128):
    """Probabilities, float32 (same numbers as the CPU website)."""
    probs = []
    for i in range(0, len(x_uint8), batch):
        x = T.normalize(T.to_input(torch.as_tensor(np.asarray(x_uint8[i:i + batch])), device))
        probs.append(model(x).softmax(1).cpu())
    return torch.cat(probs).numpy()


def metrics(p, y, mel_t, ood, ood_t, sources):
    pred = p.argmax(1)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "top3_accuracy": top_k_accuracy_score(y, p, k=3, labels=range(len(LABELS))),
        "macro_auc": roc_auc_score(y, p, multi_class="ovr", labels=range(len(LABELS))),
        "melanoma_recall_top_answer": recall_score(y, pred, labels=[MEL], average="macro"),
        "melanoma_recall_with_warning": (p[y == MEL, MEL] >= mel_t).mean(),
        "warning_on_non_melanoma": (p[y != MEL, MEL] >= mel_t).mean(),
        "conditions_called_healthy": (pred[y != HEALTHY] == HEALTHY).mean(),
        "healthy_called_condition": (pred[y == HEALTHY] != HEALTHY).mean(),
        "per_class_recall": {LABELS[c]: recall_score(y, pred, labels=[c], average="macro") for c in range(len(LABELS))},
        "per_source_accuracy": {s: accuracy_score(y[sources == s], pred[sources == s]) for s in sorted(set(sources))},
        "rejected_by_photo_check": (ood > ood_t).mean(),
        "n": len(y),
    }


def confusion_figures(p, y, out_dir):
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(LABELS))).astype(float)
    norm = cm / cm.sum(1, keepdims=True)
    short = [NAMES[k].split(",")[0].split(" (")[0] for k in LABELS]
    for theme in report.THEMES:
        report.set_theme(theme)
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(norm, cmap=report.BLUES, vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                if norm[i, j] >= 0.005:
                    ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7, color=report.text_on(norm[i, j]))
        ax.set_xticks(range(len(LABELS)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS)), short, fontsize=8)
        ax.set_xlabel("AI's answer")
        ax.set_ylabel("Confirmed answer (numbers are % of each row)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.savefig(out_dir / f"confusion{'' if theme == 'light' else '_dark'}.png")
        plt.close(fig)


def write_readme_numbers(readme, m):
    """Rewrite the README's accuracy section (between markers) from meta.json, so it never goes stale."""
    t, pc = m["test"], m["photo_check"]
    pct = lambda v: f"{v * 100:.1f}%" if 0 < v < 0.1 else f"{round(v * 100)}%"
    rows = "\n".join(f"| {m['class_names'][c]} | {pct(t['per_class_recall'][c])} |" for c in m["classes"])
    src_names = {"dermnet": "DermNet", "fitzpatrick17k": "Fitzpatrick17k", "atlas31": "Atlas (31 classes)",
                 "infections8": "Skin-infection set", "msld": "MSLD (healthy + rashes)", "skintypes": "Normal skin types"}
    srcs = "\n".join(f"| {src_names.get(k, k)} | {pct(v)} |" for k, v in t["per_source_accuracy"].items())
    block = f"""Measured on {t['n']:,} test photos the model never saw in training (no duplicate or same-patient
photo was in both), using exactly the weights shipped here (seed {m['seed']} of {m['seeds_trained']} training runs,
chosen by validation score). Dataset: {m['dataset']['train']:,} training / {m['dataset']['val']:,} validation / {m['dataset']['test']:,} test photos.

| | Test set |
|---|---|
| Accuracy (closest group is right) | {pct(t['accuracy'])} |
| Right group among the top 3 answers | {pct(t['top3_accuracy'])} |
| Macro F1 (every group counts equally) | {t['macro_f1']:.3f} |
| Balanced accuracy | {t['balanced_acc']:.3f} |
| Macro ROC-AUC | {t['macro_auc']:.3f} |
| Skin conditions wrongly called healthy skin | {pct(t['conditions_called_healthy'])} |
| Healthy skin wrongly given a condition | {pct(t['healthy_called_condition'])} |

Correct answers by group (recall):

| Group | Test set |
|---|---|
{rows}

Accuracy by source collection:

| Source | Accuracy |
|---|---|
{srcs}

### The melanoma warning

The top answer alone names {pct(t['melanoma_recall_top_answer'])} of melanomas. The site also flags a photo whenever
the melanoma probability is {m['melanoma_threshold'] * 100:.1f}% or more (threshold picked on validation data to catch
at least {pct(m['target_melanoma_recall'])} of validation melanomas): that catches {pct(t['melanoma_recall_with_warning'])} of test melanomas, and
{pct(t['warning_on_non_melanoma'])} of other photos also get the flag, to be safe.

### The photo check

Before any result, a general ImageNet-trained EfficientNet-B0 (it knows everyday objects and scenes)
describes the photo, and a small classifier trained on skin photos versus 6,000 everyday non-skin photos
decides whether it is a photo of skin. If not, the site says "this doesn't look like a photo of skin" and
gives no result. Threshold: keep {m['ood_keep_on_validation'] * 100:.1f}% of validation skin photos.

| | Turned away |
|---|---|
| Real test photos | {pct(pc['rejected_test_photos'])} |
| Healthy-face test photos | {pct(pc['rejected_test_healthy_faces'])} |
| Everyday non-skin photos held out from fitting ({pc['heldout_everyday_images']:,} objects, animals, landscapes, streets) | {pct(pc['rejected_heldout_everyday'])} |
| Non-skin images ({pc['non_skin_images']} {pc['non_skin_note']}) | {pct(pc['rejected_non_skin'])} |"""
    text = readme.read_text(encoding="utf-8")
    s, e = text.index("<!-- numbers:start -->"), text.index("<!-- numbers:end -->")
    readme.write_text(text[:s] + "<!-- numbers:start -->\n" + block + "\n" + text[e:], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=ROOT.parent / "Skin-Disease-Detection")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--datasets", type=Path, default=ROOT.parent / "datasets")
    args = ap.parse_args()
    app = args.site / "app"

    infos = [json.loads(p.read_text()) for p in RUNS.glob(f"{args.model}_seed*/info.json")]
    best = max(infos, key=lambda i: i["best_val_macro_f1"])
    state = torch.load(RUNS / f"{args.model}_seed{best['seed']}" / "best.pt", map_location="cpu", weights_only=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    torch.save(state, app / "models" / "model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=False, num_classes=len(LABELS))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.eval().to(device)

    sets = {s: (np.load(DATA / f"{s}_x.npy", mmap_mode="r"), np.load(DATA / f"{s}_y.npy")) for s in ["train", "val", "test"]}
    manifest = pd.read_csv(DATA / "manifest.csv")
    src = {s: manifest.query("split == @s").source.to_numpy() for s in ["val", "test"]}
    paths = manifest.query("split == 'test'").path.to_numpy()
    detail = manifest.query("split == 'test'").detail.to_numpy()
    out = {s: run(model, sets[s][0], device) for s in ["val", "test"]}

    pv, yv = out["val"], sets["val"][1]
    mel_p = np.sort(pv[yv == MEL, MEL])[::-1]
    mel_t = float(next(c for c in mel_p if (pv[yv == MEL, MEL] >= c).mean() >= TARGET_MEL_RECALL))

    general = timm.create_model(GENERAL_ID, pretrained=True, num_classes=0)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in general.state_dict().items()}, app / "models" / "general.pt")
    general.load_state_dict({k: v.half().float() if v.is_floating_point() else v for k, v in general.state_dict().items()})
    general.eval().to(device)
    gfeat = {s: general_features(general, x, device) for s, (x, _) in sets.items()}
    neg_fit, neg_test = non_skin_paths(args.datasets)
    nf_fit = general_features(general, load_images(neg_fit), device)
    nf_test = general_features(general, load_images(neg_test), device)
    mu, sd = gfeat["train"].mean(0), gfeat["train"].std(0) + 1e-6
    clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(
        (np.concatenate([gfeat["train"], nf_fit]) - mu) / sd, np.r_[np.ones(len(gfeat["train"])), np.zeros(len(nf_fit))])
    w, b = clf.coef_[0] / sd, float(clf.intercept_[0] - (clf.coef_[0] / sd) @ mu)  # folded: score = f @ w + b
    # score = how NOT like a skin photo (negative logit), so higher = reject, as before
    scores = {s: -(f @ w + b) for s, f in gfeat.items()}
    ood_t = float(np.quantile(scores["val"], OOD_KEEP))
    np.savez(app / "models" / "ood.npz", weight=w.astype(np.float32), bias=np.float32(b), threshold=np.float32(ood_t))
    wallpapers = sorted(p for p in Path("C:/Windows/Web").rglob("*.jpg") if p.stat().st_size > 50_000)
    wall_scores = -(general_features(general, load_images(wallpapers), device) @ w + b)
    neg_scores = -(nf_test @ w + b)

    yt, pt = sets["test"][1], out["test"]
    meta = {
        "id": args.model, "name": "EfficientNet-B0", "timm_id": T.MODELS[args.model][0],
        "classes": LABELS, "class_names": NAMES, "seed": best["seed"], "best_epoch": best["best_epoch"],
        "seeds_trained": len(infos), "val_macro_f1_by_seed": sorted(round(i["best_val_macro_f1"], 4) for i in infos),
        "file_mb": round((app / "models" / "model.pt").stat().st_size / 1e6, 1),
        "melanoma_threshold": mel_t, "target_melanoma_recall": TARGET_MEL_RECALL,
        "ood_threshold": ood_t, "ood_keep_on_validation": OOD_KEEP, "general_timm_id": GENERAL_ID,
        "test": metrics(pt, yt, mel_t, scores["test"], ood_t, src["test"]),
        "photo_check": {
            "rejected_test_photos": float((scores["test"] > ood_t).mean()),
            "rejected_test_healthy_faces": float((scores["test"][src["test"] == "skintypes"] > ood_t).mean()),
            "rejected_non_skin": float((wall_scores > ood_t).mean()), "non_skin_images": len(wallpapers),
            "non_skin_note": "built-in Windows wallpapers and images (landscapes, abstract art), never used for fitting",
            "rejected_heldout_everyday": float((neg_scores > ood_t).mean()), "heldout_everyday_images": len(neg_test),
        },
        "dataset": {s: int(len(sets[s][1])) for s in sets},
        "examples": {},
    }

    # One example per group: a confidently and correctly classified test photo that passes the
    # photo check (the 25th-percentile confidence among those, so not the single easiest one)
    ex_dir = app / "static" / "examples"
    for old in ex_dir.glob("*.jpg"):
        old.unlink()
    # Some groups have a preferred source. DermNet's herpes category also holds genital (STD) photos,
    # and the healthy faces are stock portraits of real people, so those examples come from close-ups.
    # The atlas set mixes in ISIC dermoscopy (magnified) images for moles, melanoma, cancers, benign
    # and vascular lesions; examples should look like ordinary photos, so those come from camera-photo sets.
    camera = {"dermnet", "fitzpatrick17k"}
    prefer = {"herpes": {"infections8", "atlas31"}, "healthy": {"msld"},
              **{k: camera for k in ["mole", "melanoma", "cancer", "benign", "vascular"]}}
    good = (pt.argmax(1) == yt) & (scores["test"] <= ood_t)
    for c, key in enumerate(LABELS):
        cand = np.where(good & (yt == c))[0]
        if key in prefer and np.isin(src["test"][cand], list(prefer[key])).any():
            cand = cand[np.isin(src["test"][cand], list(prefer[key]))]
        if key == "mole":  # a typical mole, not a faint birthmark
            typical = detail[cand] == "nevocytic nevus"
            cand = cand[typical] if typical.any() else cand
        pick = cand[np.argsort(-pt[cand, c])[len(cand) // 4]]
        Image.fromarray(np.asarray(sets["test"][0][pick])).save(ex_dir / f"{key}.jpg", quality=92)
        meta["examples"][key] = Path(paths[pick]).stem[:16]

    old = app / "models" / "meta.json"  # keep the Gemini evaluation (src/gemini_eval.py) across re-exports
    if old.exists():
        prev = json.loads(old.read_text())
        meta.update({k: prev[k] for k in ["gemini", "accuracy_by_confidence"] if k in prev})
    meta = json.loads(json.dumps(meta, default=float))
    (app / "models" / "meta.json").write_text(json.dumps(meta, indent=1))
    confusion_figures(pt, yt, app / "static" / "figures")
    write_readme_numbers(args.site / "README.md", meta)
    print({n: round(v, 3) for n, v in meta["test"].items() if isinstance(v, float)})
    print("per source", {k: round(v, 3) for k, v in meta["test"]["per_source_accuracy"].items()})
    print("per class", {k: round(v, 2) for k, v in meta["test"]["per_class_recall"].items()})
    print("photo check", meta["photo_check"], "seed", best["seed"], "mel_t", round(mel_t, 4), "ood_t", round(ood_t, 1))


if __name__ == "__main__":
    main()
