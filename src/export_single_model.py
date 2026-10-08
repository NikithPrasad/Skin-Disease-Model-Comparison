"""Export one trained model as a standalone detection website (the Skin-Disease-Detection repo).

Writes, into <site>/app/: the model weights (float16), models/meta.json with its scores and a
melanoma warning threshold, and confusion-matrix figures (light + dark).

The melanoma threshold is chosen on the VALIDATION set: the highest melanoma probability
cut-off that still flags at least 80% of validation melanomas. Test-set numbers are then
reported for that fixed threshold, so they are an honest estimate.

Usage: python src/export_single_model.py --model efficientnet_b0 --site ../Skin-Disease-Detection
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

import report
import train as T
from models import CLASSES, build_model

ROOT = Path(__file__).resolve().parent.parent
MEL, NV = CLASSES.index("mel"), CLASSES.index("nv")
TARGET_MEL_RECALL = 0.80
NAMES = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50", "mobilenetv3": "MobileNetV3", "cnn": "CNN"}


def scores(p, y, threshold):
    pred, flag = p.argmax(1), p[:, MEL] >= threshold
    return {
        "macro_f1": f1_score(y, pred, average="macro"), "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred), "macro_auc": roc_auc_score(y, p, multi_class="ovr"),
        "melanoma_recall_top_answer": recall_score(y, pred, labels=[MEL], average="macro"),
        "melanoma_recall_with_warning": flag[y == MEL].mean(),
        "warning_on_non_melanoma": flag[y != MEL].mean(),
        "per_class_recall": dict(zip(CLASSES, recall_score(y, pred, average=None, labels=range(len(CLASSES))))),
    }


def confusion_figures(p, y, out_dir):
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES))).astype(float)
    norm = cm / cm.sum(1, keepdims=True)
    for theme in report.THEMES:
        report.set_theme(theme)
        fig, ax = plt.subplots(figsize=(5.6, 5))
        ax.imshow(norm, cmap=report.BLUES, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=report.text_on(norm[i, j]))
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        suffix = "" if theme == "light" else "_dark"
        fig.savefig(out_dir / f"confusion{suffix}.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--site", type=Path, default=ROOT.parent / "Skin-Disease-Detection")
    args = ap.parse_args()

    comparison = json.loads((ROOT / "app" / "models" / "meta.json").read_text())
    entry = next(m for m in comparison["models"] if m["id"] == args.model)
    weights = ROOT / "app" / "models" / f"{args.model}.pt"  # best-validation seed, float16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.to(device)

    probs = {}
    for split in ["val", "test", "external"]:
        x, y = T.load_split(split)
        probs[split] = (T.predict(model, x, device), y.numpy())

    pv, yv = probs["val"]
    mel_p = np.sort(pv[yv == MEL, MEL])[::-1]
    threshold = float(next(c for c in mel_p if (pv[yv == MEL, MEL] >= c).mean() >= TARGET_MEL_RECALL))

    app = args.site / "app"
    (app / "models").mkdir(parents=True, exist_ok=True)
    (app / "static" / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, app / "models" / "model.pt")
    meta = {
        "id": args.model, "name": NAMES[args.model], "timm_id": T.MODELS[args.model][0],
        "seed": entry["seed"], "params_millions": entry["params_millions"], "cpu_latency_ms": entry["cpu_latency_ms"],
        "file_mb": round((app / "models" / "model.pt").stat().st_size / 1e6, 1),
        "melanoma_threshold": threshold, "target_melanoma_recall": TARGET_MEL_RECALL,
        "test": scores(*probs["test"], threshold), "external": scores(*probs["external"], threshold),
        "examples": comparison.get("examples", {}),
        "trained_on": "HAM10000 (7,009 training images, split by lesion)",
    }
    meta = json.loads(json.dumps(meta, default=float))  # numpy -> plain floats
    (app / "models" / "meta.json").write_text(json.dumps(meta, indent=1))
    confusion_figures(*probs["test"], app / "static" / "figures")
    print(json.dumps({k: meta[k] for k in ["name", "seed", "melanoma_threshold"]}, indent=1))
    print("test:", {k: round(v, 3) for k, v in meta["test"].items() if not isinstance(v, dict)})
    print("external:", {k: round(v, 3) for k, v in meta["external"].items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
