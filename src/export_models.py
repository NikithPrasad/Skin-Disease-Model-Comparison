"""Copy the trained models and comparison results into app/ so the website can run on
any laptop (CPU only, no internet, no API key).

For each architecture the seed with the best *validation* macro F1 is exported (choosing
by test score would be cheating). Weights are stored in float16 to halve the file size;
the app converts them back to float32 when loading.

Run after src/run_all.py:  python src/export_models.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
APP = ROOT / "app"
DISPLAY = {"cnn": "CNN (from scratch)", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
           "mobilenetv3": "MobileNetV3"}
SUMMARY_NAMES = {"cnn": "CNN (scratch)", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
                 "mobilenetv3": "MobileNetV3"}


def export_examples(meta, rng_seed=7):
    """One sample image per class for the website, taken from the internal test set (never
    trained on). Picks randomly among images that most of the exported models classify
    correctly, so the demo shows typical behaviour; the website says how they were chosen."""
    test = pd.read_csv(ROOT / "data" / "processed" / "splits.csv").query("split == 'test'").reset_index(drop=True)
    labels = np.load(ROOT / "data" / "processed" / "test_y.npy")
    correct = sum((np.load(RUNS / f"{m['id']}_seed{m['seed']}" / "test_probs.npy").argmax(1) == labels).astype(int)
                  for m in meta["models"])
    out_dir = APP / "static" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for label, code in enumerate(CLASSES):
        pool = test[(labels == label) & (correct >= 3)]
        if pool.empty:  # fall back to the best-classified images of this class
            pool = test[(labels == label) & (correct == correct[labels == label].max())]
        row = pool.sample(1, random_state=rng_seed).iloc[0]
        shutil.copy2(ROOT / "data" / "raw" / "HAM10000_images" / f"{row.image_id}.jpg", out_dir / f"{code}.jpg")
        chosen[code] = row.image_id
    print("examples:", chosen)
    return chosen


def main():
    (APP / "models").mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "results" / "summary.csv").set_index("model")
    meta = {"models": []}
    for name in DISPLAY:
        infos = [json.loads(p.read_text()) for p in RUNS.glob(f"{name}_seed*/info.json")
                 if json.loads(p.read_text())["seed"] < 90]
        best = max(infos, key=lambda i: i["best_val_macro_f1"])
        state = torch.load(RUNS / f"{name}_seed{best['seed']}" / "best.pt", map_location="cpu", weights_only=True)
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
        out = APP / "models" / f"{name}.pt"
        torch.save(state, out)

        s = summary.loc[SUMMARY_NAMES[name]]
        meta["models"].append({
            "id": name,
            "name": DISPLAY[name],
            "seed": best["seed"],
            "file_mb": round(out.stat().st_size / 1e6, 1),
            "params_millions": round(best["params_millions"], 2),
            "gflops": round(best["gflops"], 2),
            "cpu_latency_ms": round(float(s["latency_ms_cpu_batch1"]), 1),
            "test_macro_f1": [round(float(s["test_macro_f1_mean"]), 3), round(float(s["test_macro_f1_std"]), 3)],
            "test_balanced_acc": [round(float(s["test_balanced_acc_mean"]), 3), round(float(s["test_balanced_acc_std"]), 3)],
            "test_accuracy": [round(float(s["test_accuracy_mean"]), 3), round(float(s["test_accuracy_std"]), 3)],
            "test_melanoma_recall": [round(float(s["test_melanoma_recall_mean"]), 3), round(float(s["test_melanoma_recall_std"]), 3)],
            "ext_macro_f1": [round(float(s["ext_macro_f1_mean"]), 3), round(float(s["ext_macro_f1_std"]), 3)],
            "ext_accuracy": [round(float(s["ext_accuracy_mean"]), 3), round(float(s["ext_accuracy_std"]), 3)],
        })
        print(f"exported {name} (seed {best['seed']}): {out.stat().st_size / 1e6:.1f} MB")

    meta["seeds"] = int(summary["seeds"].min())
    meta["examples"] = export_examples(meta)
    leak = ROOT / "results" / "leakage_demo" / "summary.json"
    if leak.exists():
        meta["leakage_demo"] = json.loads(leak.read_text())
    (APP / "models" / "meta.json").write_text(json.dumps(meta, indent=1))

    fig_dst = APP / "static" / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for png in (ROOT / "results" / "figures").glob("*.png"):
        shutil.copy2(png, fig_dst / png.name)
    print("copied figures to", fig_dst)


if __name__ == "__main__":
    main()
