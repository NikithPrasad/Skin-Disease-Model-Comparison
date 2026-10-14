"""Results table for the models trained on the clinical-photo dataset (24 groups).

Reads results/runs_clinical/<model>_seed<k>/ and writes results/CLINICAL_RESULTS.md and
results/clinical_comparison.csv; also refreshes the table in README.md between
<!-- clinical:start --> and <!-- clinical:end -->.

Usage: python src/clinical_report.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, top_k_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs_clinical"
DATA = ROOT / "data" / "processed_clinical"
NAMES = {"cnn": "CNN (from scratch)", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0", "mobilenetv3": "MobileNetV3"}


def main():
    cls = json.loads((DATA / "classes.json").read_text())
    keys = cls["keys"]
    mel = keys.index("melanoma")
    rows, per_class = [], {}
    for run in sorted(RUNS.glob("*_seed*")):
        if not (run / "info.json").exists():
            continue
        info = json.loads((run / "info.json").read_text())
        p, y = np.load(run / "test_probs.npy"), np.load(run / "test_labels.npy")
        pred = p.argmax(1)
        rows.append({
            "model": info["model"], "seed": info["seed"],
            "macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "top3": top_k_accuracy_score(y, p, k=3, labels=range(len(keys))),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "melanoma_recall": recall_score(y, pred, labels=[mel], average="macro"),
            "params_m": info["params_millions"], "gflops": info["gflops"],
            "cpu_ms": info["latency_ms_cpu_batch1"], "train_min": info["train_time_s"] / 60,
        })
        per_class.setdefault(info["model"], []).append(
            [recall_score(y, pred, labels=[c], average="macro") for c in range(len(keys))])
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "clinical_comparison.csv", index=False)

    g = df.groupby("model")
    summary = g.mean(numeric_only=True)
    sd = g.std(numeric_only=True).fillna(0)
    n = g.size()
    summary = summary.sort_values("macro_f1", ascending=False)
    fmt = lambda m, c, d=3: f"{summary.loc[m, c]:.{d}f}" + (f" ± {sd.loc[m, c]:.{d}f}" if n[m] > 1 else "")
    lines = ["| Model | Macro F1 | Accuracy | Top-3 accuracy | Balanced acc. | Melanoma recall | Params (M) | CPU ms / image | Train time (min) | Seeds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in summary.index:
        lines.append(f"| {NAMES[m]} | {fmt(m, 'macro_f1')} | {summary.loc[m, 'accuracy']:.3f} | {summary.loc[m, 'top3']:.3f} | "
                     f"{summary.loc[m, 'balanced_acc']:.3f} | {summary.loc[m, 'melanoma_recall']:.3f} | {summary.loc[m, 'params_m']:.1f} | "
                     f"{summary.loc[m, 'cpu_ms']:.1f} | {summary.loc[m, 'train_min']:.0f} | {n[m]} |")
    table = "\n".join(lines)

    order = list(summary.index)
    pc = ["| Group | " + " | ".join(NAMES[m] for m in order) + " |", "|---|" + "---|" * len(order)]
    for c, k in enumerate(keys):
        pc.append(f"| {cls['names'][k]} | " + " | ".join(f"{np.mean([r[c] for r in per_class[m]]):.2f}" for m in order) + " |")

    counts = {s: len(np.load(DATA / f"{s}_y.npy")) for s in ["train", "val", "test"]}
    md = f"""# Clinical-photo results (24 groups)

Same training recipe as the HAM10000 comparison, but on ordinary clinical
photos: {counts['train']:,} training / {counts['val']:,} validation / {counts['test']:,} test photos in 23 condition
groups + healthy skin, merged from DermNet, Fitzpatrick17k, a dermatology atlas (incl. leprosy), a
skin-infection set and two healthy-skin sets (see `src/clinical_prepare.py`). Duplicate photos were removed and
photos of the same patient never cross splits. Augmentation keeps photos upright (rotations within ±20°).

{table}

Test set, mean over seeds. Top-3 accuracy: the right group is among the model's three highest answers.

## Recall by group

{chr(10).join(pc)}
"""
    (ROOT / "results" / "CLINICAL_RESULTS.md").write_text(md, encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "<!-- clinical:start -->" in text:
        s, e = text.index("<!-- clinical:start -->"), text.index("<!-- clinical:end -->")
        readme.write_text(text[:s] + "<!-- clinical:start -->\n" + table + "\n" + text[e:], encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
