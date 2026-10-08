"""Aggregate all runs in results/runs into tables and figures.

Outputs:
  results/summary.csv            one row per model, mean and std over seeds
  results/per_run.csv            one row per (model, seed, test set)
  results/per_class_f1.csv       per-class F1 on the internal test set
  results/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis",
    "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion",
}
MODEL_ORDER = ["cnn", "resnet50", "efficientnet_b0", "mobilenetv3"]
LABELS = {"cnn": "CNN (scratch)", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
          "mobilenetv3": "MobileNetV3"}
# Two themes so the website can show charts that match light or dark mode.
# Categorical slots 1-4 of the reference palette (fixed order), stepped per theme.
THEMES = {
    "light": dict(colors={"cnn": "#2a78d6", "resnet50": "#eb6834", "efficientnet_b0": "#1baf7a", "mobilenetv3": "#eda100"},
                  ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb",
                  ramp=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(colors={"cnn": "#3987e5", "resnet50": "#d95926", "efficientnet_b0": "#199e70", "mobilenetv3": "#c98500"},
                 ink="#f4f4f2", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", surface="#1a1a19",
                 ramp=["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]),
}
SUFFIX = ""


def set_theme(name):
    """Point the module-level colours (read by every figure function) at one theme."""
    global COLORS, INK, INK2, MUTED, GRID, AXIS, SURFACE, BLUES, SUFFIX
    t = THEMES[name]
    COLORS, INK, INK2, MUTED, GRID, AXIS, SURFACE = (t["colors"], t["ink"], t["ink2"], t["muted"],
                                                      t["grid"], t["axis"], t["surface"])
    BLUES = LinearSegmentedColormap.from_list("blues", t["ramp"])
    SUFFIX = "" if name == "light" else "_dark"
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
        "legend.labelcolor": INK2, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, fname):
    fig.savefig(FIG / fname.replace(".png", f"{SUFFIX}.png"))
    plt.close(fig)


set_theme("light")


def text_on(v):
    """Light or dark text, whichever reads better on the heatmap colour for value v."""
    r, g, b, _ = BLUES(v)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#f4f4f2" if lum < 0.5 else "#0b0b0b"


METRICS = ["macro_f1", "balanced_acc", "accuracy", "macro_precision", "macro_recall",
           "weighted_f1", "macro_auc", "melanoma_recall"]
METRIC_LABELS = {
    "macro_f1": "Macro F1", "balanced_acc": "Balanced accuracy", "accuracy": "Accuracy",
    "macro_precision": "Macro precision", "macro_recall": "Macro recall", "weighted_f1": "Weighted F1",
    "macro_auc": "Macro ROC-AUC", "melanoma_recall": "Melanoma recall",
}


def score(y, probs):
    pred = probs.argmax(1)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_auc": roc_auc_score(y, probs, multi_class="ovr", average="macro"),
        "melanoma_recall": recall_score(y, pred, labels=[CLASSES.index("mel")], average="macro"),
    }


def load_runs():
    rows, per_class, cms, histories, infos = [], [], {}, {}, {}
    for run in sorted(RUNS.iterdir()):
        info_file = run / "info.json"
        if not info_file.exists():
            continue
        info = json.loads(info_file.read_text())
        m, seed = info["model"], info["seed"]
        if m not in MODEL_ORDER or seed >= 90:  # seeds >= 90 are smoke tests
            continue
        infos.setdefault(m, []).append(info)
        histories.setdefault(m, []).append(json.loads((run / "history.json").read_text()))
        for split in ["test", "external"]:
            probs = np.load(run / f"{split}_probs.npy")
            y = np.load(run / f"{split}_labels.npy")
            rows.append({"model": m, "seed": seed, "test_set": split, **score(y, probs)})
            if split == "test":
                f1s = f1_score(y, probs.argmax(1), average=None, labels=range(len(CLASSES)))
                per_class.append({"model": m, "seed": seed, **dict(zip(CLASSES, f1s))})
            cm = confusion_matrix(y, probs.argmax(1), labels=range(len(CLASSES)))
            cms[(m, split)] = cms.get((m, split), 0) + cm
    return pd.DataFrame(rows), pd.DataFrame(per_class), cms, histories, infos


def naive_baseline():
    """What you get by always predicting the most common class (nv)."""
    out = {}
    for split in ["test", "external"]:
        y = np.load(ROOT / "data" / "processed" / f"{split}_y.npy")
        probs = np.zeros((len(y), len(CLASSES)))
        probs[:, CLASSES.index("nv")] = 1
        out[split] = score(y, probs)
    return out


def summarize(df, infos):
    out = []
    for m in MODEL_ORDER:
        if m not in infos:
            continue
        row = {"model": LABELS[m], "seeds": len(infos[m])}
        for split, tag in [("test", "test"), ("external", "ext")]:
            part = df[(df.model == m) & (df.test_set == split)]
            for k in METRICS:
                row[f"{tag}_{k}_mean"] = part[k].mean()
                row[f"{tag}_{k}_std"] = part[k].std(ddof=1) if len(part) > 1 else 0.0
        inf = pd.DataFrame(infos[m])
        for k in ["params_millions", "gflops", "size_mb", "train_time_s", "best_epoch",
                  "latency_ms_gpu_batch1", "latency_ms_cpu_batch1", "throughput_img_per_s_gpu_batch64"]:
            row[k] = inf[k].mean()
        out.append(row)
    return pd.DataFrame(out)


def bar_metrics(df, split, title, fname, naive):
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    metrics = ["macro_f1", "balanced_acc", "accuracy", "macro_auc"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        part = df[(df.model == m) & (df.test_set == split)]
        means, stds = part[metrics].mean().values, part[metrics].std(ddof=1).fillna(0).values
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, means, width * 0.92, yerr=stds,
                      color=COLORS[m], label=LABELS[m], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        for b, v, sd in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, v + sd + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    for j, k in enumerate(metrics):
        ax.hlines(naive[split][k], j - 0.42, j + 0.42, colors=MUTED, linestyles="--", linewidth=1.2)
    ax.plot([], [], color=MUTED, linestyle="--", label='"Always say nv" baseline')
    ax.set_xticks(x, [METRIC_LABELS[k] for k in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (mean over seeds, bars = ±1 std)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.1), frameon=False, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, fname)


def confusion_grid(cms, split, fname, title):
    models = [m for m in MODEL_ORDER if (m, split) in cms]
    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 4.3))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = cms[(m, split)].astype(float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color=text_on(norm[i, j]))
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABELS[m], loc="left")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title + " (row-normalised: diagonal = recall per class, summed over seeds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, fname)


def per_class_heatmap(pc):
    models = [m for m in MODEL_ORDER if m in set(pc.model)]
    mat = np.array([pc[pc.model == m][CLASSES].mean().values for m in models])
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(models) + 1.6))
    ax.imshow(mat, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=text_on(mat[i, j]))
    ax.set_xticks(range(len(CLASSES)), [f"{c}\n{CLASS_NAMES[c]}" for c in CLASSES], fontsize=8)
    ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
    ax.set_title("Per-class F1 on the internal test set (mean over seeds)", loc="left")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "per_class_f1.png")


def training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for m in MODEL_ORDER:
        if m not in histories:
            continue
        for ax, key in zip(axes, ["val_macro_f1", "train_loss"]):
            curves = np.array([[r[key] for r in h] for h in histories[m]])
            mean = curves.mean(0)
            ep = np.arange(1, len(mean) + 1)
            ax.plot(ep, mean, color=COLORS[m], linewidth=2, label=LABELS[m])
            if len(curves) > 1:
                ax.fill_between(ep, curves.min(0), curves.max(0), color=COLORS[m], alpha=0.15, linewidth=0)
    axes[0].set_title("Validation macro F1 per epoch", loc="left")
    axes[1].set_title("Training loss per epoch", loc="left")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "training_curves.png")


def efficiency_tradeoff(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = {"CNN (scratch)": "cnn", "ResNet50": "resnet50", "EfficientNet-B0": "efficientnet_b0",
            "MobileNetV3": "mobilenetv3"}
    for ax, (xcol, xlabel) in zip(axes, [("size_mb", "Model file size, float32 (MB)"),
                                         ("latency_ms_cpu_batch1", "CPU latency per image (ms)")]):
        for _, r in summary.iterrows():
            m = keys[r["model"]]
            ax.errorbar(r[xcol], r["test_macro_f1_mean"], yerr=r["test_macro_f1_std"], fmt="o",
                        color=COLORS[m], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                        ecolor=COLORS[m], elinewidth=1.2)
            ax.annotate(r["model"], (r[xcol], r["test_macro_f1_mean"]), xytext=(8, 6),
                        textcoords="offset points", fontsize=8.5, color=INK2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Internal test macro F1")
        ax.margins(x=0.25, y=0.25)
    axes[0].set_title("Accuracy vs. size  (up-left is better)", loc="left")
    axes[1].set_title("Accuracy vs. speed  (up-left is better)", loc="left")
    fig.tight_layout()
    save(fig, "efficiency_tradeoff.png")


def write_markdown(summary, pc, naive):
    """results/RESULTS.md: the tables in readable form, regenerated on every run."""
    def ms(r, tag, k):
        sd = f" ± {r[f'{tag}_{k}_std']:.3f}" if r["seeds"] > 1 else ""
        return f"{r[f'{tag}_{k}_mean']:.3f}{sd}"

    lines = ["# Results", "",
             f"Mean ± standard deviation over {int(summary['seeds'].min())} training seed(s) per model. "
             "Generated by `src/report.py`.", ""]
    for tag, title, key in [("test", "Internal test set (2,004 HAM10000 images, split by lesion)", "test"),
                            ("ext", "External test set (1,511 ISIC 2018 test images)", "external")]:
        lines += [f"## {title}", "", "| Model | Macro F1 | Balanced acc. | Accuracy | Macro AUC | Melanoma recall |",
                  "|---|---|---|---|---|---|"]
        for _, r in summary.iterrows():
            lines.append(f"| {r['model']} | " + " | ".join(ms(r, tag, k) for k in
                         ["macro_f1", "balanced_acc", "accuracy", "macro_auc", "melanoma_recall"]) + " |")
        nb = naive[key]
        lines += [f"| *Always answer \"nv\"* | {nb['macro_f1']:.3f} | {nb['balanced_acc']:.3f} | {nb['accuracy']:.3f} | "
                  f"{nb['macro_auc']:.3f} | {nb['melanoma_recall']:.3f} |", ""]
    lines += ["## Size and speed", "",
              "| Model | Params (M) | GFLOPs | File (MB, fp32) | Train time (min) | GPU ms/img | CPU ms/img | Best epoch |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {r['params_millions']:.2f} | {r['gflops']:.2f} | {r['size_mb']:.1f} | "
                     f"{r['train_time_s'] / 60:.1f} | {r['latency_ms_gpu_batch1']:.1f} | {r['latency_ms_cpu_batch1']:.1f} | "
                     f"{r['best_epoch']:.0f} |")
    lines += ["", "## Per-class F1 (internal test set)", "",
              "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for m in MODEL_ORDER:
        if m in set(pc.model):
            lines.append(f"| {LABELS[m]} | " + " | ".join(f"{v:.2f}" for v in pc[pc.model == m][CLASSES].mean()) + " |")
    lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in
              ["metrics_internal.png", "metrics_external.png", "per_class_f1.png", "confusion_internal.png",
               "confusion_external.png", "efficiency_tradeoff.png", "training_curves.png"]]
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df, pc, cms, histories, infos = load_runs()
    naive = naive_baseline()
    summary = summarize(df, infos)

    df.to_csv(ROOT / "results" / "per_run.csv", index=False)
    pc.to_csv(ROOT / "results" / "per_class_f1.csv", index=False)
    summary.to_csv(ROOT / "results" / "summary.csv", index=False)
    (ROOT / "results" / "naive_baseline.json").write_text(json.dumps(naive, indent=1))
    write_markdown(summary, pc, naive)

    for theme in THEMES:
        set_theme(theme)
        bar_metrics(df, "test", "Internal test set (HAM10000, split by lesion)", "metrics_internal.png", naive)
        bar_metrics(df, "external", "External test set (official ISIC 2018 test images)", "metrics_external.png", naive)
        confusion_grid(cms, "test", "confusion_internal.png", "Confusion matrices, internal test set")
        confusion_grid(cms, "external", "confusion_external.png", "Confusion matrices, external test set")
        per_class_heatmap(pc)
        training_curves(histories)
        efficiency_tradeoff(summary)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    cols = ["model", "seeds"] + [f"{t}_{k}_mean" for t in ["test", "ext"] for k in ["macro_f1", "balanced_acc", "accuracy", "melanoma_recall"]]
    print(summary[cols].round(3).to_string(index=False))
    print(summary[["model", "params_millions", "gflops", "size_mb", "train_time_s", "best_epoch",
                   "latency_ms_gpu_batch1", "latency_ms_cpu_batch1"]].round(2).to_string(index=False))
    print("naive baseline:", {s: {k: round(v, 3) for k, v in d.items()} for s, d in naive.items()})


if __name__ == "__main__":
    main()
