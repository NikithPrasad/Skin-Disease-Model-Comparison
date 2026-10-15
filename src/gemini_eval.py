"""Measure Gemini, the local model and their combination on the clinical-photo dataset.

Samples N photos per group from the validation set (used to choose how to combine the two) and N per
group from the test set (used only to report). Each photo (original file, longest side 768 px, JPEG) is
sent to Gemini through SkinCheck's app/gemini.py, which asks independently, without the model's answer.
Answers are cached in results/gemini_eval/answers.json, so the script can be stopped and resumed.

Combination rule: score = model probabilities + w * Gemini confidence on Gemini's group; the weight w
is the one with the best validation accuracy. Results go to results/gemini_eval/summary.json and are
copied into SkinCheck's meta.json by export (see --site).

Note: Gemini was trained on large amounts of web images and may have seen some of these public
photos before, so its score here can be optimistic.

Usage: python src/gemini_eval.py [--per-group 5] [--site ../Skin-Disease-Detection]
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_clinical"
RUN = ROOT / "results" / "runs_clinical" / "efficientnet_b0_seed0"
OUT = ROOT / "results" / "gemini_eval"
WEIGHTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0]


def jpeg(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--site", type=Path, default=ROOT.parent / "Skin-Disease-Detection")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, str(args.site / "app"))
    import gemini

    OUT.mkdir(parents=True, exist_ok=True)
    keys = json.loads((DATA / "classes.json").read_text())["keys"]
    man = pd.read_csv(DATA / "manifest.csv")
    probs = {}
    for split in ["val", "test"]:
        if split == "test":
            probs[split] = np.load(RUN / "test_probs.npy")
        else:  # validation probabilities from the shipped weights
            import torch
            sys.path.insert(0, str(ROOT / "src"))
            import train as T
            from models import build_model
            m = build_model("efficientnet_b0", pretrained=False, num_classes=len(keys))
            m.load_state_dict(torch.load(RUN / "best.pt", weights_only=True))
            probs[split] = T.predict(m.cuda().eval(), torch.from_numpy(np.load(DATA / "val_x.npy")), "cuda")

    rng = np.random.default_rng(0)
    picks = []
    for split in ["val", "test"]:
        part = man[man.split == split].reset_index(drop=True)
        for c, k in enumerate(keys):
            idx = np.where(part.y == c)[0]
            for i in rng.choice(idx, min(args.per_group, len(idx)), replace=False):
                picks.append({"split": split, "i": int(i), "y": c, "path": part.path[i], "source": part.source[i]})

    cache_file = OUT / f"answers_{gemini.MODEL}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    def run(p):
        key = p["path"]
        if key in cache:
            return
        for wait in [0, 20, 60]:
            time.sleep(wait)
            try:
                cache[key] = gemini.ask(jpeg(p["path"]))
                return
            except gemini.GeminiError as e:
                err = str(e)
        cache[key] = {"error": err}

    todo = [p for p in picks if p["path"] not in cache or "error" in cache[p["path"]]]
    for p in todo:
        cache.pop(p["path"], None)
    print(f"{len(picks)} photos, {len(todo)} to ask Gemini", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for n, _ in enumerate(ex.map(run, todo), 1):
            if n % 20 == 0:
                cache_file.write_text(json.dumps(cache, indent=0))
                print(f"  {n}/{len(todo)}", flush=True)
    cache_file.write_text(json.dumps(cache, indent=0))

    def scores(split, w):
        rows = [p for p in picks if p["split"] == split and "error" not in cache[p["path"]]]
        y = np.array([p["y"] for p in rows])
        pm = probs[split][[p["i"] for p in rows]]
        g = np.array([keys.index(cache[p["path"]]["group"]) if cache[p["path"]]["group"] in keys else -1 for p in rows])
        conf = np.array([cache[p["path"]]["confidence"] for p in rows])
        comb = pm.copy()
        ok = g >= 0
        comb[np.where(ok)[0], g[ok]] += w * conf[ok]
        top3 = np.argsort(-comb, 1)[:, :3]
        return {"n": len(rows), "model": float((pm.argmax(1) == y).mean()), "gemini": float((g == y).mean()),
                "combined": float((comb.argmax(1) == y).mean()),
                "combined_top3": float((top3 == y[:, None]).any(1).mean()),
                "model_top3": float((np.argsort(-pm, 1)[:, :3] == y[:, None]).any(1).mean()),
                "agree": float((pm.argmax(1) == g).mean()),
                "accuracy_when_agree": float((comb.argmax(1) == y)[pm.argmax(1) == g].mean()),
                "accuracy_when_differ": float((comb.argmax(1) == y)[pm.argmax(1) != g].mean()),
                "errors": sum("error" in cache[p["path"]] for p in picks if p["split"] == split)}

    val = {w: scores("val", w) for w in WEIGHTS}
    best_w = max(WEIGHTS, key=lambda w: (val[w]["combined"], -abs(w - 0.5)))
    test = scores("test", best_w)
    summary = {"gemini_model": gemini.MODEL, "per_group": args.per_group, "weight": best_w,
               "validation_by_weight": {str(w): round(v["combined"], 4) for w, v in val.items()},
               "validation": val[best_w], "test": test}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    meta_file = args.site / "app" / "models" / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["gemini"] = {"model": gemini.MODEL, "weight": best_w, "test": test, "per_group": args.per_group}
    # Without Gemini: how often the model's answer was right, by its top probability (validation)
    pv, yv = probs["val"], np.load(DATA / "val_y.npy")
    top = pv.max(1)
    bands = [(0.5, 1.01), (0.3, 0.5), (0.0, 0.3)]
    meta["accuracy_by_confidence"] = [{"from": lo, "to": min(hi, 1.0), "accuracy": float((pv.argmax(1) == yv)[(top >= lo) & (top < hi)].mean())}
                                      for lo, hi in bands]
    meta_file.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
