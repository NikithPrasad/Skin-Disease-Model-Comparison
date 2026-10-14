"""Clinical-photo pipeline: prepare the 24-group dataset, then train EfficientNet-B0 on it
(the model chosen in the HAM10000 comparison; pass --models to train others too).
Finished runs are skipped, so it is safe to stop and restart.

Usage: python src/clinical_run_all.py [--seeds 0 1] [--models efficientnet_b0 resnet50 ...]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed_clinical"
RUNS = ROOT / "results" / "runs_clinical"
MODELS = ["efficientnet_b0", "resnet50", "mobilenetv3", "cnn"]
NUM_CLASSES = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=["efficientnet_b0"], choices=MODELS)
    args = ap.parse_args()
    if not (DATA / "test_y.npy").exists():
        subprocess.run([sys.executable, str(SRC / "clinical_prepare.py")], check=True)
    for seed in args.seeds:
        for model in args.models:
            if (RUNS / f"{model}_seed{seed}" / "info.json").exists():
                print(f"skip {model} seed {seed} (done)", flush=True)
                continue
            print(f"== training {model} seed {seed} ==", flush=True)
            subprocess.run([sys.executable, str(SRC / "train.py"), "--model", model, "--seed", str(seed),
                            "--data-dir", str(DATA), "--runs-dir", str(RUNS),
                            "--num-classes", str(NUM_CLASSES), "--clinical"], check=True)


if __name__ == "__main__":
    main()
