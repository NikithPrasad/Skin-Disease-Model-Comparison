"""The whole pipeline in one command:

  1. prepare the data (skipped if already done)
  2. train every model with every seed (finished runs are skipped, so it is safe to stop and restart)
  3. build the comparison tables and figures
  4. export the best models and figures to the website

Usage: python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODELS = ["cnn", "resnet50", "efficientnet_b0", "mobilenetv3"]
SEEDS = [0, 1, 2]


def run(script, *args):
    subprocess.run([sys.executable, str(SRC / script), *args], check=True)


def main():
    if not (ROOT / "data" / "processed" / "external_y.npy").exists():
        print("== preparing data ==", flush=True)
        run("prepare_data.py")

    for seed in SEEDS:  # one full model comparison per seed, so partial results are usable early
        for model in MODELS:
            if (ROOT / "results" / "runs" / f"{model}_seed{seed}" / "info.json").exists():
                print(f"skip {model} seed {seed} (done)", flush=True)
                continue
            print(f"== training {model} seed {seed} ==", flush=True)
            run("train.py", "--model", model, "--seed", str(seed))

    print("== building report ==", flush=True)
    run("report.py")
    print("== updating website ==", flush=True)
    run("export_models.py")


if __name__ == "__main__":
    main()
