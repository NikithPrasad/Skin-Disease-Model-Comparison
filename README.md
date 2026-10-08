# Skin Disease Detection — Model Comparison on HAM10000

Compares four image classifiers on the HAM10000 dermoscopy dataset (10,015 images, 7 lesion classes):

| Model | Type | Why it is in the comparison |
|---|---|---|
| CNN (scratch) | Custom CNN (strided stem + 4 conv blocks), 2.4M params, trained from random weights | Baseline: what you get without transfer learning |
| ResNet50 | ImageNet-pretrained, fine-tuned | Classic, widely used transfer-learning backbone |
| EfficientNet-B0 | ImageNet-pretrained, fine-tuned | Modern, accuracy-per-compute efficient |
| MobileNetV3-Large | ImageNet-pretrained, fine-tuned | Lightweight, built for phones/edge devices |

## Results (3 seeds per model)

| Model | Macro F1 (internal) | Macro F1 (external) | Accuracy | Melanoma recall | CPU ms / image |
|---|---|---|---|---|---|
| ResNet50 | 0.718 ± 0.027 | 0.724 ± 0.014 | 0.858 | 0.474 | 44.7 |
| EfficientNet-B0 | 0.700 ± 0.036 | 0.695 ± 0.013 | 0.836 | 0.580 | 28.1 |
| MobileNetV3 | 0.672 ± 0.011 | 0.677 ± 0.004 | 0.833 | 0.518 | 17.2 |
| CNN (from scratch) | 0.483 ± 0.031 | 0.462 ± 0.077 | 0.738 | 0.482 | 6.3 |

ResNet50 and EfficientNet-B0 are effectively tied for best; MobileNetV3 is the speed/size pick;
the scratch CNN shows how much transfer learning helps. Full tables and figures:
[results/RESULTS.md](results/RESULTS.md). Interpretation, limitations and a review of the
original plan: [FINDINGS.md](FINDINGS.md).

## Classes

| Code | Name | Images |
|---|---|---|
| akiec | Actinic keratosis / intraepithelial carcinoma | 327 |
| bcc | Basal cell carcinoma | 514 |
| bkl | Benign keratosis | 1,099 |
| df | Dermatofibroma | 115 |
| mel | Melanoma | 1,113 |
| nv | Melanocytic nevus (common mole) | 6,705 |
| vasc | Vascular lesion | 142 |

The dataset is heavily imbalanced: 67% of images are `nv`. A model that always answers `nv`
scores ~67% accuracy while being useless, so **macro F1** (every class counts equally) is
the main metric, with balanced accuracy, ROC-AUC and melanoma recall alongside plain accuracy.

## Method (what makes the comparison fair)

- **Split by lesion, not by image.** HAM10000 has 10,015 images of only 7,470 distinct
  lesions; many lesions were photographed several times. Splitting by image would put
  near-identical photos in train and test and inflate scores. All splits are grouped by
  `lesion_id` and stratified by class: 7,009 train / 1,002 validation / 2,004 test.
- **External test set.** Models are also evaluated on the official ISIC 2018 Task 3 test
  images (1,511 images, collected separately), which shows how well they generalise.
- **Identical training for every model:** 224×224 input, same augmentation (random
  rotation 0–360°, flips, zoom, shift, colour jitter), class-weighted cross-entropy with
  label smoothing, AdamW, cosine learning-rate schedule, 30 epochs, batch 64. The
  checkpoint with the best validation macro F1 is the one tested.
  Learning rate: 5e-4 for pretrained models, 2e-3 for the scratch CNN.
- **3 seeds per model.** Results are reported as mean ± standard deviation, so a lucky
  run cannot decide the ranking.
- **Efficiency:** parameters, GFLOPs, model file size, training time, single-image
  latency on GPU and CPU, GPU throughput.

## Website (runs on any laptop — no GPU, no API key)

`app/` is **SkinCheck Compare**, a local website: create an account, upload a lesion image, and all four
models give their indication side by side, plus the full comparison table and charts. The trained models
are included in `app/models/` (~70 MB total), so nothing needs training and nothing is sent to the
internet. On a normal laptop CPU, one image through all four models takes well under a second.

It shares the calm, accessible design of [SkinCheck](https://github.com/NikithPrasad/Skin-Disease-Detection)
(light theme, 18px text, WCAG AA contrast, large buttons, no images of skin conditions in the interface).
Each result says *"Your image most closely matches..."* (majority vote) with how many models agree, what
it means, what to do next, how it is usually managed and when to talk to a doctor. If any model sees
melanoma, or the models disagree, the advice is *"Consider discussing this result with a qualified
healthcare professional"*; otherwise *"This result appears less concerning, but changes in a skin lesion
should still be monitored"*. Optional, folded away: where each model looked (Grad-CAM) and each model's
full answer. Every result ends with *"This is an AI-based prediction, not a medical diagnosis."*

Unlike SkinCheck, these four 7-class models have no "healthy skin" answer and no check that the image is a
close-up of skin. Photos by Ato Aikins, Sarah Sheedy, Asal Davletyarovaasss and Nate Johnston on Unsplash.

**Windows (easiest):**
1. Install Python 3.10+ from python.org (tick "Add python.exe to PATH").
2. Double-click `setup.bat` once (downloads CPU-only PyTorch, ~200 MB; needs internet this one time).
3. Double-click `run.bat`. The website opens at http://localhost:8000.

**Mac / Linux:**
```
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r app/requirements.txt
python app/server.py
```

After setup the website works offline. For a presentation, run `setup.bat` on the
presentation laptop the day before rather than relying on the venue's Wi-Fi.

### Accounts and privacy

- **Photos are never stored.** An upload is read into memory, passed through the models and
  discarded when the request ends. It is never written to disk, never put in the database,
  never logged, and never sent anywhere. `tests/test_app.py` uploads a photo carrying a unique
  marker and then scans the app folder and the system temp folder to prove no copy was written
  (and that test was checked to fail when a leak is deliberately planted).
- **The database (`app/data/users.db`) holds only** usernames, salted scrypt password hashes and
  hashed login-session tokens. It has no column that could hold an image. It is excluded from git.
- **Security basics:** passwords hashed with scrypt (never stored in plain text); constant-time
  comparisons; 5 failed logins lock an account for 5 minutes; session cookies are `HttpOnly` +
  `SameSite=Strict` and expire after 7 days; every POST must carry a custom header (blocks
  cross-site request forgery); a strict Content-Security-Policy blocks injected scripts; the
  server only accepts connections from the same computer (127.0.0.1).
- Users can remove a photo from the screen, sign out, or delete their account (and its sessions)
  from the Privacy tab.

### Tests

```
python -m unittest discover -s tests -v
```
30 tests: password hashing, validation, lockout, sessions and expiry, account deletion, CSRF
protection, cookie flags, bad uploads, the no-photo-on-disk check, and the same server tests
against the real trained models.

## Layout

```
src/prepare_data.py   build lesion-grouped splits, cache resized images (data/processed)
src/train.py          train + evaluate one model:  python src/train.py --model resnet50 --seed 0
src/run_all.py        the whole pipeline: data -> train all models x seeds -> report -> website (resumable)
src/report.py         tables (results/*.csv, results/RESULTS.md) and figures (results/figures/*.png)
src/leakage_demo.py   what goes wrong with a split by image instead of by lesion
src/export_models.py  copy the best checkpoint per model (fp16) + figures into app/
src/models.py         model definitions shared by training and the website
results/runs/         per-run checkpoint, training history, predictions, timings (not in git)
app/server.py         website server (routes, cookies, security headers)
app/auth.py           accounts and sessions (SQLite)
app/predictor.py      loads the models, classifies an in-memory image
app/static/           the web page (index.html, styles.css, app.js), figures, sample images
app/models/           exported models (float16) + meta.json with their scores
tests/                automated tests
```

## Reproduce

1. Download the "Original Format ZIP" of HAM10000 from Harvard Dataverse
   (doi:10.7910/DVN/DBW86T) and extract it into `data/raw/` (the inner image ZIPs into
   `data/raw/HAM10000_images/` and `data/raw/ISIC2018_test/`).
2. `python src/run_all.py` — prepares the data, trains everything (about 1.5 h on an RTX 4060),
   builds the report and updates the website. Safe to stop and restart: finished runs are skipped.
3. Optional: `python src/leakage_demo.py` (~6 min) re-runs the split-by-image experiment.

Requires Python 3, PyTorch with CUDA, timm, scikit-learn, pandas, matplotlib.
Hardware used: RTX 4060 Laptop GPU (8 GB), i9-13900HX.

## Performance notes (where the time goes)

In deep learning almost all the time is spent on the network's arithmetic: ResNet50 needs about
8.2 billion floating-point operations per image, times 7,009 images, times 30 epochs, times ~3
for the backward pass. Clever data structures can't shrink that; what matters is keeping the GPU
busy and never doing the same work twice. What this project does:

| Where | What was done | Effect |
|---|---|---|
| Image loading | Decode + resize each JPEG **once** and keep all images in RAM as one uint8 array | Decoding costs ~10 ms per image. Doing it every epoch would be 10,015 × 30 epochs × 12 runs ≈ 3.6M decodes (~10 hours of CPU time); caching makes it 10,015 decodes (~1.5 minutes) |
| Batching | A batch is gathered by indexing the in-memory array: O(batch size), no disk access | The GPU never waits on the disk |
| Augmentation | Rotations, flips, zoom and colour changes run **on the GPU for the whole batch at once** (one `affine_grid` + `grid_sample`) instead of a Python loop per image | No CPU bottleneck |
| Arithmetic | bfloat16 mixed precision (tensor cores) | Measured on this GPU: convolutions 1.6× and matrix multiplies 5× faster than float32 |
| Measured, not assumed | `channels_last` memory layout is "supposed" to be faster; measured on this GPU it was 25% slower, so it was removed | |
| Website | Models loaded once at start-up, not per request | Each prediction only runs the networks |
| Website | Usernames and session tokens are looked up through SQLite indexes (B-trees): O(log n) per request | Fast with any number of users |
| Website | Failed-login tracking: dict of deques, expired entries dropped from the front — amortised O(1) | |
| Website | Passwords hashed with scrypt, deliberately slow (~70 ms) and memory-hard | The one place slowness is the point: it makes password guessing expensive |
