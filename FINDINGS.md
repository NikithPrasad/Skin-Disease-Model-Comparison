# Findings: which model is best, and was ChatGPT right?

All numbers come from real training runs on this laptop (RTX 4060): 4 models × 3 random
seeds, 30 epochs each, evaluated on 2,004 held-out HAM10000 images (split by lesion) and
1,511 external ISIC 2018 test images. Full tables: [results/RESULTS.md](results/RESULTS.md).

## The result

| Model | Macro F1 (internal) | Macro F1 (external) | Accuracy | Melanoma recall | CPU ms / image | File size |
|---|---|---|---|---|---|---|
| **ResNet50** | **0.718 ± 0.027** | **0.724 ± 0.014** | **0.858** | 0.474 | 44.7 | 47 MB |
| **EfficientNet-B0** | 0.700 ± 0.036 | 0.695 ± 0.013 | 0.836 | **0.580** | 28.1 | 8 MB |
| MobileNetV3 | 0.672 ± 0.011 | 0.677 ± 0.004 | 0.833 | 0.518 | 17.2 | 9 MB |
| CNN (from scratch) | 0.483 ± 0.031 | 0.462 ± 0.077 | 0.738 | 0.482 | 6.3 | 5 MB |
| *Always answer "mole"* | *0.115* | *0.107* | *0.669* | *0.000* | | |

(File sizes are the half-precision copies used by the website.)

**In one sentence:** ResNet50 and EfficientNet-B0 are the best two and effectively tied;
ResNet50 has slightly higher F1 and accuracy, EfficientNet-B0 catches more melanomas with
6× fewer parameters; MobileNetV3 is the best choice for a phone; the CNN trained from scratch
is far behind, which shows the value of transfer learning.

Why "tied": ResNet50 beat EfficientNet-B0 on the external set in 3 of 3 seeds but on the
internal test set in only 2 of 3, and a bootstrap test puts the 95% confidence interval of the
gap at [−0.010, +0.071] macro F1. That interval includes zero, so the difference isn't
reliable at this sample size.

## Verdict on ChatGPT's plan

| ChatGPT said | Verdict |
|---|---|
| Compare CNN, ResNet50, EfficientNet-B0, MobileNetV3 | **Right.** A sensible, standard set covering baseline / classic / efficient / mobile. |
| Use accuracy, precision, recall, macro F1, confusion matrix, per-class results, training & inference time, model size | **Right**, and complete. |
| Handle class imbalance rather than reporting accuracy alone | **Right, and it matters a lot:** answering "mole" for every image scores 67% accuracy but 0.115 macro F1. |
| Download the "Original Format ZIP" | **Right.** |
| *(not mentioned)* Split by lesion, not by image | **Missed — the biggest gap.** Measured here: with a naive split by image, 36% of test photos show a lesion already seen in training. MobileNetV3's test score jumps from 0.679 to 0.735 macro F1, but on the external set it is *no better* (0.655 vs 0.679). The higher score was memorisation. |
| *(not mentioned)* Use several training runs | **Missed.** With one run the ranking flips: on seed 0 EfficientNet-B0 came 3rd on the test set; on seed 2 it came 1st. |
| *(not mentioned)* Test on outside data | **Missed.** The external ISIC 2018 set (it was in the same ZIP) is the best evidence the models generalise. |
| "I can do the actual model comparison" | **Doubtful.** Its sandbox has no GPU and limited time; training these models took about 1.5 hours on a dedicated GPU. |

**Overall:** ChatGPT's plan was a good starting point. The model choice and metrics were
correct, but it left out three methodology steps that decide whether the numbers can be
trusted. My prediction before training ("EfficientNet-B0 or ResNet50 wins, MobileNetV3 close,
CNN last") turned out right; the extra finding was that the top two are too close to separate.

## Honest limitations (good to mention — teachers like this)

- **Melanoma is the weak spot.** Every model calls 26–36% of melanomas "mole" (see the
  confusion matrices). Best melanoma recall is 0.58 (EfficientNet-B0). A real screening tool
  would need higher sensitivity, for example by lowering the decision threshold for melanoma.
- Rare classes have few test images (only 23 dermatofibromas), so their per-class scores are noisy.
- One training recipe for all models; per-model tuning might change the order slightly.
- The leakage experiment used one model and one seed.
- This is a student project, not a medical device.

## What to tell your ma'am (talking points)

1. "We compared four architectures fairly: same data, same augmentation, same loss, same 30 epochs, 3 seeds each."
2. "We split by lesion, because HAM10000 has 10,015 photos of only 7,470 lesions. We measured
   what happens if you don't: the score inflates by 0.056 while real performance doesn't improve."
3. "We used macro F1 because 67% of images are moles; accuracy alone is misleading."
4. "ResNet50 and EfficientNet-B0 tie for best (≈0.70–0.72 macro F1); MobileNetV3 is 2.6× faster
   and 5.5× smaller for 0.046 less F1, so it's the pick for a mobile app; a CNN from scratch
   reaches only 0.48 — transfer learning matters."
5. "Our models were also tested on 1,511 images from a different collection (ISIC 2018), and the scores held up."
6. "Biggest weakness: melanoma recall is only ~0.5–0.6."

## About the "DSA / time complexity" question

In deep learning almost all the time goes into the network's arithmetic (billions of
multiply-adds per image), not into the surrounding code, so DSA-style tricks matter much
less than in competitive programming. The big wins were: decoding each JPEG once instead of
every epoch (avoids ~3.6 million decodes), keeping images in RAM, doing augmentation on the
GPU in batches, and mixed precision. The website uses indexed lookups (O(log n)) for users and
sessions. See "Performance notes" in the [README](README.md) for the full list with numbers.
