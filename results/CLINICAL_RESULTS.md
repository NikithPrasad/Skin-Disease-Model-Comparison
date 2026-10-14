# Clinical-photo results (24 groups)

Same training recipe as the HAM10000 comparison, but on ordinary clinical
photos: 24,792 training / 5,310 validation / 5,308 test photos in 23 condition
groups + healthy skin, merged from DermNet, Fitzpatrick17k, a dermatology atlas (incl. leprosy), a
skin-infection set and two healthy-skin sets (see `src/clinical_prepare.py`). Duplicate photos were removed and
photos of the same patient never cross splits. Augmentation keeps photos upright (rotations within ±20°).

| Model | Macro F1 | Accuracy | Top-3 accuracy | Balanced acc. | Melanoma recall | Params (M) | CPU ms / image | Train time (min) | Seeds |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.629 | 0.649 | 0.794 | 0.647 | 0.625 | 4.0 | 20.9 | 49 | 1 |

Test set, mean over seeds. Top-3 accuracy: the right group is among the model's three highest answers.

## Recall by group

| Group | EfficientNet-B0 |
|---|---|
| Acne, rosacea and clogged pores | 0.80 |
| Bacterial skin infection | 0.68 |
| Blistering skin disease | 0.51 |
| Eczema and dermatitis | 0.60 |
| Fungal infection (ringworm) | 0.66 |
| Granulomatous skin disease | 0.46 |
| Hair loss | 0.81 |
| Healthy skin | 0.99 |
| Herpes, shingles and chickenpox | 0.74 |
| Leprosy (Hansen's disease) | 0.84 |
| Sun-related and pigment changes | 0.43 |
| Lupus and connective-tissue disease | 0.38 |
| Melanoma | 0.62 |
| Mole or birthmark | 0.70 |
| Nail condition | 0.93 |
| Psoriasis, lichen planus and similar | 0.57 |
| Scabies, bites and infestations | 0.57 |
| Skin cancer or pre-cancer | 0.70 |
| Benign growth, cyst or scar | 0.67 |
| Nutritional or systemic disease signs | 0.45 |
| Hives (urticaria) | 0.60 |
| Vascular lesion | 0.63 |
| Drug reaction, viral rash or vasculitis | 0.48 |
| Warts and molluscum | 0.68 |
