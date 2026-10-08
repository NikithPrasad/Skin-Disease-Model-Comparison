"""Loads the four trained models and classifies an image held in memory.

Images arrive as bytes, are decoded in memory and discarded after prediction; nothing
here writes to disk.
"""
import io
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "src"))
from models import CLASSES, build_model  # noqa: E402

CACHE_SIZE, IMG = 256, 224  # must match src/prepare_data.py and src/train.py
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
Image.MAX_IMAGE_PIXELS = 50_000_000  # refuse absurdly large images (decompression bombs)

CLASS_INFO = {
    "akiec": ("Actinic keratosis", "Rough, scaly patch from sun damage. Pre-cancerous and can turn into skin cancer."),
    "bcc": ("Basal cell carcinoma", "The most common skin cancer. Grows slowly and rarely spreads, but needs treatment."),
    "bkl": ("Benign keratosis", "Harmless growth such as a seborrheic keratosis or sun spot."),
    "df": ("Dermatofibroma", "Harmless firm bump in the skin, often on the legs."),
    "mel": ("Melanoma", "The most dangerous skin cancer. Early detection matters a lot."),
    "nv": ("Melanocytic nevus", "An ordinary mole. Almost always harmless."),
    "vasc": ("Vascular lesion", "Growth made of blood vessels, such as a cherry angioma. Usually harmless."),
}
SERIOUS = {"mel", "bcc", "akiec"}

# What to do next, per lesion type. General guidance only (no medicines named), in line with
# common public-health advice. level: "urgent" (see a doctor soon), "doctor" (book an
# appointment), "selfcare" (usually harmless; look after it and watch for changes).
GUIDANCE = {
    "mel": {"level": "urgent", "headline": "See a doctor soon, ideally within 2 weeks", "steps": [
        "Book an appointment with a GP or dermatologist and show them this lesion. Early treatment of melanoma works very well.",
        "Do not try to remove, cut, burn or treat it at home.",
        "Take a clear photo now so the doctor can see if it changes.",
        "Keep it out of the sun and use SPF 30+ sunscreen on exposed skin."]},
    "bcc": {"level": "doctor", "headline": "Book a doctor's appointment in the next few weeks", "steps": [
        "Basal cell carcinoma grows slowly and rarely spreads, but it does not go away on its own and needs treatment.",
        "A doctor can confirm it and remove it, usually with a minor procedure.",
        "Do not pick at it or try home removal products.",
        "Protect your skin from the sun: SPF 30+, a hat and shade around midday."]},
    "akiec": {"level": "doctor", "headline": "Book a doctor's appointment in the next few weeks", "steps": [
        "Actinic keratosis is caused by sun damage and can slowly turn into skin cancer, so it is worth treating.",
        "A doctor can treat it simply, for example by freezing it or with a prescribed cream.",
        "Do not scratch or pick at the scaly surface.",
        "Use SPF 30+ sunscreen every day on sun-exposed skin to prevent new patches."]},
    "nv": {"level": "selfcare", "headline": "Usually no treatment needed", "steps": [
        "Ordinary moles are harmless and don't need treatment.",
        "Check it once a month with the ABCDE rule: Asymmetry, uneven Border, more than one Colour, Diameter over 6 mm, or Evolving (changing).",
        "Take a photo now so you can compare it later.",
        "Use sunscreen and avoid sunburn and tanning beds, which raise the risk of new moles becoming melanoma."]},
    "bkl": {"level": "selfcare", "headline": "Usually no treatment needed", "steps": [
        "Benign keratoses (such as seborrheic keratoses and sun spots) are harmless and very common with age.",
        "Don't pick or scratch it; it can bleed or get irritated.",
        "If it itches, a fragrance-free moisturiser can help. A doctor can remove it if it bothers you.",
        "Use sunscreen to stop sun spots from darkening."]},
    "df": {"level": "selfcare", "headline": "Usually no treatment needed", "steps": [
        "Dermatofibromas are harmless firm bumps and often stay the same for years.",
        "Take care when shaving over it, as nicking it can make it sore.",
        "A doctor can remove it if it is painful or bothers you, although that leaves a small scar.",
        "Have it checked if it grows quickly or changes colour."]},
    "vasc": {"level": "selfcare", "headline": "Usually no treatment needed", "steps": [
        "Vascular lesions such as cherry angiomas are harmless clusters of blood vessels.",
        "If it bleeds after a knock, press on it with a clean cloth for 10 minutes.",
        "A doctor can remove it for cosmetic reasons or if it keeps bleeding.",
        "Have it checked if it grows quickly or bleeds without being injured."]},
}
# Shown with every result: signs that mean seeing a doctor whatever the model says
URGENT_SIGNS = [
    "It bleeds, oozes or crusts without being injured.",
    "It grows, or changes shape or colour, over a few weeks.",
    "It is a sore that hasn't healed after 3 to 4 weeks.",
    "It is new, looks different from your other spots, or becomes painful or itchy.",
]


class InvalidImage(Exception):
    pass


def preprocess(image_bytes):
    """Same pipeline as training: RGB -> 256x256 bicubic -> 224x224 antialiased -> ImageNet normalise."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                img = im.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC)
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise InvalidImage(str(e)) from None
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN) / STD


class Predictor:
    def __init__(self, device=torch.device("cpu"), models_dir=APP / "models"):
        self.device = device
        self.meta = json.loads((models_dir / "meta.json").read_text())
        self.models = {}
        for m in self.meta["models"]:
            model = build_model(m["id"], pretrained=False)
            state = torch.load(models_dir / f"{m['id']}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
            self.models[m["id"]] = model.eval().to(device)
        self.lock = threading.Lock()  # one prediction at a time keeps memory use flat
        with torch.no_grad():  # warm-up so the first real request is not slow
            dummy = torch.zeros(1, 3, IMG, IMG, device=device)
            for model in self.models.values():
                model(dummy)

    @torch.no_grad()
    def predict(self, image_bytes):
        x = preprocess(image_bytes).to(self.device)
        results = []
        with self.lock:
            for m in self.meta["models"]:
                t = time.perf_counter()
                probs = self.models[m["id"]](x).softmax(1)[0].tolist()
                ms = (time.perf_counter() - t) * 1000
                order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])  # 7 classes: sorting is trivial
                results.append({
                    "model": m["id"], "name": m["name"], "ms": round(ms, 1), "top": CLASSES[order[0]],
                    "probs": [{"code": CLASSES[i], "p": round(probs[i], 4)} for i in order],
                })
        del x
        return results
