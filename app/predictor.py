"""Loads the four trained models and classifies an image held in memory.

Images arrive as bytes, are decoded in memory and discarded after prediction; nothing
here writes to disk.
"""
import base64
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
    "akiec": ("Actinic keratosis", "A rough, scaly patch caused by years of sun exposure. It is common and treatable, and worth having looked at because it can change over time."),
    "bcc": ("Basal cell carcinoma", "A common, slow-growing type of skin cancer. It rarely spreads and is very treatable."),
    "bkl": ("Benign keratosis", "A common, harmless skin growth, such as a sun spot or a raised spot that appears with age."),
    "df": ("Dermatofibroma", "A common, harmless firm bump in the skin, often on the legs."),
    "mel": ("Melanoma", "A type of skin cancer that starts in the cells that give skin its colour. When found early it is very treatable."),
    "nv": ("Melanocytic nevus", "An ordinary mole. Most people have many, and they are usually harmless."),
    "vasc": ("Vascular lesion", "A small growth made of blood vessels, such as a cherry spot. Usually harmless."),
}
SERIOUS = {"mel", "bcc", "akiec"}

CONSIDER = "Consider discussing this result with a qualified healthcare professional"
MONITOR = "This result appears less concerning, but changes in a skin lesion should still be monitored"

# Plain-language guidance per type. General information only (no medicines named), in line with
# common public-health advice. The AI never decides for the user whether they need a doctor.
#   level      "urgent" / "doctor": worth discussing with a professional; "selfcare": appears less concerning
#   headline   the main "what to do next" message
#   looks      what this condition typically looks like (general signs, not read from the photo)
#   treatment  how it is usually managed
#   steps      what the person can do now
GUIDANCE = {
    "mel": {"level": "urgent", "headline": CONSIDER,
            "looks": "uneven colours (brown, black, sometimes blue-grey or red), an irregular edge and a lopsided shape",
            "treatment": "If a doctor confirms it, it is usually removed with a small operation. Found early, this is very effective.",
            "steps": ["Book an appointment with a GP or dermatologist and show them this spot. It is a good idea to arrange this soon rather than waiting.",
                      "Take a clear photo now, so you can show the doctor and notice any changes.",
                      "Please don't try to remove or treat it at home.",
                      "Protect it from the sun and use SPF 30+ sunscreen."]},
    "bcc": {"level": "doctor", "headline": CONSIDER,
            "looks": "a shiny, pearly or pink bump, sometimes with tiny visible blood vessels or a small sore in the middle",
            "treatment": "If a doctor confirms it, it is usually removed with a minor procedure or treated with a prescribed cream. Once treated it rarely comes back.",
            "steps": ["Mention it to a GP or dermatologist at your next convenient appointment.",
                      "Avoid picking at it or using home removal products.",
                      "Protect your skin from the sun: SPF 30+, a hat and shade around midday."]},
    "akiec": {"level": "doctor", "headline": CONSIDER,
              "looks": "a rough, dry, scaly patch, pink or red, on skin that gets a lot of sun",
              "treatment": "Most patches clear with a simple treatment from a doctor, such as freezing, a prescribed cream or light therapy.",
              "steps": ["Mention it to a GP or dermatologist at your next convenient appointment.",
                        "Avoid scratching or picking at the scaly surface.",
                        "Use SPF 30+ sunscreen every day to help prevent new patches."]},
    "nv": {"level": "selfcare", "headline": MONITOR,
           "looks": "an evenly coloured brown spot with a smooth, regular edge",
           "treatment": "Ordinary moles usually need no treatment. A doctor can remove one if it bothers you.",
           "steps": ["Check it once a month using the ABCDE guide: Asymmetry, uneven Border, several Colours, Diameter over 6 mm, or Evolving (changing).",
                     "Take a photo now, so you can compare it later.",
                     "Use sunscreen and avoid sunburn and tanning beds."]},
    "bkl": {"level": "selfcare", "headline": MONITOR,
            "looks": "a waxy, 'stuck-on' looking light or dark brown spot, often with a slightly rough surface",
            "treatment": "Usually no treatment is needed. If it itches or catches on clothing, a doctor can remove it quickly.",
            "steps": ["Try not to pick or scratch it, as it can bleed or get irritated.",
                      "If it itches, a fragrance-free moisturiser can help.",
                      "Use sunscreen to stop sun spots from getting darker."]},
    "df": {"level": "selfcare", "headline": MONITOR,
           "looks": "a small firm brown or pink bump, often with a paler centre, that dimples when pinched",
           "treatment": "Usually no treatment is needed; it often stays the same for years. A doctor can remove it if it is painful.",
           "steps": ["Take care when shaving over it, as nicking it can make it sore.",
                     "Keep an eye on its size and colour over time."]},
    "vasc": {"level": "selfcare", "headline": MONITOR,
             "looks": "a bright red, purple or dark red spot made of tiny blood vessels",
             "treatment": "Usually no treatment is needed. A doctor can remove it with a laser or by freezing if it bleeds often or you would like it gone.",
             "steps": ["If it bleeds after a knock, press on it gently with a clean cloth for 10 minutes.",
                       "Keep an eye on its size over time."]},
}

# Shown with every result: signs worth discussing with a doctor whatever the AI says
URGENT_SIGNS = [
    "It bleeds, oozes or crusts without being injured.",
    "It grows, or changes shape or colour, over a few weeks.",
    "It is a sore that hasn't healed after 3 to 4 weeks.",
    "It is new, looks different from your other spots, or becomes painful or itchy.",
]


class InvalidImage(Exception):
    pass


def load_image(image_bytes):
    """Decode in memory and resize to 256x256, as in training."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                return im.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC)
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise InvalidImage(str(e)) from None


def to_input(img):
    """256x256 RGB -> 224x224 antialiased -> ImageNet normalise (as in training)."""
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN) / STD


def preprocess(image_bytes):
    return to_input(load_image(image_bytes))


def attention_image(img, cam):
    """Photo with the areas the model ignored dimmed, as an in-memory JPEG data URL."""
    weight = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    w = np.asarray(weight, dtype=np.float32)[..., None] / 255
    shown = np.asarray(img, dtype=np.float32) * (0.18 + 0.82 * w)  # dim, never black out, the rest
    buf = io.BytesIO()
    Image.fromarray(np.uint8(shown.clip(0, 255))).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Predictor:
    def __init__(self, device=torch.device("cpu"), models_dir=APP / "models"):
        self.device = device
        self.meta = json.loads((models_dir / "meta.json").read_text())
        self.models = {}
        for m in self.meta["models"]:
            model = build_model(m["id"], pretrained=False)
            state = torch.load(models_dir / f"{m['id']}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
            self.models[m["id"]] = model.eval().to(device).requires_grad_(False)
        self.lock = threading.Lock()  # one prediction at a time keeps memory use flat
        with torch.no_grad():  # warm-up so the first real request is not slow
            dummy = torch.zeros(1, 3, IMG, IMG, device=device)
            for model in self.models.values():
                model(dummy)

    def predict(self, image_bytes):
        img = load_image(image_bytes)
        x = to_input(img).to(self.device)
        results = []
        with self.lock:
            for m in self.meta["models"]:
                model = self.models[m["id"]]
                t = time.perf_counter()
                # Grad-CAM (see the detection site): gradient of the top class score with
                # respect to the last feature map; only the classifier head needs a backward pass.
                body, head = split(model)
                with torch.no_grad():
                    feats = body(x)
                feats.requires_grad_(True)
                logits = head(feats)
                probs = logits.softmax(1)[0].detach().tolist()
                top = max(range(len(CLASSES)), key=lambda i: probs[i])
                logits[0, top].backward()
                cam = F.relu((feats.grad.mean((2, 3), keepdim=True) * feats).sum(1))[0].detach()
                cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else np.ones(cam.shape, np.float32)
                ms = (time.perf_counter() - t) * 1000
                order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])  # 7 classes: sorting is trivial
                results.append({
                    "model": m["id"], "name": m["name"], "ms": round(ms, 1), "top": CLASSES[order[0]],
                    "probs": [{"code": CLASSES[i], "p": round(probs[i], 4)} for i in order],
                    "attention": attention_image(img, cam),
                })
        del x
        return results


def split(model):
    """(feature extractor, classifier head) for Grad-CAM: timm models and the custom CNN."""
    if hasattr(model, "forward_features"):
        return model.forward_features, model.forward_head
    return model.features, model.head
