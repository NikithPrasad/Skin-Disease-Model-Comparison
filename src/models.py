"""Model definitions shared by training (src/train.py) and the web app (app/server.py)."""
import timm
import torch.nn as nn

NUM_CLASSES = 7
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

MODELS = {
    # name: (timm id or None for the custom CNN, learning rate)
    "cnn": (None, 2e-3),
    "resnet50": ("resnet50.tv2_in1k", 5e-4),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k", 5e-4),
    "mobilenetv3": ("mobilenetv3_large_100.ra_in1k", 5e-4),
}


class SimpleCNN(nn.Module):
    """Baseline CNN trained from scratch: strided stem -> 4 conv blocks -> global average pool
    -> classifier. About 2.5M parameters and ~1.2 GFLOPs, i.e. far cheaper than ResNet50."""

    def __init__(self, num_classes=NUM_CLASSES, widths=(64, 128, 256, 256)):
        super().__init__()
        c_in = 32
        layers = [nn.Conv2d(3, c_in, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c_in), nn.ReLU(inplace=True)]
        for c_out in widths:
            layers += [
                nn.MaxPool2d(2),
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, pretrained=True, num_classes=NUM_CLASSES):
    timm_id, _ = MODELS[name]
    if timm_id is None:
        return SimpleCNN(num_classes)
    return timm.create_model(timm_id, pretrained=pretrained, num_classes=num_classes)
