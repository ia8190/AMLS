import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main(folder="images", threshold=50.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # project root: scripts/visual -> scripts -> root
    ROOT = Path(__file__).resolve().parents[2]

    CKPT_PATH = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    FOLDER_PATH = (ROOT / folder) if not Path(folder).is_absolute() else Path(folder)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    if not FOLDER_PATH.exists():
        raise FileNotFoundError(f"Folder not found: {FOLDER_PATH}")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = CIFAR_CNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        str(FOLDER_PATH / f)
        for f in os.listdir(FOLDER_PATH)
        if os.path.splitext(f.lower())[1] in exts
    ]

    if not paths:
        print(f"No images found in: {FOLDER_PATH}")
        return

    accepted = 0
    total = 0

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            conf, pred = probs.max(dim=0)

        conf_pct = conf.item() * 100
        total += 1

        if conf_pct < threshold:
            label = "N/A"
        else:
            label = classes[pred.item()]
            accepted += 1

        print(f"{Path(p).name:30s} -> {label:10s} ({conf_pct:5.1f}%)")

    coverage = accepted / total * 100
    print(f"\nThreshold: {threshold:.1f}% | Coverage: {coverage:.1f}% ({accepted}/{total})")


if __name__ == "__main__":
    # folder is relative to project root (ADV_ML_SYS/images)
    main(folder="images", threshold=50.0)
