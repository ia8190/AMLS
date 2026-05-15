import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# cifar-10 class names
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# cnn model used for cifar-10
class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # feature layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # forward pass
    def forward(self, x):
        return self.classifier(self.features(x))


# load a saved model checkpoint
def load_model(device, model_path):
    model = CIFAR_CNN().to(device)
    ckpt = torch.load(model_path, map_location=device)

    # support both checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# create an fgsm adversarial image
def fgsm_attack(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)

    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()

    return x_adv.detach()


def main(folder="images", threshold=50.0, eps_255=0.0, model_path=None):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = Path(__file__).resolve().parents[2]

    # use default model if none is given
    if model_path is None:
        model_path = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    else:
        model_path = Path(model_path)

    # get folder path
    FOLDER_PATH = (ROOT / folder) if not Path(folder).is_absolute() else Path(folder)

    # check files and folders exist
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if not FOLDER_PATH.exists():
        raise FileNotFoundError(f"Folder not found: {FOLDER_PATH}")

    EPS = eps_255 / 255

    print(f"Using folder: {FOLDER_PATH}")
    print(f"Using epsilon: {eps_255}/255")
    print(f"Using model: {model_path}")

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # image transform for the model
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = load_model(device, model_path)

    # supported image types
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # get all images in the folder
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
        # load image
        img = Image.open(p).convert("RGB")

        # prepare image for model
        x = transform(img).unsqueeze(0).to(device)

        # clean prediction
        with torch.no_grad():
            clean_probs = torch.softmax(model(x), dim=1)[0]
            clean_conf, clean_pred = clean_probs.max(dim=0)

        # apply fgsm attack if epsilon is greater than zero
        if EPS > 0:
            y = torch.tensor([clean_pred.item()]).to(device)
            x_eval = fgsm_attack(model, x, y, EPS)
        else:
            x_eval = x

        # prediction after optional attack
        with torch.no_grad():
            probs = torch.softmax(model(x_eval), dim=1)[0]
            conf, pred = probs.max(dim=0)

        conf_pct = conf.item() * 100
        total += 1

        # hide label if confidence is low
        if conf_pct < threshold:
            label = "N/A"
        else:
            label = classes[pred.item()]
            accepted += 1

        clean_label = classes[clean_pred.item()]
        clean_conf_pct = clean_conf.item() * 100

        # print result
        if EPS > 0:
            print(
                f"{Path(p).name:30s} | "
                f"clean: {clean_label:10s} ({clean_conf_pct:5.1f}%) | "
                f"after FGSM: {label:10s} ({conf_pct:5.1f}%)"
            )
        else:
            print(f"{Path(p).name:30s} -> {label:10s} ({conf_pct:5.1f}%)")

    # show how many predictions passed the threshold
    coverage = accepted / total * 100
    print(f"\nThreshold: {threshold:.1f}% | Coverage: {coverage:.1f}% ({accepted}/{total})")


# run from the command line
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    eps_255 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    main(
        folder=folder,
        threshold=50.0,
        eps_255=eps_255,
        model_path=model_path
    )