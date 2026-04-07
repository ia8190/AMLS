from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]


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


def top3(probs):
    top3_conf, top3_idx = torch.topk(probs, 3)
    return " | ".join([f"{classes[i]} {c*100:.1f}%" for c, i in zip(top3_conf.tolist(), top3_idx.tolist())])


def fgsm_attack(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    model.eval()
    x_orig = x.detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        x_adv = x_orig.clone()

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.detach()

    return x_adv


def load_model(device, ckpt_path: Path):
    model = CIFAR_CNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = Path(__file__).resolve().parents[3]

    # ---- choose defense checkpoint (FGSM adversarially trained) ----
    CKPT_PATH = ROOT / "checkpoints" / "cnn_cifar10_fgsmadv_best.pt"

    # ---- choose image ----
    IMAGE_PATH = ROOT / "images" / "birds.jpg"

    # ---- choose attack ----
    ATTACK = "fgsm"  # "fgsm" or "pgd"
    EPS = 8 / 255

    # PGD params (only used if ATTACK="pgd")
    STEPS = 20
    ALPHA = 1 / 255
    RANDOM_START = True

    # CIFAR-10 norm (must match training)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    to_model = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    model = load_model(device, CKPT_PATH)

    img = Image.open(IMAGE_PATH).convert("RGB")
    x = to_model(img).unsqueeze(0).to(device)

    # 1. clean prediction
    with torch.no_grad():
        clean_probs = torch.softmax(model(x), dim=1)[0]
        clean_conf, clean_pred = clean_probs.max(dim=0)

    print("\n=== DEFENDED MODEL (FGSM-ADV TRAINED) ===")
    print(f"Checkpoint: {CKPT_PATH.name}")
    print(f"Image: {IMAGE_PATH.name}")
    print(f"CLEAN  -> {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
    print(f"Top-3  -> {top3(clean_probs)}")

    # choose label y for untargeted demo
    # (for external images we don't have a true label)
    y = torch.tensor([clean_pred.item()]).to(device)

    # 2. attack then predict again
    if ATTACK.lower() == "fgsm":
        x_adv = fgsm_attack(model, x, y, EPS)
        attack_desc = f"FGSM eps={EPS:.5f}"
    elif ATTACK.lower() == "pgd":
        x_adv = pgd_attack(model, x, y, EPS, ALPHA, STEPS, RANDOM_START)
        attack_desc = f"PGD eps={EPS:.5f}, alpha={ALPHA:.5f}, steps={STEPS}, rs={RANDOM_START}"
    else:
        raise ValueError("ATTACK must be 'fgsm' or 'pgd'")

    with torch.no_grad():
        adv_probs = torch.softmax(model(x_adv), dim=1)[0]
        adv_conf, adv_pred = adv_probs.max(dim=0)

    print(f"\nATTACK > {attack_desc}")
    print(f"AFTER  > {classes[adv_pred.item()]} ({adv_conf.item()*100:.2f}%)")
    print(f"Top-3  > {top3(adv_probs)}")

    flipped = (adv_pred.item() != clean_pred.item())
    print(f"\nLabel flipped? {'YES' if flipped else 'NO'}")


if __name__ == "__main__":
    main()
