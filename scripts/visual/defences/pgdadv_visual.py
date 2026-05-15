# scripts/visual/defences/pgdadv_visual.py

import sys
from pathlib import Path
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
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


def find_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / "checkpoints").exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found (no 'checkpoints' folder).")


def load_model(device, ckpt_path: Path) -> nn.Module:
    model = CIFAR_CNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    probs = torch.softmax(model(x), dim=1)[0]
    conf, pred = probs.max(dim=0)
    return probs, conf, pred


def denorm_to_01(x_norm, mean, std):
    mean_t = torch.tensor(mean, device=x_norm.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x_norm.device).view(1, 3, 1, 1)
    return (x_norm * std_t + mean_t).clamp(0, 1)


def pgd_attack_norm(
    model,
    x_norm,
    y,
    eps_px=8 / 255,
    alpha_px=2 / 255,
    steps=7,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    random_start=True,
):
    model.eval()
    device = x_norm.device

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    eps = eps_px / std_t
    alpha = alpha_px / std_t

    x_min = (0.0 - mean_t) / std_t
    x_max = (1.0 - mean_t) / std_t

    x_orig = x_norm.detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)
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
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)

        x_adv = x_adv.detach()

    return x_adv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    USE_CUSTOM_IMAGE = True
    IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "images" / "bird.jpg"

    IDX = None

    BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    DEFENCE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_pgdadv_best.pt"

    EPS_PX = float(sys.argv[2]) / 255 if len(sys.argv) > 2 else 4 / 255
    ALPHA_PX = 2 / 255
    STEPS = 7
    RANDOM_START = True

    print(f"Using image: {IMAGE_PATH}")
    print(f"Epsilon: {EPS_PX * 255:.0f}/255")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    baseline = load_model(device, BASELINE_CKPT)
    defence = load_model(device, DEFENCE_CKPT)

    if USE_CUSTOM_IMAGE:
        tfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if not IMAGE_PATH.exists():
            raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

        pil_img = Image.open(IMAGE_PATH).convert("RGB")
        x = tfm(pil_img).unsqueeze(0).to(device)

        true_label_str = "unknown (custom image)"

        _, clean_conf, clean_pred = predict(baseline, x)
        y_t = torch.tensor([clean_pred.item()], device=device)

    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        testset = torchvision.datasets.CIFAR10(
            root=str(ROOT / "data"),
            train=False,
            download=False,
            transform=tf
        )

        if IDX is None:
            IDX = random.randint(0, len(testset) - 1)

        x0, y0 = testset[IDX]
        x = x0.unsqueeze(0).to(device)
        y_t = torch.tensor([y0], device=device)

        true_label_str = classes[y0]

        _, clean_conf, clean_pred = predict(baseline, x)

    x_adv = pgd_attack_norm(
        baseline,
        x,
        y_t,
        eps_px=EPS_PX,
        alpha_px=ALPHA_PX,
        steps=STEPS,
        mean=mean,
        std=std,
        random_start=RANDOM_START
    )

    _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
    _, adv_conf_def, adv_pred_def = predict(defence, x_adv)

    print("\n===== RESULTS (same attacked image) =====")
    print(f"Image: {IMAGE_PATH} | True label: {true_label_str}")
    print(f"Clean (Baseline): {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
    print(f"After PGD (Baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
    print(f"After PGD (Defence):  {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

    clean_img = denorm_to_01(x, mean, std)[0].permute(1, 2, 0).detach().cpu()
    adv_img = denorm_to_01(x_adv, mean, std)[0].permute(1, 2, 0).detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    titles = [
        f"Clean Prediction\nBaseline: {classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)",
        f"After PGD Attack\nBaseline: {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)",
        f"Defence After PGD\nDefence: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)",
    ]

    axes[0, 0].imshow(clean_img, interpolation="nearest")
    axes[0, 0].set_title(titles[0])
    axes[0, 0].axis("off")

    axes[0, 1].imshow(adv_img, interpolation="nearest")
    axes[0, 1].set_title(titles[1])
    axes[0, 1].axis("off")

    axes[0, 2].imshow(adv_img, interpolation="nearest")
    axes[0, 2].set_title(titles[2])
    axes[0, 2].axis("off")

    axes[1, 0].imshow(clean_img, interpolation="lanczos")
    axes[1, 0].set_title("Human-eye view")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(adv_img, interpolation="lanczos")
    axes[1, 1].set_title("Human-eye view")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(adv_img, interpolation="lanczos")
    axes[1, 2].set_title("Human-eye view")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()