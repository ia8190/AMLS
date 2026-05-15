import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
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


def fgsm_attack(model, x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad()
    loss.backward()
    return (x_adv + eps * x_adv.grad.sign()).detach()


def top3_string(probs):
    top3_conf, top3_idx = torch.topk(probs, 3)
    parts = []
    for c, i in zip(top3_conf.tolist(), top3_idx.tolist()):
        parts.append(f"{classes[i]} {c*100:.1f}%")
    return " | ".join(parts)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = Path(__file__).resolve().parents[2]

    default_image = ROOT / "images" / "birds.jpg"

    if len(sys.argv) > 1:
        IMAGE_PATH = Path(sys.argv[1])
    else:
        IMAGE_PATH = default_image

    CKPT_PATH = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    if len(sys.argv) > 2:
        EPS = float(sys.argv[2]) / 255
    else:
        EPS = 8 / 255
    CONF_THRESHOLD = 50.0
    SAVE_FIG = False
    SAVE_PATH = ROOT / "fgsm_external_demo_top3.png"

    print(f"Using image: {IMAGE_PATH}")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    to_model = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = CIFAR_CNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pil_img = Image.open(IMAGE_PATH).convert("RGB")
    x = to_model(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        clean_probs = torch.softmax(model(x), dim=1)[0]
        clean_conf, clean_pred = clean_probs.max(dim=0)

    clean_conf_pct = clean_conf.item() * 100
    clean_label = classes[clean_pred.item()] if clean_conf_pct >= CONF_THRESHOLD else "N/A"
    clean_top3 = top3_string(clean_probs)

    y = torch.tensor([clean_pred.item()]).to(device)

    x_adv = fgsm_attack(model, x, y, EPS)

    with torch.no_grad():
        adv_probs = torch.softmax(model(x_adv), dim=1)[0]
        adv_conf, adv_pred = adv_probs.max(dim=0)

    adv_conf_pct = adv_conf.item() * 100
    adv_label = classes[adv_pred.item()] if adv_conf_pct >= CONF_THRESHOLD else "N/A"
    adv_top3 = top3_string(adv_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(pil_img)
    axes[0].set_title(
        f"CLEAN (human view)\nPred: {clean_label} ({clean_conf_pct:.1f}%)"
    )
    axes[0].set_xlabel(f"Top-3: {clean_top3}")
    axes[0].axis("off")

    axes[1].imshow(pil_img)
    axes[1].set_title(
        f"AFTER FGSM (human view)\nε={EPS:.4f}\nPred: {adv_label} ({adv_conf_pct:.1f}%)"
    )
    axes[1].set_xlabel(f"Top-3: {adv_top3}")
    axes[1].axis("off")

    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {SAVE_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
