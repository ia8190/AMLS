import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

        # classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # forward pass
    def forward(self, x):
        return self.classifier(self.features(x))


# create an fgsm adversarial image
def fgsm_attack(model, x, y, eps):
    model.eval()

    x_adv = x.detach().clone().requires_grad_(True)

    loss = nn.CrossEntropyLoss()(model(x_adv), y)

    model.zero_grad(set_to_none=True)

    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()

    return x_adv.detach()


# find the project root folder
def find_root(start: Path):
    p = start

    while p != p.parent:
        if (p / "checkpoints").exists() or (p / "custom_train").exists():
            return p

        p = p.parent

    raise RuntimeError("Project root not found.")


# load a saved model checkpoint
def load_model(device, ckpt_path: Path):
    model = CIFAR_CNN().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # support both checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    return model


# make a prediction
@torch.no_grad()
def predict(model, x):
    probs = torch.softmax(model(x), dim=1)[0]

    conf, pred = probs.max(dim=0)

    return probs, conf, pred


def main():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # get image path from command line or use default
    IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "images" / "ship.jpg"

    # get epsilon from command line or use default
    EPS = float(sys.argv[2]) / 255 if len(sys.argv) > 2 else 16 / 255

    # default models
    BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    DEFENCE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_fgsmadv_best.pt"

    # use custom model if provided
    if len(sys.argv) > 3:
        BASELINE_CKPT = Path(sys.argv[3])

    print(f"Using image: {IMAGE_PATH}")
    print(f"Using epsilon: {EPS * 255:.0f}/255")
    print(f"Using baseline model: {BASELINE_CKPT}")
    print(f"Using defence model: {DEFENCE_CKPT}")

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)

    std = (0.2470, 0.2435, 0.2616)

    # image transform for the model
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # check image exists
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # load models and image
    baseline = load_model(device, BASELINE_CKPT)
    defence = load_model(device, DEFENCE_CKPT)
    img = Image.open(IMAGE_PATH).convert("RGB")

    # prepare image for model
    x = tfm(img).unsqueeze(0).to(device)

    # clean prediction
    _, clean_conf, clean_pred = predict(baseline, x)

    # use clean prediction as label
    y = torch.tensor([clean_pred.item()]).to(device)

    # attack baseline model
    x_adv = fgsm_attack(baseline, x, y, EPS)

    # predict using attacked image
    _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
    _, adv_conf_def, adv_pred_def = predict(defence, x_adv)

    print("\nRESULTS (same attacked image)")
    print(f"Clean Prediction: {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
    print(f"After FGSM Attack (baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
    print(f"Defence After FGSM Attack: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

    # create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # clean image prediction
    axes[0].imshow(img)
    axes[0].set_title(
        f"Clean Prediction\n"
        f"{classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)"
    )
    axes[0].axis("off")

    # baseline result after attack
    axes[1].imshow(img)
    axes[1].set_title(
        f"After FGSM Attack\n"
        f"{classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)"
    )
    axes[1].axis("off")

    # defence result after attack
    axes[2].imshow(img)
    axes[2].set_title(
        f"Defence After FGSM Attack\n"
        f"{classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# start the script
if __name__ == "__main__":
    main()