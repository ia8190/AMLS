import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


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
    x_adv = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)

    model.zero_grad()
    loss.backward()

    return (x_adv + eps * x_adv.grad.sign()).detach()


# return top 3 predictions as text
def top3_string(probs):
    top3_conf, top3_idx = torch.topk(probs, 3)
    parts = []

    for c, i in zip(top3_conf.tolist(), top3_idx.tolist()):
        parts.append(f"{classes[i]} {c*100:.1f}%")

    return " | ".join(parts)


def main():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = Path(__file__).resolve().parents[2]

    default_image = ROOT / "images" / "birds.jpg"
    default_model = ROOT / "checkpoints" / "cnn_cifar10_best.pt"

    # get image path from command line or use default
    if len(sys.argv) > 1:
        IMAGE_PATH = Path(sys.argv[1])
    else:
        IMAGE_PATH = default_image

    # get epsilon from command line or use default
    if len(sys.argv) > 2:
        EPS = float(sys.argv[2]) / 255
    else:
        EPS = 8 / 255

    # get model path from command line or use default
    if len(sys.argv) > 3:
        CKPT_PATH = Path(sys.argv[3])
    else:
        CKPT_PATH = default_model

    print(f"Using image: {IMAGE_PATH}")
    print(f"Using epsilon: {EPS*255:.0f}/255")
    print(f"Using model: {CKPT_PATH}")

    CONF_THRESHOLD = 50.0

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # image transform for the model
    to_model = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = load_model(device, CKPT_PATH)

    # load image
    pil_img = Image.open(IMAGE_PATH).convert("RGB")

    # prepare image for model
    x = to_model(pil_img).unsqueeze(0).to(device)

    # make clean prediction
    with torch.no_grad():
        clean_probs = torch.softmax(model(x), dim=1)[0]
        clean_conf, clean_pred = clean_probs.max(dim=0)

    clean_conf_pct = clean_conf.item() * 100

    # hide label if confidence is low
    clean_label = (
        classes[clean_pred.item()]
        if clean_conf_pct >= CONF_THRESHOLD
        else "N/A"
    )

    clean_top3 = top3_string(clean_probs)

    # use clean prediction as label
    y = torch.tensor([clean_pred.item()]).to(device)

    # run fgsm attack
    x_adv = fgsm_attack(model, x, y, EPS)

    # make prediction after attack
    with torch.no_grad():
        adv_probs = torch.softmax(model(x_adv), dim=1)[0]
        adv_conf, adv_pred = adv_probs.max(dim=0)

    adv_conf_pct = adv_conf.item() * 100

    # hide label if confidence is low
    adv_label = (
        classes[adv_pred.item()]
        if adv_conf_pct >= CONF_THRESHOLD
        else "N/A"
    )

    adv_top3 = top3_string(adv_probs)

    # create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # clean image result
    axes[0].imshow(pil_img)
    axes[0].set_title(
        f"CLEAN\nPred: {clean_label} ({clean_conf_pct:.1f}%)"
    )
    axes[0].set_xlabel(f"Top-3: {clean_top3}")
    axes[0].axis("off")

    # attacked image result
    axes[1].imshow(pil_img)
    axes[1].set_title(
        f"AFTER FGSM\nε={EPS*255:.0f}/255\nPred: {adv_label} ({adv_conf_pct:.1f}%)"
    )
    axes[1].set_xlabel(f"Top-3: {adv_top3}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# start the script
if __name__ == "__main__":
    main()