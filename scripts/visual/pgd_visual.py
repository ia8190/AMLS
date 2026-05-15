import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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


# return top 3 predictions as text
def top3_string(probs):
    top3_conf, top3_idx = torch.topk(probs, 3)

    return " | ".join([
        f"{classes[i]} {c*100:.1f}%"
        for c, i in zip(top3_conf.tolist(), top3_idx.tolist())
    ])


# create a pgd adversarial image and return the perturbation
def pgd_attack_with_delta(model, x, y, eps, alpha, steps, random_start=True):
    model.eval()
    x_orig = x.detach()

    # return original image if attack is disabled
    if eps == 0 or steps == 0:
        return x_orig, torch.zeros_like(x_orig)

    # start from random noise near the image
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        x_adv = x_orig.clone()

    x_adv = x_adv.detach()
    loss_fn = nn.CrossEntropyLoss()

    # apply pgd steps
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # keep image inside epsilon range
            x_adv = torch.max(
                torch.min(x_adv, x_orig + eps),
                x_orig - eps
            )

        x_adv = x_adv.detach()

    delta = (x_adv - x_orig).detach()

    return x_adv, delta


# convert normalised image back to display range
def denorm_batch(x, mean, std):
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    out = (x.cpu() * std_t + mean_t).clamp(0, 1)
    return out[0]


# convert perturbation sign to rgb image
def delta_to_rgb_sign(delta):
    s = delta.sign().cpu()[0]
    rgb = (s + 1) / 2.0
    return rgb.clamp(0, 1)


# convert perturbation to heatmap values
def delta_to_heatmap(delta):
    mag = delta.abs().mean(dim=1)[0].cpu().numpy()
    return mag


def main():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    default_image = os.path.join(ROOT, "images", "birds.jpg")
    default_model = os.path.join(ROOT, "checkpoints", "cnn_cifar10_best.pt")

    # get image, epsilon and model from command line if provided
    IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else default_image
    EPS = float(sys.argv[2]) / 255 if len(sys.argv) > 2 else 8 / 255
    CKPT_PATH = sys.argv[3] if len(sys.argv) > 3 else default_model

    # pgd settings
    STEPS = 20
    ALPHA = 1 / 255
    RANDOM_START = True

    # display settings
    UPSCALE_TO = 256
    SHOW_HEATMAP = False
    SAVE_FIG = False

    SAVE_PATH = os.path.join(
        ROOT,
        "results",
        "figures",
        "pgd_4panel_with_modelview.png"
    )

    print(f"Using image: {IMAGE_PATH}")
    print(f"Using epsilon: {EPS*255:.0f}/255")
    print(f"Using model: {CKPT_PATH}")

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
    human = Image.open(IMAGE_PATH).convert("RGB")

    # prepare image for model
    x = to_model(human).unsqueeze(0).to(device)

    # clean prediction
    with torch.no_grad():
        clean_probs = torch.softmax(model(x), dim=1)[0]
        clean_conf, clean_pred = clean_probs.max(dim=0)

    # use clean prediction as label
    y = torch.tensor([clean_pred.item()]).to(device)

    # run pgd attack
    x_adv, delta = pgd_attack_with_delta(
        model,
        x,
        y,
        eps=EPS,
        alpha=ALPHA,
        steps=STEPS,
        random_start=RANDOM_START
    )

    # prediction after attack
    with torch.no_grad():
        adv_probs = torch.softmax(model(x_adv), dim=1)[0]
        adv_conf, adv_pred = adv_probs.max(dim=0)

    # choose perturbation display
    if SHOW_HEATMAP:
        mag = delta_to_heatmap(delta)
        mag = mag / (mag.max() + 1e-8)

        mid_img = Image.fromarray(
            (mag * 255).astype(np.uint8)
        ).resize((UPSCALE_TO, UPSCALE_TO), Image.NEAREST)

        mid_title = "Perturbation magnitude |δ| (upscaled)"
        mid_cmap = "gray"

    else:
        rgb = delta_to_rgb_sign(delta)
        mid_pil = transforms.ToPILImage()(rgb)

        mid_img = mid_pil.resize(
            (UPSCALE_TO, UPSCALE_TO),
            resample=Image.NEAREST
        )

        mid_title = "sign(δ) pattern (upscaled)"
        mid_cmap = None

    # prepare adversarial image for display
    adv_vis = denorm_batch(x_adv, mean, std)
    adv_vis_pil = transforms.ToPILImage()(adv_vis)

    adv_model_view = adv_vis_pil.resize(
        (UPSCALE_TO, UPSCALE_TO),
        resample=Image.BICUBIC
    )

    # create comparison figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # clean human view
    axes[0].imshow(human)
    axes[0].set_title(
        f"CLEAN (human view)\n"
        f"Pred: {classes[clean_pred]} ({clean_conf.item()*100:.1f}%)"
    )
    axes[0].set_xlabel(f"Top-3: {top3_string(clean_probs)}")
    axes[0].axis("off")

    # perturbation view
    axes[1].imshow(mid_img, cmap=mid_cmap)
    axes[1].set_title(mid_title)
    axes[1].set_xlabel(
        f"ε={EPS:.4f}, steps={STEPS}, α={ALPHA:.4f}"
    )
    axes[1].axis("off")

    # attacked human view
    axes[2].imshow(human)
    axes[2].set_title(
        f"AFTER PGD (human view)\n"
        f"Pred: {classes[adv_pred]} ({adv_conf.item()*100:.1f}%)"
    )
    axes[2].set_xlabel(f"Top-3: {top3_string(adv_probs)}")
    axes[2].axis("off")

    # attacked model view
    axes[3].imshow(adv_model_view)
    axes[3].set_title(
        "AFTER PGD (model view)\n"
        "32×32 adv input (upscaled)"
    )
    axes[3].set_xlabel("Denormalised for display")
    axes[3].axis("off")

    plt.tight_layout()

    # save figure if enabled
    if SAVE_FIG:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        plt.savefig(
            SAVE_PATH,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved figure to: {SAVE_PATH}")

    plt.show()


# start the script
if __name__ == "__main__":
    main()