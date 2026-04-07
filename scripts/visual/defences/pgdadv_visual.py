# scripts/visual/defences/pgdadv_visual.py

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


# ---------------- Model ----------------
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


# ---------------- Utils ----------------
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
    """(1,3,32,32) normalized -> (1,3,32,32) in [0,1]"""
    mean_t = torch.tensor(mean, device=x_norm.device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=x_norm.device).view(1, 3, 1, 1)
    return (x_norm * std_t + mean_t).clamp(0, 1)

def pgd_attack_norm(
    model,
    x_norm,
    y,
    eps_px=8/255,
    alpha_px=2/255,
    steps=7,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    random_start=True,
):
    """
    PGD (L_inf). eps/alpha specified in PIXEL space (0..1),
    applied in NORMALISED space by scaling per-channel with std.
    Clamped to valid normalised range corresponding to pixel [0,1].
    """
    model.eval()
    device = x_norm.device

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)

    # Convert pixel-space eps/alpha to normalized space per-channel values
    eps = eps_px / std_t
    alpha = alpha_px / std_t

    # Valid normalized bounds that map back to pixel [0,1]
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
            # Project to Linf ball around original
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            # Clamp to valid range
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)

        x_adv = x_adv.detach()

    return x_adv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # ================= SETTINGS =================
    USE_CUSTOM_IMAGE = True          # True = use IMAGE_PATH, False = random CIFAR-10 test image each run
    IMAGE_PATH = ROOT / "images" / "bird.jpg"   # any image file 

    IDX = None                       # if None random each run, else set 0..9999 (only used when USE_CUSTOM_IMAGE=False)

    BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    DEFENCE_CKPT  = ROOT / "checkpoints" / "cnn_cifar10_pgdadv_best.pt"  # PGD-AT model

    EPS_PX = 4/255
    ALPHA_PX = 2/255
    STEPS = 7
    RANDOM_START = True
    # ===========================================

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    baseline = load_model(device, BASELINE_CKPT)
    defence  = load_model(device, DEFENCE_CKPT)

    # Load input either from custom image OR CIFAR test 
    if USE_CUSTOM_IMAGE:
        tfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if not IMAGE_PATH.exists():
            raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

        pil_img = Image.open(IMAGE_PATH).convert("RGB")
        x = tfm(pil_img).unsqueeze(0).to(device)  # (1,3,32,32)

        # We'll just print "unknown" for true label since custom image has none.
        true_label_str = "unknown (custom image)"
        true_y = None

        # For demo use baseline's clean prediction as y for the attack (like your FGSM script)
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

        true_y = y0
        true_label_str = classes[y0]

        # Clean prediction (baseline)
        _, clean_conf, clean_pred = predict(baseline, x)

    # Attack using BASELINE model, produce ONE attacked image 
    x_adv = pgd_attack_norm(
        baseline, x, y_t,
        eps_px=EPS_PX, alpha_px=ALPHA_PX, steps=STEPS,
        mean=mean, std=std, random_start=RANDOM_START
    )

    # Evaluate BOTH models on same attacked image 
    _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
    _, adv_conf_def,  adv_pred_def  = predict(defence,  x_adv)

    # Print results 
    print("\n===== RESULTS (same attacked image) =====")
    if not USE_CUSTOM_IMAGE:
        print(f"Index: {IDX} | True label: {true_label_str}")
    else:
        print(f"Image: {IMAGE_PATH} | True label: {true_label_str}")

    print(f"Clean (Baseline): {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
    print(f"After PGD (Baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
    print(f"After PGD (Defence):  {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

    # Convert to displayable images
    clean_img = denorm_to_01(x, mean, std)[0].permute(1, 2, 0).detach().cpu()
    adv_img   = denorm_to_01(x_adv, mean, std)[0].permute(1, 2, 0).detach().cpu()

    show_clean = clean_img
    show_adv = adv_img

    # ---- Plot (3 panels) ----
    # ---- Plot pixel vs human-eye view ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    titles = [
        f"Clean\nBaseline: {classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)",
        f"After PGD Attack\nBaseline: {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)",
        f"Defence After PGD\nDefence: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)",
    ]

    # Row 1 - pixel view (true CIFAR)
    axes[0,0].imshow(clean_img, interpolation="nearest")
    axes[0,0].set_title(titles[0])
    axes[0,0].axis("off")

    axes[0,1].imshow(adv_img, interpolation="nearest")
    axes[0,1].set_title(titles[1])
    axes[0,1].axis("off")

    axes[0,2].imshow(adv_img, interpolation="nearest")
    axes[0,2].set_title(titles[2])
    axes[0,2].axis("off")

    # Row 2 - human-eye smooth view
    axes[1,0].imshow(clean_img, interpolation="lanczos")
    axes[1,0].set_title("Human-eye view")
    axes[1,0].axis("off")

    axes[1,1].imshow(adv_img, interpolation="lanczos")
    axes[1,1].set_title("Human-eye view")
    axes[1,1].axis("off")

    axes[1,2].imshow(adv_img, interpolation="lanczos")
    axes[1,2].set_title("Human-eye view")
    axes[1,2].axis("off")

    plt.tight_layout()
    plt.show()  
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axes[0].imshow(show_clean)
    # axes[0].set_title(
    #     f"Clean\n"
    #     f"Baseline: {classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)"
    # )
    # axes[0].axis("off")

    # axes[1].imshow(show_adv)
    # axes[1].set_title(
    #     f"After PGD Attack\n"
    #     f"Baseline: {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)"
    # )
    # axes[1].axis("off")

    # axes[2].imshow(show_adv)
    # axes[2].set_title(
    #     f"Defence After PGD\n"
    #     f"Defence: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)"
    # )
    # axes[2].axis("off")

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()

# # scripts/visual/defences/pgdadv_visual.py
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import random


# classes = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck"
# ]

# # ---------------- Model ----------------
# class CIFAR_CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x):
#         return self.classifier(self.features(x))

# # ---------------- Utils ----------------
# def find_root(start: Path) -> Path:
#     p = start
#     while p != p.parent:
#         if (p / "checkpoints").exists():
#             return p
#         p = p.parent
#     raise RuntimeError("Project root not found (no 'checkpoints' folder).")

# def load_model(device, ckpt_path: Path) -> nn.Module:
#     model = CIFAR_CNN().to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
#     return model

# @torch.no_grad()
# def predict(model, x):
#     probs = torch.softmax(model(x), dim=1)[0]
#     conf, pred = probs.max(dim=0)
#     return probs, conf, pred

# def denorm_to_01(x_norm, mean, std):
#     """(1,3,32,32) normalized -> (1,3,32,32) in [0,1]"""
#     mean_t = torch.tensor(mean, device=x_norm.device).view(1, 3, 1, 1)
#     std_t  = torch.tensor(std,  device=x_norm.device).view(1, 3, 1, 1)
#     return (x_norm * std_t + mean_t).clamp(0, 1)

# def pgd_attack_norm(
#     model,
#     x_norm,
#     y,
#     eps_px=8/255,
#     alpha_px=2/255,
#     steps=7,
#     mean=(0.4914, 0.4822, 0.4465),
#     std=(0.2470, 0.2435, 0.2616),
#     random_start=True,
# ):
#     """
#     PGD (L_inf). eps/alpha specified in PIXEL space (0..1),
#     applied in NORMALISED space by scaling per-channel with std.
#     Clamped to valid normalised range corresponding to pixel [0,1].
#     """
#     model.eval()
#     device = x_norm.device

#     mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
#     std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)

#     # Convert pixel-space eps/alpha to normalized-space per-channel values
#     eps = eps_px / std_t
#     alpha = alpha_px / std_t

#     # Valid normalized bounds that map back to pixel [0,1]
#     x_min = (0.0 - mean_t) / std_t
#     x_max = (1.0 - mean_t) / std_t

#     x_orig = x_norm.detach()

#     if random_start:
#         x_adv = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
#         x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
#         x_adv = torch.max(torch.min(x_adv, x_max), x_min)
#     else:
#         x_adv = x_orig.clone()

#     loss_fn = nn.CrossEntropyLoss()

#     for _ in range(steps):
#         x_adv.requires_grad_(True)
#         loss = loss_fn(model(x_adv), y)
#         model.zero_grad(set_to_none=True)
#         loss.backward()

#         with torch.no_grad():
#             x_adv = x_adv + alpha * x_adv.grad.sign()
#             # Project to Linf ball around original
#             x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
#             # Clamp to valid range
#             x_adv = torch.max(torch.min(x_adv, x_max), x_min)

#         x_adv = x_adv.detach()

#     return x_adv

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     ROOT = find_root(Path(__file__).resolve())

#     # -------- SETTINGS --------
#     IDX = random.randint(0, 9999)
#     BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
#     DEFENCE_CKPT  = ROOT / "checkpoints" / "cnn_cifar10_pgdadv_best.pt"  # your PGD-AT best model

#     EPS_PX = 8/255
#     ALPHA_PX = 2/255
#     STEPS = 7
#     RANDOM_START = True
#     # -------------------------

#     mean = (0.4914, 0.4822, 0.4465)
#     std  = (0.2470, 0.2435, 0.2616)

#     tf = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     # Load CIFAR-10 test set
#     testset = torchvision.datasets.CIFAR10(
#         root=str(ROOT / "data"),
#         train=False,
#         download=False,
#         transform=tf
#     )

#     x, y = testset[IDX]                 # x: normalized (3,32,32), y: int
#     x = x.unsqueeze(0).to(device)       # (1,3,32,32)
#     y_t = torch.tensor([y], device=device)

#     baseline = load_model(device, BASELINE_CKPT)
#     defence  = load_model(device, DEFENCE_CKPT)

#     # 1) Clean prediction (baseline)
#     _, clean_conf, clean_pred = predict(baseline, x)

#     # 2) Make ONE attacked image using BASELINE model (PGD)
#     x_adv = pgd_attack_norm(
#         baseline, x, y_t,
#         eps_px=EPS_PX, alpha_px=ALPHA_PX, steps=STEPS,
#         mean=mean, std=std, random_start=RANDOM_START
#     )

#     # 3) Evaluate BOTH models on the SAME attacked image
#     _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
#     _, adv_conf_def,  adv_pred_def  = predict(defence,  x_adv)

#     # Terminal print
#     print("\n===== RESULTS (same attacked image) =====")
#     print(f"Index: {IDX} | True label: {classes[y]}")
#     print(f"Clean (Baseline): {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
#     print(f"After PGD (Baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
#     print(f"After PGD (Defence):  {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

#     # Convert to displayable images
#     clean_img = denorm_to_01(x, mean, std)[0].permute(1, 2, 0).detach().cpu()
#     adv_img   = denorm_to_01(x_adv, mean, std)[0].permute(1, 2, 0).detach().cpu()

#     # Plot (3 panels like your FGSM visualise)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))

#     # Row 1: true pixel view (nearest) 
#     axes[0,0].imshow(clean_img, interpolation="nearest")
#     axes[0,0].set_title(
#         f"Clean (true={classes[y]})\n"
#         f"{classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)"
#     )
#     axes[0,0].axis("off")

#     axes[0,1].imshow(adv_img, interpolation="nearest")
#     axes[0,1].set_title(
#         f"After PGD Attack\n"
#         f"{classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)"
#     )
#     axes[0,1].axis("off")

#     axes[0,2].imshow(adv_img, interpolation="nearest")
#     axes[0,2].set_title(
#         f"Defence After PGD\n"
#         f"{classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)"
#     )
#     axes[0,2].axis("off")

#     # --- Row 2: human-eye smooth view (bilinear) ---
#     axes[1,0].imshow(clean_img, interpolation="bilinear")
#     axes[1,0].set_title("Human-eye view")
#     axes[1,0].axis("off")

#     axes[1,1].imshow(adv_img, interpolation="bilinear")
#     axes[1,1].set_title("Human-eye view")
#     axes[1,1].axis("off")

#     axes[1,2].imshow(adv_img, interpolation="bilinear")
#     axes[1,2].set_title("Human-eye view")
#     axes[1,2].axis("off")

#     plt.tight_layout()
#     plt.show()

#     # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # axes[0].imshow(clean_img)
#     # axes[0].set_title(
#     #     f"Clean (true={classes[y]})\n"
#     #     f"{classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)"
#     # )
#     # axes[0].axis("off")

#     # axes[1].imshow(adv_img)
#     # axes[1].set_title(
#     #     f"After PGD Attack\n"
#     #     f"{classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)"
#     # )
#     # axes[1].axis("off")

#     # axes[2].imshow(adv_img)
#     # axes[2].set_title(
#     #     f"Defence After PGD\n"
#     #     f"{classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)"
#     # )
#     # axes[2].axis("off")

#     # plt.tight_layout()
#     # plt.show()

# if __name__ == "__main__":
#     main()
