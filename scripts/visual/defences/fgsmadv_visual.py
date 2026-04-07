# from pathlib import Path
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from PIL import Image

# classes = [
#     'airplane','automobile','bird','cat','deer',
#     'dog','frog','horse','ship','truck'
# ]

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

# def fgsm_attack(model, x, y, eps):
#     """FGSM in normalised space."""
#     model.eval()
#     x_adv = x.detach().clone().requires_grad_(True)
#     loss = nn.CrossEntropyLoss()(model(x_adv), y)
#     model.zero_grad(set_to_none=True)
#     loss.backward()
#     with torch.no_grad():
#         x_adv = x_adv + eps * x_adv.grad.sign()
#     return x_adv.detach()

# def find_root(start: Path):
#     p = start
#     while p != p.parent:
#         if (p / "checkpoints").exists():
#             return p
#         p = p.parent
#     raise RuntimeError("Project root not found (no 'checkpoints' folder).")

# def load_model(device, ckpt_path: Path):
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

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     ROOT = find_root(Path(__file__).resolve())

#     # pick an image that exists
#     IMAGE_PATH = ROOT / "images" / "bird.jpg"   # change if needed
#     #IMAGE_PATH = ROOT / "outputs" / "attacked_images" / "005_fgsm_eps0.03137_pred-dog.png"   # attacked_images.py

#     BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
#     DEFENCE_CKPT  = ROOT / "checkpoints" / "cnn_cifar10_fgsmadv_best.pt"

#     EPS = 8/255

#     mean = (0.4914, 0.4822, 0.4465)
#     std  = (0.2470, 0.2435, 0.2616)

#     tfm = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     if not IMAGE_PATH.exists():
#         raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

#     baseline = load_model(device, BASELINE_CKPT)
#     defence  = load_model(device, DEFENCE_CKPT)

#     img = Image.open(IMAGE_PATH).convert("RGB")
#     x = tfm(img).unsqueeze(0).to(device)

#     # 1) Clean prediction (baseline)
#     _, clean_conf, clean_pred = predict(baseline, x)

#     # Use baseline clean pred as label for attack demo
#     y = torch.tensor([clean_pred.item()]).to(device)

#     # 2) Make ONE attacked image using BASELINE model
#     x_adv = fgsm_attack(baseline, x, y, EPS)

#     # 3) Evaluate BOTH models on the SAME attacked image
#     _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
#     _, adv_conf_def,  adv_pred_def  = predict(defence,  x_adv)

#     # Terminal print
#     print("\nRESULTS (same attacked image)")
#     print(f"Clean Prediction: {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
#     print(f"After FGSM Attack (baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
#     print(f"Defence After FGSM Attack: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

#     # Plot
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(img)
#     axes[0].set_title(f"Clean Prediction\n{classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)")
#     axes[0].axis("off")

#     axes[1].imshow(img)
#     axes[1].set_title(f"After FGSM Attack\n{classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)")
#     axes[1].axis("off")

#     axes[2].imshow(img)
#     axes[2].set_title(f"Defence After FGSM Attack\n{classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()


##################
#######################
#####################
# from pathlib import Path
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from PIL import Image


# classes = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck"
# ]


# # ----------------------------
# # Model
# # ----------------------------
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


# # ----------------------------
# # FGSM Attack
# # ----------------------------
# def fgsm_attack(model, x, y, eps):
#     model.eval()
#     x_adv = x.detach().clone().requires_grad_(True)
#     loss = nn.CrossEntropyLoss()(model(x_adv), y)
#     model.zero_grad(set_to_none=True)
#     loss.backward()
#     with torch.no_grad():
#         x_adv = x_adv + eps * x_adv.grad.sign()
#     return x_adv.detach()


# # ----------------------------
# # Utilities
# # ----------------------------
# def find_root(start: Path) -> Path:
#     p = start
#     while p != p.parent:
#         if (p / "checkpoints").exists():
#             return p
#         p = p.parent
#     raise RuntimeError("Project root not found.")


# def load_model(device, ckpt_path):
#     model = CIFAR_CNN().to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     model.eval()
#     return model


# @torch.no_grad()
# def predict(model, x):
#     probs = torch.softmax(model(x), dim=1)[0]
#     conf, pred = probs.max(dim=0)
#     return conf.item(), pred.item()


# def denorm_to_img(x_norm, mean, std):
#     mean_t = torch.tensor(mean).view(1, 3, 1, 1)
#     std_t = torch.tensor(std).view(1, 3, 1, 1)
#     x = (x_norm.cpu() * std_t + mean_t).clamp(0, 1)[0]
#     return x.permute(1, 2, 0).numpy()


# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     ROOT = find_root(Path(__file__).resolve())

#     # -------- SETTINGS --------
#     BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
#     DEFENCE_CKPT  = ROOT / "checkpoints" / "cnn_cifar10_fgsmadv_best.pt"

#     CLEAN_IMAGE_PATH = ROOT / "outputs" / "attacked_images" / "006_fgsm_eps0.03137_pred-automobile.png"   # attacked_images.py

#     EPS = 8 / 255  # CHANGE THIS WHENEVER YOU WANT
#     # --------------------------

#     if not CLEAN_IMAGE_PATH.exists():
#         raise FileNotFoundError(f"Clean image not found: {CLEAN_IMAGE_PATH}")

#     mean = (0.4914, 0.4822, 0.4465)
#     std  = (0.2470, 0.2435, 0.2616)

#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     baseline = load_model(device, BASELINE_CKPT)
#     defence  = load_model(device, DEFENCE_CKPT)

#     # Load clean image
#     pil_clean = Image.open(CLEAN_IMAGE_PATH).convert("RGB")
#     x_clean = transform(pil_clean).unsqueeze(0).to(device)

#     # Clean prediction
#     clean_conf, clean_pred = predict(baseline, x_clean)

#     # Generate attack using TRUE predicted label
#     y = torch.tensor([clean_pred], device=device)
#     x_adv = fgsm_attack(baseline, x_clean, y, EPS)

#     # Predictions on attacked image
#     adv_conf_base, adv_pred_base = predict(baseline, x_adv)
#     adv_conf_def,  adv_pred_def  = predict(defence,  x_adv)

#     print("\nClean Prediction:", classes[clean_pred], f"({clean_conf*100:.1f}%)")
#     print("After FGSM Attack:", classes[adv_pred_base], f"({adv_conf_base*100:.1f}%)")
#     print("Defence After FGSM Attack:", classes[adv_pred_def], f"({adv_conf_def*100:.1f}%)")

#     clean_img = pil_clean
#     adv_img = denorm_to_img(x_adv, mean, std)

#     fig, axes = plt.subplots(1, 3, figsize=(16, 5))

#     axes[0].imshow(clean_img)
#     axes[0].set_title(f"Clean Prediction\n{classes[clean_pred]} ({clean_conf*100:.1f}%)")
#     axes[0].axis("off")

#     axes[1].imshow(adv_img)
#     axes[1].set_title(f"After FGSM Attack\nε={EPS:.4f}\n{classes[adv_pred_base]} ({adv_conf_base*100:.1f}%)")
#     axes[1].axis("off")

#     axes[2].imshow(adv_img)
#     axes[2].set_title(f"Defence After FGSM Attack\n{classes[adv_pred_def]} ({adv_conf_def*100:.1f}%)")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()





##################

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

def fgsm_attack(model, x, y, eps):
    """FGSM in normalised space."""
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach()

def find_root(start: Path):
    p = start
    while p != p.parent:
        if (p / "checkpoints").exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found (no 'checkpoints' folder).")

def load_model(device, ckpt_path: Path):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # pick an image that exists
    IMAGE_PATH = ROOT / "images" / "ship.jpg"   # change if needed
    #IMAGE_PATH = ROOT / "outputs" / "attacked_images" / "005_fgsm_eps0.03137_pred-dog.png"   # attacked_images.py

    BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    DEFENCE_CKPT  = ROOT / "checkpoints" / "cnn_cifar10_fgsmadv_best.pt"

    EPS = 16/255

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    baseline = load_model(device, BASELINE_CKPT)
    defence  = load_model(device, DEFENCE_CKPT)

    img = Image.open(IMAGE_PATH).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    # 1) Clean prediction (baseline)
    _, clean_conf, clean_pred = predict(baseline, x)

    # Use baseline clean pred as label for attack demo
    y = torch.tensor([clean_pred.item()]).to(device)

    # 2) Make ONE attacked image using BASELINE model
    x_adv = fgsm_attack(baseline, x, y, EPS)

    # 3) Evaluate BOTH models on the SAME attacked image
    _, adv_conf_base, adv_pred_base = predict(baseline, x_adv)
    _, adv_conf_def,  adv_pred_def  = predict(defence,  x_adv)

    # Terminal print
    print("\nRESULTS (same attacked image)")
    print(f"Clean Prediction: {classes[clean_pred.item()]} ({clean_conf.item()*100:.2f}%)")
    print(f"After FGSM Attack (baseline): {classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.2f}%)")
    print(f"Defence After FGSM Attack: {classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.2f}%)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Clean Prediction\n{classes[clean_pred.item()]} ({clean_conf.item()*100:.1f}%)")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].set_title(f"After FGSM Attack\n{classes[adv_pred_base.item()]} ({adv_conf_base.item()*100:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].set_title(f"Defence After FGSM Attack\n{classes[adv_pred_def.item()]} ({adv_conf_def.item()*100:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
