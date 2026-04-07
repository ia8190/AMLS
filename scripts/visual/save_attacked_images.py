from pathlib import Path
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
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

# ---------------- Attacks ----------------
def fgsm_attack(model, x, y, eps):
    """FGSM in NORMALISED space."""
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach()

def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    """PGD (L_inf) in NORMALISED space."""
    model.eval()
    x_orig = x.detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    else:
        x_adv = x_orig.clone()

    x_adv = x_adv.detach()
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

# ---------------- Utils ----------------
def find_root(start: Path) -> Path:
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

def denorm_to_uint8(x_norm, mean, std):
    """
    x_norm: (1,3,32,32) tensor in normalised space
    returns PIL image RGB (uint8) for saving
    """
    mean_t = torch.tensor(mean).view(1,3,1,1)
    std_t  = torch.tensor(std).view(1,3,1,1)
    x = (x_norm.cpu() * std_t + mean_t).clamp(0, 1)[0]  # (3,32,32)
    x = (x.permute(1,2,0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(x, mode="RGB")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # --------- SETTINGS ----------
    CKPT_PATH = ROOT / "checkpoints" / "cnn_cifar10_best.pt"   # baseline model
    OUT_DIR   = ROOT / "outputs" / "attacked_images"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_IMAGES = 10                  # how many test images to save
    SEED = 0

    EPS_FGSM = 8/255

    ALSO_SAVE_PGD = True
    EPS_PGD   = 8/255
    ALPHA_PGD = 1/255
    STEPS_PGD = 20
    RANDOM_START = True
    # ----------------------------

    torch.manual_seed(SEED)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # CIFAR-10 test set (labels included)
    testset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"), train=False, download=False, transform=test_tf
    )

    model = load_model(device, CKPT_PATH)

    metadata = {
        "checkpoint": str(CKPT_PATH),
        "n_images": N_IMAGES,
        "fgsm_eps": float(EPS_FGSM),
        "pgd": {
            "enabled": bool(ALSO_SAVE_PGD),
            "eps": float(EPS_PGD),
            "alpha": float(ALPHA_PGD),
            "steps": int(STEPS_PGD),
            "random_start": bool(RANDOM_START),
        },
        "items": []
    }

    print(f"Saving to: {OUT_DIR}")

    for i in range(N_IMAGES):
        x, y = testset[i]  # x is normalised tensor (3,32,32), y is label int
        x = x.unsqueeze(0).to(device)
        y_t = torch.tensor([y], device=device)

        # clean prediction
        _, clean_conf, clean_pred = predict(model, x)

        # FGSM (true label y)
        x_fgsm = fgsm_attack(model, x, y_t, EPS_FGSM)
        _, fgsm_conf, fgsm_pred = predict(model, x_fgsm)

        # PGD (optional)
        if ALSO_SAVE_PGD:
            x_pgd = pgd_attack(model, x, y_t, EPS_PGD, ALPHA_PGD, STEPS_PGD, random_start=RANDOM_START)
            _, pgd_conf, pgd_pred = predict(model, x_pgd)

        # save images
        clean_img = denorm_to_uint8(x, mean, std)
        fgsm_img  = denorm_to_uint8(x_fgsm, mean, std)

        clean_path = OUT_DIR / f"{i:03d}_clean_true-{classes[y]}.png"
        fgsm_path  = OUT_DIR / f"{i:03d}_fgsm_eps{EPS_FGSM:.5f}_pred-{classes[fgsm_pred.item()]}.png"

        clean_img.save(clean_path)
        fgsm_img.save(fgsm_path)

        pgd_path = None
        if ALSO_SAVE_PGD:
            pgd_img = denorm_to_uint8(x_pgd, mean, std)
            pgd_path = OUT_DIR / f"{i:03d}_pgd_eps{EPS_PGD:.5f}_s{STEPS_PGD}_pred-{classes[pgd_pred.item()]}.png"
            pgd_img.save(pgd_path)

        item = {
            "index": i,
            "true_label": classes[y],
            "clean": {
                "pred": classes[clean_pred.item()],
                "conf_pct": round(clean_conf.item()*100, 2),
                "file": clean_path.name
            },
            "fgsm": {
                "pred": classes[fgsm_pred.item()],
                "conf_pct": round(fgsm_conf.item()*100, 2),
                "file": fgsm_path.name
            }
        }
        if ALSO_SAVE_PGD:
            item["pgd"] = {
                "pred": classes[pgd_pred.item()],
                "conf_pct": round(pgd_conf.item()*100, 2),
                "file": pgd_path.name
            }

        metadata["items"].append(item)

        print(f"[{i:03d}] true={classes[y]:10s} | clean={classes[clean_pred.item()]:10s} ({clean_conf.item()*100:5.1f}%)"
              f" | fgsm={classes[fgsm_pred.item()]:10s} ({fgsm_conf.item()*100:5.1f}%)"
              + (f" | pgd={classes[pgd_pred.item()]:10s} ({pgd_conf.item()*100:5.1f}%)" if ALSO_SAVE_PGD else ""))

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Wrote: {OUT_DIR / 'metadata.json'}")

if __name__ == "__main__":
    main()
