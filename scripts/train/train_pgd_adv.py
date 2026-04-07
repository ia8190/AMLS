from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

# ---------------- Utils ----------------
def find_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / "checkpoints").exists():
            return p
        p = p.parent
    raise RuntimeError("Project root not found (no 'checkpoints' folder).")

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def pgd_attack_normalized_space(
    model, x_norm, y, eps_px, alpha_px, steps, mean, std, random_start=True
):
    """
    PGD-Linf where eps/alpha are specified in PIXEL space (0..1),
    but attack is applied in NORMALISED space by scaling per-channel with std.

    Also clamps to valid normalised range corresponding to pixel [0,1].
    """
    model.eval()

    device = x_norm.device
    mean_t = torch.tensor(mean, device=device).view(1,3,1,1)
    std_t  = torch.tensor(std, device=device).view(1,3,1,1)

    # Convert pixel-space eps/alpha to normalized-space eps/alpha (per channel)
    eps = (eps_px / std_t)
    alpha = (alpha_px / std_t)

    # Clamp bounds in normalized space (pixel in [0,1])
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
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # project back into Linf ball around x_orig (per-channel eps)
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            # clamp to valid image range
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)

        x_adv = x_adv.detach()

    return x_adv

def pgd_eval_accuracy(model, loader, device, eps_px, alpha_px, steps, mean, std, random_start=True):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # gradients needed here to craft adversarial examples
        x_adv = pgd_attack_normalized_space(
            model, x, y,
            eps_px=eps_px, alpha_px=alpha_px, steps=steps,
            mean=mean, std=std, random_start=random_start
        )

        # no gradients needed for measuring accuracy
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # ---------------- SETTINGS ----------------
    DATA_DIR = ROOT / "data"
    CKPT_DIR = ROOT / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Start from baseline weights? (set to None to train from scratch)
    BASELINE_CKPT = CKPT_DIR / "cnn_cifar10_best.pt"

    EPOCHS = 20
    BATCH_SIZE = 128
    LR = 1e-3
    WEIGHT_DECAY = 5e-4

    # PGD params
    EPS_PX = 8/255
    ALPHA_PX = 2/255
    STEPS = 7
    RANDOM_START = True

    # Mix clean + adv loss
    ADV_WEIGHT = 0.5   

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=tf)
    testset  = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=tf)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = CIFAR_CNN().to(device)

    if BASELINE_CKPT.exists():
        ckpt = torch.load(BASELINE_CKPT, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded baseline weights from: {BASELINE_CKPT}")
        else:
            # in case raw state_dict was saved 
            model.load_state_dict(ckpt)
            print(f"Loaded baseline state_dict from: {BASELINE_CKPT}")
    else:
        print("Baseline ckpt not found; training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_pgd_acc = -1.0
    history = []

    last_path = CKPT_DIR / "cnn_cifar10_pgdadv_last.pt"
    best_path = CKPT_DIR / "cnn_cifar10_pgdadv_best.pt"
    hist_path = CKPT_DIR / "cnn_cifar10_pgdadv_history.json"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_clean, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Generate PGD adversarial examples (attack uses current model)
            x_adv = pgd_attack_normalized_space(
                model, x, y, eps_px=EPS_PX, alpha_px=ALPHA_PX, steps=STEPS,
                mean=mean, std=std, random_start=RANDOM_START
            )

            # Switch back to train for BN/dropout
            model.train()
            optimizer.zero_grad(set_to_none=True)

            logits_clean = model(x)
            logits_adv   = model(x_adv)

            loss_clean = loss_fn(logits_clean, y)
            loss_adv   = loss_fn(logits_adv, y)

            loss = (1.0 - ADV_WEIGHT) * loss_clean + ADV_WEIGHT * loss_adv
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

            with torch.no_grad():
                pred_clean = logits_clean.argmax(1)
                correct_clean += (pred_clean == y).sum().item()
                total += y.numel()

        train_loss = running_loss / total
        train_acc  = correct_clean / total

        # Evaluate
        clean_acc = accuracy(model, test_loader, device)
        pgd_acc   = pgd_eval_accuracy(
            model, test_loader, device,
            eps_px=EPS_PX, alpha_px=ALPHA_PX, steps=STEPS,
            mean=mean, std=std, random_start=RANDOM_START
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_clean_acc": round(clean_acc, 6),
            "test_pgd_acc": round(pgd_acc, 6),
            "eps_px": float(EPS_PX),
            "alpha_px": float(ALPHA_PX),
            "steps": int(STEPS),
            "adv_weight": float(ADV_WEIGHT),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.1f}% | "
            f"test_clean={clean_acc*100:5.1f}% test_pgd={pgd_acc*100:5.1f}%"
        )

        # Save last
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "meta": {
                "mean": mean, "std": std,
                "eps_px": float(EPS_PX),
                "alpha_px": float(ALPHA_PX),
                "steps": int(STEPS),
                "random_start": bool(RANDOM_START),
                "adv_weight": float(ADV_WEIGHT),
            }
        }, last_path)

        # Save best by PGD accuracy
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_pgd_acc": best_pgd_acc,
                "meta": {
                    "mean": mean, "std": std,
                    "eps_px": float(EPS_PX),
                    "alpha_px": float(ALPHA_PX),
                    "steps": int(STEPS),
                    "random_start": bool(RANDOM_START),
                    "adv_weight": float(ADV_WEIGHT),
                }
            }, best_path)
            print(f"Saved BEST (by PGD acc): {best_path}")

        hist_path.write_text(json.dumps(history, indent=2))

    print("\nDone.")
    print("Last:", last_path)
    print("Best:", best_path)
    print("History:", hist_path)

if __name__ == "__main__":
    main()
