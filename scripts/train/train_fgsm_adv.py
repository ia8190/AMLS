from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ----------------------------
# Model
# ----------------------------
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
        x = self.features(x)
        return self.classifier(x)


# ----------------------------
# FGSM Attack (training)
# ----------------------------
def fgsm_attack(model, x, y, eps):
    """
    FGSM on NORMALISED inputs.
    Returns adversarial x_adv (still normalised).
    """
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach()


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ----------------------------
# Main Training
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Resolve project root from scripts/train/
    ROOT = Path(__file__).resolve().parents[2]
    CKPT_DIR = ROOT / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- Settings ----------------
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    EPS = 8 / 255          # FGSM epsilon (normalised space)
    ADV_RATIO = 0.5        # 0.5  half batch clean, half batch adversarial
    SAVE_BEST_AS = CKPT_DIR / "cnn_cifar10_fgsmadv_best.pt"
    SAVE_LAST_AS = CKPT_DIR / "cnn_cifar10_fgsmadv_last.pt"
    LOG_PATH = CKPT_DIR / "cnn_cifar10_fgsmadv_log.json"
    # ------------------------------------------

    # CIFAR-10 normalisation (must match your other scripts)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CIFAR_CNN().to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    print(f"FGSM Adv Training: eps={EPS:.5f}, adv_ratio={ADV_RATIO}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Decide how many in batch to convert to adversarial
            if ADV_RATIO > 0:
                n_adv = int(x.size(0) * ADV_RATIO)
            else:
                n_adv = 0

            # Build batch first part clean second part adversarial
            if n_adv > 0:
                x_clean, y_clean = x[:-n_adv], y[:-n_adv]
                x_part,  y_part  = x[-n_adv:], y[-n_adv:]

                # Generate adversarial samples using current model
                x_adv = fgsm_attack(model, x_part, y_part, EPS)

                x_mix = torch.cat([x_clean, x_adv], dim=0)
                y_mix = torch.cat([y_clean, y_part], dim=0)
            else:
                x_mix, y_mix = x, y

            opt.zero_grad(set_to_none=True)
            logits = model(x_mix)
            loss = loss_fn(logits, y_mix)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y_mix.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y_mix).sum().item()
            total += y_mix.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}% | {dt:.1f}s")

        # Save last
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "eps": EPS,
            "adv_ratio": ADV_RATIO,
            "mean": mean,
            "std": std,
        }, SAVE_LAST_AS)

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "eps": EPS,
                "adv_ratio": ADV_RATIO,
                "mean": mean,
                "std": std,
            }, SAVE_BEST_AS)
            print(f"New best saved: {SAVE_BEST_AS.name} (test_acc={best_acc*100:.2f}%)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "eps": float(EPS),
            "adv_ratio": float(ADV_RATIO),
            "seconds": dt
        })

        LOG_PATH.write_text(json.dumps(history, indent=2))

    print(f"\nDone. Best test accuracy: {best_acc*100:.2f}%")
    print(f"Best checkpoint: {SAVE_BEST_AS}")
    print(f"Last checkpoint: {SAVE_LAST_AS}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
