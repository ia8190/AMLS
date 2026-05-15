from pathlib import Path
import json
import time
import argparse
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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


def safe_name(name):
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name if name else "custom_fgsm_adv"


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGSM adversarial CNN on CIFAR-10")

    parser.add_argument("--name", type=str, default="custom_fgsm_adv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=8.0, help="FGSM epsilon value out of 255")
    parser.add_argument("--adv-ratio", type=float, default=0.5, help="Ratio of each batch converted to adversarial examples")

    return parser.parse_args()


def fgsm_attack(model, x, y, eps):
    x_adv = x.detach().clone().requires_grad_(True)

    loss = nn.CrossEntropyLoss()(model(x_adv), y)

    model.zero_grad(set_to_none=True)

    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()

    return x_adv.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    args = parse_args()

    args.name = safe_name(args.name)

    if args.eps < 0 or args.eps > 255:
        raise ValueError("Epsilon must be between 0 and 255.")

    if args.adv_ratio < 0 or args.adv_ratio > 1:
        raise ValueError("Adversarial ratio must be between 0 and 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = Path(__file__).resolve().parents[2]

    SAVE_DIR = ROOT / "custom_train"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    EPS = args.eps / 255

    SAVE_BEST_AS = SAVE_DIR / f"{args.name}_best.pt"
    SAVE_LAST_AS = SAVE_DIR / f"{args.name}_last.pt"
    LOG_PATH = SAVE_DIR / f"{args.name}_log.json"

    print("\nFGSM adversarial training settings:")
    print(f"Model name: {args.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Epsilon: {args.eps}/255")
    print(f"Adversarial ratio: {args.adv_ratio}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

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

    train_set = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"),
        train=True,
        download=True,
        transform=train_tf
    )

    test_set = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"),
        train=False,
        download=True,
        transform=test_tf
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = CIFAR_CNN().to(device)

    opt = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            n_adv = int(x.size(0) * args.adv_ratio)

            if n_adv > 0:
                x_clean = x[:-n_adv]
                y_clean = y[:-n_adv]

                x_part = x[-n_adv:]
                y_part = y[-n_adv:]

                x_adv = fgsm_attack(model, x_part, y_part, EPS)

                x_mix = torch.cat([x_clean, x_adv], dim=0)
                y_mix = torch.cat([y_clean, y_part], dim=0)

            else:
                x_mix = x
                y_mix = y

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

        seconds = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc*100:.2f}% | "
            f"test_acc={test_acc*100:.2f}% | "
            f"{seconds:.1f}s"
        )

        checkpoint = {
            "model_name": args.name,
            "model_type": "fgsm_adversarial",
            "model_state": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "eps": EPS,
            "eps_255": args.eps,
            "adv_ratio": args.adv_ratio,
            "mean": mean,
            "std": std,
            "settings": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "eps_255": args.eps,
                "adv_ratio": args.adv_ratio,
            }
        }

        torch.save(checkpoint, SAVE_LAST_AS)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, SAVE_BEST_AS)
            print(f"New best saved: {SAVE_BEST_AS.name} (test_acc={best_acc*100:.2f}%)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "eps": float(EPS),
            "eps_255": float(args.eps),
            "adv_ratio": float(args.adv_ratio),
            "seconds": seconds
        })

        LOG_PATH.write_text(json.dumps(history, indent=2))

    print(f"\nDone. Best test accuracy: {best_acc*100:.2f}%")
    print(f"Best checkpoint: {SAVE_BEST_AS}")
    print(f"Last checkpoint: {SAVE_LAST_AS}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()