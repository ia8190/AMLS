import os
import random
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_name(name):
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name if name else "custom_baseline"


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        correct += (logits.argmax(1) == y).sum().item()

        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        correct += (logits.argmax(1) == y).sum().item()

        total += x.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CNN on CIFAR-10")

    parser.add_argument(
        "--name",
        type=str,
        default="custom_baseline",
        help="Name for the saved model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.name = safe_name(args.name)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("\nTraining baseline CNN with settings:")
    print(f"Model name: {args.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seed: {args.seed}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_tfms
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_tfms
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = CIFAR_CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    custom_dir = "custom_train"
    os.makedirs(custom_dir, exist_ok=True)

    best_path = os.path.join(custom_dir, f"{args.name}.pt")

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            testloader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(
                {
                    "model_name": args.name,
                    "model_type": "baseline",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_acc": best_acc,
                    "settings": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                    }
                },
                best_path
            )

    print(f"\nBest test accuracy: {best_acc*100:.2f}%")
    print(f"Saved custom model to: {best_path}")


if __name__ == "__main__":
    main()