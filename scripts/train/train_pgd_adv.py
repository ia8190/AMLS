from pathlib import Path
import json
import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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


# make model name safe for saving
def safe_name(name):
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    return name if name else "custom_pgd_adv"


# read command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train PGD adversarial CNN on CIFAR-10")
    parser.add_argument("--name", type=str, default="custom_pgd_adv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eps", type=float, default=8.0, help="PGD epsilon value out of 255")
    parser.add_argument("--alpha", type=float, default=2.0, help="PGD alpha value out of 255")
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument(
        "--no-random-start",
        action="store_true",
        help="Disable random start for PGD"
    )

    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch instead of starting from baseline checkpoint"
    )

    return parser.parse_args()


# find the project root folder
def find_root(start: Path) -> Path:
    p = start

    while p != p.parent:
        if (p / "checkpoints").exists() or (p / "custom_train").exists():
            return p

        p = p.parent

    return Path(__file__).resolve().parents[2]


# calculate clean accuracy
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / total


# create a pgd adversarial batch in normalised space
def pgd_attack_normalized_space(
    model,
    x_norm,
    y,
    eps_px,
    alpha_px,
    steps,
    mean,
    std,
    random_start=True
):
    model.eval()
    device = x_norm.device
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    # convert pixel values to normalised space
    eps = eps_px / std_t
    alpha = alpha_px / std_t

    # valid normalised image range
    x_min = (0.0 - mean_t) / std_t
    x_max = (1.0 - mean_t) / std_t

    x_orig = x_norm.detach()

    # start from random noise near the image
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)
    else:
        x_adv = x_orig.clone()

    loss_fn = nn.CrossEntropyLoss()

    # apply pgd steps
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # keep image inside epsilon range
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

            # keep image inside valid range
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)

        x_adv = x_adv.detach()

    return x_adv


# evaluate model under pgd attack
def pgd_eval_accuracy(
    model,
    loader,
    device,
    eps_px,
    alpha_px,
    steps,
    mean,
    std,
    random_start=True
):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # create pgd examples
        x_adv = pgd_attack_normalized_space(
            model,
            x,
            y,
            eps_px=eps_px,
            alpha_px=alpha_px,
            steps=steps,
            mean=mean,
            std=std,
            random_start=random_start
        )

        with torch.no_grad():
            pred = model(x_adv).argmax(1)

            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total


def main():
    # get settings from command line
    args = parse_args()

    # clean model name
    args.name = safe_name(args.name)

    # check parameter ranges
    if args.eps < 0 or args.eps > 255:
        raise ValueError("Epsilon must be between 0 and 255.")

    if args.alpha < 0 or args.alpha > 255:
        raise ValueError("Alpha must be between 0 and 255.")

    if args.steps < 1:
        raise ValueError("Steps must be at least 1.")

    if args.adv_weight < 0 or args.adv_weight > 1:
        raise ValueError("Adversarial weight must be between 0 and 1.")

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # paths
    DATA_DIR = ROOT / "data"
    SAVE_DIR = ROOT / "custom_train"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_CKPT = ROOT / "checkpoints" / "cnn_cifar10_best.pt"

    # training settings
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay

    # attack settings
    EPS_PX = args.eps / 255
    ALPHA_PX = args.alpha / 255
    STEPS = args.steps
    RANDOM_START = not args.no_random_start
    ADV_WEIGHT = args.adv_weight

    print("\nPGD adversarial training settings:")
    print(f"Model name: {args.name}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epsilon: {args.eps}/255")
    print(f"Alpha: {args.alpha}/255")
    print(f"Steps: {STEPS}")
    print(f"Random start: {RANDOM_START}")
    print(f"Adversarial weight: {ADV_WEIGHT}")
    print(f"Start from baseline: {not args.from_scratch}")

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # data transform
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load cifar-10 training set
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=tf
    )

    # load cifar-10 test set
    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=tf
    )

    # training data loader
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # test data loader
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # create model
    model = CIFAR_CNN().to(device)

    # optionally load baseline weights
    if not args.from_scratch and BASELINE_CKPT.exists():
        ckpt = torch.load(BASELINE_CKPT, map_location=device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded baseline weights from: {BASELINE_CKPT}")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded baseline state_dict from: {BASELINE_CKPT}")

    elif not args.from_scratch:
        print("Baseline checkpoint not found; training from scratch.")

    else:
        print("Training from scratch.")

    # optimiser
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss()

    best_pgd_acc = -1.0
    history = []

    # save paths
    last_path = SAVE_DIR / f"{args.name}_last.pt"
    best_path = SAVE_DIR / f"{args.name}_best.pt"
    hist_path = SAVE_DIR / f"{args.name}_history.json"

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        correct_clean = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # create pgd examples for training
            x_adv = pgd_attack_normalized_space(
                model,
                x,
                y,
                eps_px=EPS_PX,
                alpha_px=ALPHA_PX,
                steps=STEPS,
                mean=mean,
                std=std,
                random_start=RANDOM_START
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss_clean = loss_fn(logits_clean, y)
            loss_adv = loss_fn(logits_adv, y)

            # mix clean and adversarial loss
            loss = (1.0 - ADV_WEIGHT) * loss_clean + ADV_WEIGHT * loss_adv
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)

            with torch.no_grad():
                pred_clean = logits_clean.argmax(1)
                correct_clean += (pred_clean == y).sum().item()
                total += y.numel()

        train_loss = running_loss / total
        train_acc = correct_clean / total

        # evaluate clean accuracy
        clean_acc = accuracy(model, test_loader, device)

        # evaluate pgd accuracy
        pgd_acc = pgd_eval_accuracy(
            model,
            test_loader,
            device,
            eps_px=EPS_PX,
            alpha_px=ALPHA_PX,
            steps=STEPS,
            mean=mean,
            std=std,
            random_start=RANDOM_START
        )

        # store epoch results
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_clean_acc": round(clean_acc, 6),
            "test_pgd_acc": round(pgd_acc, 6),
            "eps_px": float(EPS_PX),
            "eps_255": float(args.eps),
            "alpha_px": float(ALPHA_PX),
            "alpha_255": float(args.alpha),
            "steps": int(STEPS),
            "random_start": bool(RANDOM_START),
            "adv_weight": float(ADV_WEIGHT),
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.1f}% | "
            f"test_clean={clean_acc*100:5.1f}% test_pgd={pgd_acc*100:5.1f}%"
        )

        # create checkpoint data
        checkpoint = {
            "model_name": args.name,
            "model_type": "pgd_adversarial",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "meta": {
                "mean": mean,
                "std": std,
                "eps_px": float(EPS_PX),
                "eps_255": float(args.eps),
                "alpha_px": float(ALPHA_PX),
                "alpha_255": float(args.alpha),
                "steps": int(STEPS),
                "random_start": bool(RANDOM_START),
                "adv_weight": float(ADV_WEIGHT),
                "from_scratch": bool(args.from_scratch),
            },
            "settings": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "eps_255": args.eps,
                "alpha_255": args.alpha,
                "steps": STEPS,
                "random_start": RANDOM_START,
                "adv_weight": ADV_WEIGHT,
                "from_scratch": args.from_scratch,
            }
        }

        # save last checkpoint
        torch.save(checkpoint, last_path)

        # save best checkpoint
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            torch.save(checkpoint, best_path)

            print(f"Saved BEST (by PGD acc): {best_path}")

        # save training history
        hist_path.write_text(json.dumps(history, indent=2))

    print("\nDone.")
    print("Best PGD accuracy:", f"{best_pgd_acc*100:.2f}%")
    print("Last:", last_path)
    print("Best:", best_path)
    print("History:", hist_path)


# start the script
if __name__ == "__main__":
    main()