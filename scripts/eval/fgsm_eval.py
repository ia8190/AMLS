import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Model (same as training)

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

# FGSM attack
def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)

    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + epsilon * grad_sign

    return x_adv.detach()

# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def evaluate_fgsm(model, loader, device, epsilon):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if epsilon > 0:
            x = fgsm_attack(model, x, y, epsilon)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # CIFAR-10 normalization (same as training)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_tfms
    )
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Load model
    model = CIFAR_CNN().to(device)
    ckpt = torch.load("checkpoints/cnn_cifar10_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # FGSM epsilons 
    epsilons = [0.0, 2/255, 4/255, 8/255, 16/255]

    print("\nFGSM Evaluation Results")
    print("-" * 40)
    print(f"{'Epsilon':>10} | {'Accuracy (%)':>12}")
    print("-" * 40)

    for eps in epsilons:
        acc = evaluate_fgsm(model, testloader, device, eps)
        print(f"{eps:10.5f} | {acc*100:12.2f}")

    print("-" * 40)

if __name__ == "__main__":
    main()
