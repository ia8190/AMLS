import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


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


# create a pgd adversarial batch
def pgd_attack(model, x, y, eps, alpha, steps):
    model.eval()
    x_orig = x.detach()

    # start from random noise near the image
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
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
            grad_sign = x_adv.grad.sign()

            # move in direction that increases loss
            x_adv = x_adv + alpha * grad_sign

            # keep image inside epsilon range
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

        x_adv = x_adv.detach()

    return x_adv


# evaluate model under pgd attack
def evaluate_pgd(model, loader, device, eps_list, alpha, steps):
    results = []

    for eps in eps_list:
        correct = 0
        total = 0

        for x, y in tqdm(loader, desc=f"PGD eps={eps:.5f}", leave=False):
            x, y = x.to(device), y.to(device)

            # apply pgd if epsilon is greater than zero
            if eps > 0:
                x_adv = pgd_attack(
                    model,
                    x,
                    y,
                    eps=eps,
                    alpha=alpha,
                    steps=steps
                )
            else:
                x_adv = x

            with torch.no_grad():
                logits = model(x_adv)
                pred = logits.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        results.append((eps, acc))

    return results


def main():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # test transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load cifar-10 test set
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # load model
    model = CIFAR_CNN().to(device)
    ckpt = torch.load("checkpoints/cnn_cifar10_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # pgd epsilon values
    eps_list = [0.0, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

    # pgd settings
    steps = 10
    alpha = 2 / 255

    results = evaluate_pgd(
        model,
        testloader,
        device,
        eps_list,
        alpha=alpha,
        steps=steps
    )

    print("\nPGD Evaluation Results")
    print("----------------------------------------")
    print("   Epsilon | Accuracy (%)")
    print("----------------------------------------")

    # print each result
    for eps, acc in results:
        print(f"  {eps:8.5f} | {acc:12.2f}")


# start the script
if __name__ == "__main__":
    main()