import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils.image_args import get_image_path

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

def load_model(device):
    model = CIFAR_CNN().to(device)
    ckpt = torch.load("checkpoints/cnn_cifar10_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def predict_image(image_path, threshold=50.0, show_image=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = load_model(device)

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = probs.max(dim=0)

    conf_pct = conf.item() * 100
    top3_conf, top3_idx = torch.topk(probs, 3)

    if conf_pct < threshold:
        label = "N/A (low confidence)"
    else:
        label = classes[pred.item()]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {label} | max confidence: {conf_pct:.2f}% | threshold: {threshold:.1f}%")
    print("Top-3:")
    for c, i in zip(top3_conf.tolist(), top3_idx.tolist()):
        print(f"  {classes[i]}: {c*100:.2f}%")

    if show_image:
        title = f"Prediction: {label} ({conf_pct:.1f}%)"
        subtitle = f"Top-3: {classes[top3_idx[0]]} {top3_conf[0]*100:.1f}%, " \
                   f"{classes[top3_idx[1]]} {top3_conf[1]*100:.1f}%, " \
                   f"{classes[top3_idx[2]]} {top3_conf[2]*100:.1f}%"

        plt.figure(figsize=(7, 5))
        plt.imshow(image)
        plt.title(title)
        plt.xlabel(subtitle)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    import sys

    image_path = "images/deer.webp"  # default image

    if len(sys.argv) > 1:
        image_path = sys.argv[1]  # image selected from main.py

    predict_image(image_path, threshold=50.0, show_image=True)
