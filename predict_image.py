import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# cifar-10 class names
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# cnn model used for cifar-10 prediction
class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

        # final classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # forward pass through the model
    def forward(self, x):
        return self.classifier(self.features(x))


# load a saved model checkpoint
def load_model(device, model_path):
    model = CIFAR_CNN().to(device)

    ckpt = torch.load(model_path, map_location=device)

    # support both checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    return model


# predict the class of one image
def predict_image(
    image_path,
    model_path="checkpoints/cnn_cifar10_best.pt",
    threshold=50.0,
    show_image=True
):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print(f"Using image: {image_path}")
    print(f"Using model: {model_path}")

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # resize and normalise image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    model = load_model(device, model_path)

    # open image and convert to rgb
    image = Image.open(image_path).convert("RGB")

    # add batch dimension
    x = transform(image).unsqueeze(0).to(device)

    # make prediction without training gradients
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = probs.max(dim=0)

    conf_pct = conf.item() * 100

    # get top 3 predictions
    top3_conf, top3_idx = torch.topk(probs, 3)

    # reject prediction if confidence is too low
    if conf_pct < threshold:
        label = "N/A (low confidence)"
    else:
        label = classes[pred.item()]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {label} | max confidence: {conf_pct:.2f}% | threshold: {threshold:.1f}%")
    print("Top-3:")

    for c, i in zip(top3_conf.tolist(), top3_idx.tolist()):
        print(f"  {classes[i]}: {c*100:.2f}%")

    # show the image with prediction result
    if show_image:
        title = f"Prediction: {label} ({conf_pct:.1f}%)"

        subtitle = (
            f"Top-3: {classes[top3_idx[0]]} {top3_conf[0]*100:.1f}%, "
            f"{classes[top3_idx[1]]} {top3_conf[1]*100:.1f}%, "
            f"{classes[top3_idx[2]]} {top3_conf[2]*100:.1f}%"
        )

        plt.figure(figsize=(7, 5))
        plt.imshow(image)
        plt.title(title)
        plt.xlabel(subtitle)
        plt.axis("off")
        plt.show()


# run prediction from the command line
if __name__ == "__main__":
    image_path = "images/deer.webp"
    model_path = "checkpoints/cnn_cifar10_best.pt"

    # allow image path from command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    # allow model path from command line
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    predict_image(
        image_path,
        model_path=model_path,
        threshold=50.0,
        show_image=True
    )