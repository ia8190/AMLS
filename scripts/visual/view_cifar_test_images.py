import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# No normalization for viewing
transform = transforms.ToTensor()

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

for idx in [0, 10, 100, 999, 1234]:
    img, label = testset[idx]

    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Index {idx} – Label: {classes[label]}")
    plt.axis("off")
    plt.show()
