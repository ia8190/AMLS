import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# cifar-10 class names
classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# use no normalisation for viewing
transform = transforms.ToTensor()

# load cifar-10 test set
testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

# show selected test images
for idx in [0, 10, 100, 999, 1234]:
    img, label = testset[idx]
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Index {idx} – Label: {classes[label]}")
    plt.axis("off")
    plt.show()