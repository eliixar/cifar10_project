import torchvision
from torchvision import transforms

def load_cifar10():

    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
    testset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)
    # cifar10 has 10 classes, initialize 10 zeroes to count
    train_count = [0] * 10
    test_count = [0] * 10

    train_subset = []
    test_subset = []