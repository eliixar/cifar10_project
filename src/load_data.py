import torchvision
from torchvision import transforms

def load_cifar10():

    transform = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
    test_set = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)
    # cifar10 has 10 classes, initialize 10 zeroes to count
    train_count = [0] * 10
    test_count = [0] * 10

    train_data = []
    test_data = []

    # add 500 images to the train set
    for img, label in train_set:
        if train_count[label] < 500:
            train_data.append(img,label)
            train_count[label] +=1 # update counter

    # do the same for the test set but with 100 images
    for img, label in test_set:
        if test_data[label] < 100:
            test_data.append((img, label))
            test_data[label] += 1

    # return the train and test sets with required number of images
    return train_data, test_data