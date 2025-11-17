import torchvision
from torchvision import transforms

"""
Function to load testing and training images from CIFAR10.
500 train images are added to each class of CIFAR10, and
100 test images are added to each class. Returns both data
sets in the form of lists containing tuples of (image,class label.)
"""
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

    # add 500 images to each train set class (5000 total images in train_data)
    for img, label in train_set:
        if train_count[label] < 500:
            train_data.append((img,label)) # append as a tuple
            train_count[label] +=1 # update counter

    # do the same for the test set but with 100 images
    for img, label in test_set:
        if test_count[label] < 100:
            test_data.append((img, label))
            test_count[label] += 1

    # return the train and test sets with required number of images
    return train_data, test_data
