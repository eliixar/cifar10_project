import torch
import torchvision # get the images from here
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Loads the CIFAR-10 training and test sets as image tensors.
    Used for VGG11 since it requires raw images and not feature vectors.
    Returns:
        train_loader
        test_loader
        class_names  (list of class labels)
    """

    # standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
    ])

    # download + load training data
    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False, # training false for test set
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_set.classes  # list of 10 labels

    return train_loader, test_loader, class_names
