import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from cifar_loader import get_cifar10_loaders


# make a VGG block with variable kernel size
def conv_block(in_c, out_c, k=3, pool=False):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=k//2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


# flexible-kernel VGG
class VGG_Kernel(nn.Module):
    def __init__(self, kernel_size=3, num_classes=10):
        super().__init__()

        # fixed channel pattern from VGG11
        channels = [64, 128, 256, 256, 512, 512, 512, 512]

        layers = []
        in_c = 3

        # pool after 2,4,6,8 like original VGG
        for i, out_c in enumerate(channels):
            pool = (i in [1, 3, 5, 7])
            layers.extend(conv_block(in_c, out_c, k=kernel_size, pool=pool))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        # compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            f = self.features(dummy)
            flatten_dim = f.numel()

        # auto-adjust FC layer size
        fc_hidden = 1024 if flatten_dim < 8000 else 4096

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, fc_hidden),
            nn.ReLU(), nn.Dropout(0.5),

            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(), nn.Dropout(0.5),

            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# fast training
def train_vgg_kernel(kernel_size, save_path, epochs=1, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, _ = get_cifar10_loaders()

    model = VGG_Kernel(kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"\nTraining VGG (kernel={kernel_size}) for {epochs} epochs")

    MAX_STEPS = 100   #  global
    for ep in range(epochs):
        model.train()

        for step, (imgs, labels) in enumerate(train_loader):
            if step >= MAX_STEPS:
                break   # stop early = FAST

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print(f"[kernel={kernel_size}] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model



# evaluation
def evaluate_vgg_kernel(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader, _ = get_cifar10_loaders()

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = (y_true == y_pred).mean()

    return acc, y_true, y_pred


# main
def run_vgg_kernel_experiment():
    os.makedirs("saved_models", exist_ok=True)

    kernel_sizes = [3, 5, 7]
    results = []

    for k in kernel_sizes:
        path = f"saved_models/vgg_kernel_{k}.pth"

        if os.path.exists(path):
            print(f"\nLoading saved VGG kernel={k}")
            model = VGG_Kernel(kernel_size=k)
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            model = train_vgg_kernel(k, save_path=path)

        acc, y_true, y_pred = evaluate_vgg_kernel(model)
        print(f"Kernel {k}×{k}: Accuracy = {acc:.4f}")

        results.append((k, acc, y_true, y_pred))

    return results
