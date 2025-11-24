import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from cifar_loader import get_cifar10_loaders



#  base VGG11 blocks (reusable for deeper / shallower models)
def conv_block(in_c, out_c, pool=False):
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


# ---------------------------------------------------------
#  Generate VGG with adjustable depth
#  depth_factor = 0.5 → shallower
#  depth_factor = 1   → normal VGG11
#  depth_factor = 1.5 → deeper
# ---------------------------------------------------------
class VGG_Flexible(nn.Module):
    def __init__(self, depth_factor=1.0, num_classes=10):
        super().__init__()

        # base channel sizes
        channels = [64, 128, 256, 256, 512, 512, 512, 512]

        # adjust depth
        keep = int(len(channels) * depth_factor)
        keep = max(2, min(keep, len(channels)))  # keep at least 2 layers

        selected_channels = channels[:keep]

        layers = []
        in_c = 3

        # build convolutional sequence
        for i, out_c in enumerate(selected_channels):
            pool = (i in [1, 3, 5, 7])  # same pooling pattern as VGG11
            layers.extend(conv_block(in_c, out_c, pool))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        # classifier always the same
        self.classifier = nn.Sequential(
            nn.Linear(in_c, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# train a flexible-depth VGG
def train_vgg_depth(depth_factor, save_path, epochs=5, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, _ = get_cifar10_loaders()

    model = VGG_Flexible(depth_factor=depth_factor).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for ep in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print(f"[Depth {depth_factor}] Epoch {ep+1}/{epochs} Loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model


# evaluate
def evaluate_vgg_depth(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader, _ = get_cifar10_loaders()

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = (y_true == y_pred).mean()
    return accuracy, y_true, y_pred


# experiment runner
def run_vgg_depth_experiment():
    os.makedirs("saved_models", exist_ok=True)

    depth_factors = [0.5, 1.0, 1.5]  # shallower, normal, deeper
    results = []

    for df in depth_factors:
        path = f"saved_models/vgg_depth_{df}.pth"

        if os.path.exists(path):
            print(f"Loading VGG depth={df}")
            model = VGG_Flexible(df)
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            print(f"Training VGG depth={df}")
            model = train_vgg_depth(df, path)

        acc, y_true, y_pred = evaluate_vgg_depth(model)
        print(f"Accuracy at depth_factor {df}: {acc:.4f}")

        results.append((df, acc, y_true, y_pred))

    return results
