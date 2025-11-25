import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from cifar_loader import get_cifar10_loaders


# base vgg conv block
def conv_block(in_c, out_c, pool=False):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


# flexible depth vgg model
class VGG_Flexible(nn.Module):
    def __init__(self, depth_factor=1.0, num_classes=10):
        super().__init__()

        # original VGG11 channel sequence
        base_channels = [64, 128, 256, 256, 512, 512, 512, 512]

        # how many conv outputs to keep
        keep = int(len(base_channels) * depth_factor)
        keep = max(2, min(keep, len(base_channels)))
        selected_channels = base_channels[:keep]

        layers = []
        in_c = 3

        # apply same pooling pattern as VGG11 (pool after conv 2,4,6,8)
        for i, out_c in enumerate(selected_channels):
            pool = (i in [1, 3, 5, 7])
            layers.extend(conv_block(in_c, out_c, pool))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        # dynamic flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            f = self.features(dummy)
            flatten_dim = f.numel()

        # classifier always has the same structure
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


# train
def train_vgg_depth(depth_factor, save_path, epochs=1, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, _ = get_cifar10_loaders()

    model = VGG_Flexible(depth_factor).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"\nTraining VGG (depth={depth_factor}) for {epochs} epoch")

    for ep in range(epochs):
        model.train()
        MAX_STEPS = 120  # limit time

        for step, (imgs, labels) in enumerate(train_loader):
            if step >= MAX_STEPS:
                break

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        print(f"[depth={depth_factor}] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model




# evaluate flexible-depth VGG
def evaluate_vgg_depth(model):
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



# run depth experiment
def run_vgg_depth_experiment():
    os.makedirs("saved_models", exist_ok=True)

    depth_factors = [0.5, 1.0, 1.5]
    results = []

    for df in depth_factors:
        path = f"saved_models/vgg_depth_{df}.pth"

        if os.path.exists(path):
            print(f"\nLoading saved VGG depth={df}")
            model = VGG_Flexible(depth_factor=df)
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            model = train_vgg_depth(df, path)

        acc, y_true, y_pred = evaluate_vgg_depth(model)
        print(f"Accuracy (depth={df}) = {acc:.4f}")

        results.append((df, acc, y_true, y_pred))

    return results
