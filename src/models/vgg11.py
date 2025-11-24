import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from cifar_loader import get_cifar10_loaders
"""
VGG11 implementation for CIFAR-10.
Includes:
- VGG11 model
- train_vgg()
- load_or_train_vgg()
- predict_vgg()
"""
# vgg11 model

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),

            # Block 4
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),

            # Block 5
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),

            # Block 6
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),

            # Block 7
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),

            # Block 8
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# training function

def train_vgg11(save_path="saved_models/vgg11.pth", epochs=10, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, classes = get_cifar10_loaders()

    model = VGG11().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} — Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print("VGG11 saved.")
    return model

def load_or_train_vgg11(path="saved_models/vgg11.pth"):
    if os.path.exists(path):
        print("Loading saved VGG11 model...")
        model = VGG11()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    else:
        print("Training VGG11 (this may take a few minutes)...")
        return train_vgg11(path)

def predict_vgg11(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    return np.array(all_labels), np.array(all_preds)

# evaluation function
def evaluate_vgg11(save_path="saved_models/vgg11.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader, _ = get_cifar10_loaders()

    model = VGG11().to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))

    y_true, y_pred = predict_vgg11(model, test_loader)

    accuracy = (y_true == y_pred).mean()
    print(f"VGG11 Test Accuracy = {accuracy:.4f}")

    return y_true, y_pred

# entry point
def run_vgg11():
    train_vgg11()
    evaluate_vgg11()