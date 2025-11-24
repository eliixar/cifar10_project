import torch
import torch.nn as nn
import torch.optim as optim
import os

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

def train_vgg(model, train_loader, epochs=10, lr=0.01, save_path="saved_models/vgg11.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  |  Loss = {total_loss:.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved VGG11 model to {save_path}")

    return model

# load or train model

def load_or_train_vgg(train_loader, epochs=10, lr=0.01):
    save_path = "saved_models/vgg11.pth"
    model = VGG11()

    if os.path.exists(save_path):
        print("Loading saved VGG11 model...")
        model.load_state_dict(torch.load(save_path))
        return model
    else:
        print("Training VGG11 CNN...")
        return train_vgg(model, train_loader, epochs, lr, save_path)

# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------

def predict_vgg(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            out = model(images)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())

    return preds
