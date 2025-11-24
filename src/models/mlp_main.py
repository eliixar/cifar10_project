import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np


"""
Implements the 3-layer MLP.
Architecture:
    Linear(50 → 512) → ReLU
    Linear(512 → 512) → BatchNorm → ReLU
    Linear(512 → 10)

Optimizer: SGD(momentum=0.9)
Loss: CrossEntropyLoss
"""


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(model, X_train, y_train, epochs=20, lr=0.01, batch=64):
    """
    Train the MLP on PCA-reduced CIFAR10 (50 features).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Convert numpy → torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch,
        shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return model


def predict_mlp(model, X_test):
    """
    Predict classes using a trained MLP.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()


def train_base_mlp(X_train, y_train):
    """
    Build and train the base 3-layer MLP.
    Returns trained model.
    """
    model = MLP()
    model = train_mlp(model, X_train, y_train, epochs=20)
    return model


def load_base_mlp(X_train, y_train, X_test, y_test):
    """
    Loads a saved base MLP if it exists.
    Otherwise trains it, saves it, and returns predictions.
    """

    path = "saved_models/base_mlp.pkl"

    # Load if exists
    if os.path.exists(path):
        print("Loading saved Base MLP model...")
        model = joblib.load(path)
    else:
        print("Training Base MLP and saving...")
        model = train_base_mlp(X_train, y_train)
        joblib.dump(model, path)

    # Predict
    preds = predict_mlp(model, X_test)
    return model, preds
