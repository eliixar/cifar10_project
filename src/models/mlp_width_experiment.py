import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import numpy as np

"""
MLP architecture used for the width experiment.
Structure matches assignment:

    Input: 50
    Hidden 1: H units + ReLU
    Hidden 2: H units + BatchNorm + ReLU
    Output: 10 classes

This file trains, saves, loads, and evaluates width-based MLPs.
"""


class MLPWidth(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(50, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_width(model, X_train, y_train, epochs=20, lr=0.01, batch=64):
    """Mini-batch training for width MLPs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # convert numpy → torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

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
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def load_or_train_width(H, X_train, y_train):
    """Loads model if exists, otherwise trains and saves it."""
    os.makedirs("saved_models", exist_ok=True)

    path = f"saved_models/mlp_width_H_{H}.pt"

    if os.path.exists(path):
        print(f"Loading saved MLP Width H={H} model...")
        return torch.load(path)

    print(f"Training MLP Width H={H} and saving model...")

    model = MLPWidth(H)
    model = train_mlp_width(model, X_train, y_train, epochs=5) # avoid long runtimes
    torch.save(model, path)

    return model


def predict_width(model, X_test):
    """Runs prediction with trained width-MLP."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()
