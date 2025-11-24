import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
MLP architectures with different depths.
Each hidden layer has 512 units.
"""

def make_mlp(depth):
    """
    Create an MLP with:
    - Input: 50
    - Hidden layers: depth (each with 512 units)
    - Output: 10 classes
    """
    layers = []
    input_dim = 50

    for _ in range(depth):
        layers.append(nn.Linear(input_dim, 512))
        layers.append(nn.ReLU())
        input_dim = 512

    # final output layer
    layers.append(nn.Linear(512, 10))

    return nn.Sequential(*layers)


"""
Mini-batch training function for all MLP models.
This is required so the MLP actually learns correctly.
"""
def train_mlp(model, X_train, y_train, epochs=20, lr=0.01, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # convert numpy → torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    # DataLoader (CRITICAL for learning)
    loader = torch.utils.data.DataLoader(
        list(zip(X_train_t, y_train_t)),
        batch_size=batch_size,
        shuffle=True
    )

    # training loop
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return model


"""
Train an MLP with the specified depth and evaluate accuracy.
Returns accuracy and predictions.
"""
def train_and_eval_depth(depth, X_train, y_train, X_test, y_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build the model for this depth
    model = make_mlp(depth)

    # train using mini-batch SGD
    model = train_mlp(model, X_train, y_train, epochs=20)

    # evaluate
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()

    accuracy = (preds == y_test).mean()
    return accuracy, preds
