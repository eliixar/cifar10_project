import torch
import torch.nn as nn
import torch.optim as optim
import os

# mlp width experiment model
class MLPWidth(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 10)
        )

    def forward(self, x):
        return self.net(x)


# train and save
def train_width(H, X_train, y_train, path, epochs=5, lr=0.01, batch=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPWidth(H).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch,
        shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # save ONLY weights (this avoids all pickle issues)
    torch.save(model.state_dict(), path)

    return model


# load model weights
def load_or_train_width(H, X_train, y_train):
    os.makedirs("saved_models", exist_ok=True)
    path = f"saved_models/mlp_width_{H}.pth"

    if os.path.exists(path):
        print(f"Loading saved MLP Width H={H} weights...")
        model = MLPWidth(H)
        model.load_state_dict(torch.load(path, weights_only=False))
        return model

    print(f"Training MLP Width H={H} and saving weights...")
    return train_width(H, X_train, y_train, path)


# predict
def predict_width(model, X_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds
