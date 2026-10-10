"""
Model architectures for Shokri et al. (2017) Membership Inference.

Provides:
- TargetTabularMLP: 1-hidden-layer MLP with Tanh activation (Section VI-B & VIII-B)
- TargetCNN: 2 conv + maxpool + FC 128 + Tanh for CIFAR-10 (Section VI-B)
- AttackModel: 1-hidden-layer MLP with 64 units and ReLU (Section VI-D)
- Training helper functions with L2 regularization, Dropout, Early Stopping, and DP-SGD (Opacus).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TargetTabularMLP(nn.Module):
    """
    Target / Shadow model for tabular datasets (Purchase-100, Texas-100, Adult).
    As described in Section VI-B:
    - One hidden layer (default 128 units, or 256/1000 for mitigation experiments)
    - Tanh activation
    - Softmax output (logits returned for temperature scaling)
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits."""
        h = torch.tanh(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Returns softmax probabilities scaled by temperature."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
        return probs


class TargetCNN(nn.Module):
    """
    Standard CNN for CIFAR-10 as described in Section VI-B:
    Two convolution & max pooling layers plus a fully connected layer
    of size 128 and SoftMax. Tanh activations.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x8x8
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
        return probs


class AttackModel(nn.Module):
    """
    Attack model as described in Section VI-D:
    A fully connected neural network with one hidden layer of size 64
    with ReLU activation functions and binary output (membership probability).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(-1)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    lr_decay: float = 1e-7,
    patience: int = 0,
    use_dp: bool = False,
    dp_target_epsilon: float = 5.0,
    dp_target_delta: float = 1e-5,
    dp_max_grad_norm: float = 1.0,
    device: str = "cpu"
) -> dict:
    """
    Trains a classification model using PyTorch.
    Supports standard training, L2 regularization (weight_decay),
    Early Stopping (if patience > 0 and val data provided),
    and DP-SGD (using Opacus PrivacyEngine if use_dp=True).
    """
    device = torch.device(device)
    model.to(device)

    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    if use_dp:
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                target_epsilon=dp_target_epsilon,
                target_delta=dp_target_delta,
                epochs=epochs,
                max_grad_norm=dp_max_grad_norm,
            )
        except Exception as e:
            print(f"Warning: Opacus DP initialization failed ({e}), falling back to non-private training.")
            use_dp = False

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "dp_epsilon": None}
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        if lr_decay > 0 and not use_dp:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (1.0 + lr_decay * epoch * len(dataloader))

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_y = torch.tensor(y_val, dtype=torch.long).to(device)
                val_out = model(val_x)
                val_loss = criterion(val_out, val_y).item()
                val_preds = torch.argmax(val_out, dim=-1)
                val_acc = (val_preds == val_y).float().mean().item()
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if best_weights is not None:
                                model.load_state_dict(best_weights)
                            break

    if use_dp and privacy_engine is not None:
        try:
            history["dp_epsilon"] = privacy_engine.get_epsilon(delta=dp_target_delta)
        except Exception:
            history["dp_epsilon"] = dp_target_epsilon

    return history


def train_attack_model(
    attack_model: AttackModel,
    X_attack: np.ndarray,
    y_attack: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu"
) -> AttackModel:
    """
    Trains a binary attack classifier (membership prediction).
    X_attack: probability vectors (or features) from shadow models
    y_attack: 1 for member (in), 0 for non-member (out)
    """
    device = torch.device(device)
    attack_model.to(device)

    tensor_x = torch.tensor(X_attack, dtype=torch.float32)
    tensor_y = torch.tensor(y_attack, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        attack_model.train()
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = attack_model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    attack_model.eval()
    return attack_model
