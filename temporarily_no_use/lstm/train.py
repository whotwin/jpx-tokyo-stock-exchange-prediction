# LSTM Training Functions

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .config import device, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LEARNING_RATE


def train_lstm_model(model, train_loader, epochs=None, lr=None):
    """Train LSTM model."""
    if epochs is None:
        epochs = LSTM_EPOCHS
    if lr is None:
        lr = LSTM_LEARNING_RATE

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Skip batches with NaN targets
            if torch.isnan(batch_y).any():
                continue

            optimizer.zero_grad()
            pred = model(batch_X)

            # Skip if predictions are NaN
            if torch.isnan(pred).any():
                continue

            loss = criterion(pred, batch_y)

            # Skip if loss is NaN
            if torch.isnan(loss):
                continue

            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        if num_batches > 0 and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def create_train_loader(X_train, y_train, batch_size=None):
    """Create DataLoader for training."""
    if batch_size is None:
        batch_size = LSTM_BATCH_SIZE

    # Sample training data if too large
    max_train_samples = 300000
    if len(X_train) > max_train_samples:
        step = len(X_train) // max_train_samples
        indices = np.arange(0, len(X_train), step)[:max_train_samples]
        X_train_sampled = torch.FloatTensor(X_train[indices])
        y_train_sampled = torch.FloatTensor(y_train[indices])
    else:
        X_train_sampled = torch.FloatTensor(X_train)
        y_train_sampled = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(X_train_sampled, y_train_sampled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader


def create_test_loader(X_test, batch_size=None):
    """Create DataLoader for testing/prediction."""
    if batch_size is None:
        batch_size = LSTM_BATCH_SIZE

    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return test_loader
