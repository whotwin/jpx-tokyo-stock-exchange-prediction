# iTransformer Training Functions

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import device


class FeatureProjector(nn.Module):
    """
    特征投影层：将每个股票的多个特征投影到 d_model 维度

    输入: (batch, seq_len, num_stocks, num_features)
    输出: (batch, seq_len, num_stocks, dim)
    """
    def __init__(self, num_features, dim):
        super().__init__()
        self.projection = nn.Linear(num_features, dim)

    def forward(self, x):
        # x: (batch, seq_len, num_stocks, num_features)
        batch, seq_len, num_stocks, num_features = x.shape
        # Reshape to (batch * seq_len, num_stocks, num_features)
        x = x.view(batch * seq_len, num_stocks, num_features)
        # Project features: (batch * seq_len, num_stocks, dim)
        x = self.projection(x)
        # Reshape back to (batch, seq_len, num_stocks, dim)
        x = x.view(batch, seq_len, num_stocks, -1)
        return x


def masked_mse_loss(pred, target, mask):
    """
    计算 masked MSE loss，只计算非 mask 位置（有效位置）的损失

    pred: (batch, num_stocks)
    target: (batch, num_stocks)
    mask: (batch, num_stocks), 1 for valid, 0 for invalid
    """
    # Create squared error
    se = (pred - target) ** 2

    # Apply mask and compute mean only on valid positions
    masked_se = se * mask

    # Sum and divide by number of valid positions
    num_valid = mask.sum()
    if num_valid > 0:
        loss = masked_se.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device)

    return loss


def train_itransformer_model(model, train_loader, epochs=10, lr=0.001, projector=None):
    """Train iTransformer model with masked loss for NaN handling.

    Args:
        model: iTransformer model
        train_loader: DataLoader for training
        epochs: number of training epochs
        lr: learning rate
        projector: Optional FeatureProjector to project features from num_features to dim
    """
    model.to(device)
    if projector is not None:
        projector.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    if projector is not None:
        projector.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            # batch_X: (batch, seq_length, num_stocks, num_features)
            # batch_y: (batch, num_stocks)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Apply feature projection if provided
            if projector is not None:
                batch_X = projector(batch_X)
                # Now batch_X is (batch, seq_len, num_stocks, dim)

            # Reshape to 3D for iTransformer: (batch, seq_len, num_stocks * dim)
            batch_size, seq_len, num_stocks, dim = batch_X.shape
            batch_X = batch_X.reshape(batch_size, seq_len, num_stocks * dim)

            batch_y = batch_y.to(device)

            # Create mask for valid (non-NaN) target values
            mask = ~torch.isnan(batch_y)
            mask = mask.float()

            # If no valid values in batch, skip
            if mask.sum() == 0:
                continue

            # Replace NaN in target with 0 for model input (mask will handle loss)
            batch_y_filled = torch.where(torch.isnan(batch_y), torch.zeros_like(batch_y), batch_y)

            optimizer.zero_grad()
            pred = model(batch_X)

            # Skip if predictions are NaN
            if torch.isnan(pred).any():
                continue

            # pred shape: (batch, pred_length, num_variates)
            # Take only the first prediction step and all variates
            pred = pred[:, 0, :]

            # Compute masked loss
            loss = masked_mse_loss(pred, batch_y_filled, mask)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0 and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.6f}")

    return model


def cross_sectional_normalize(X, num_features=1, eps=1e-8):
    """
    对每个时间步，在所有股票间做z-score归一化（横截面归一化）
    支持多特征：每个特征独立进行横截面归一化

    X: 4D case: (batch, seq_len, num_stocks, num_features)
       or 3D case: (batch, seq_len, num_variates)

    num_features: 特征数量
                  如果 X 是 4D，这个参数用于指定特征维度大小
                  如果 X 是 3D 且 num_features>1，将 num_variates 分解为 (num_stocks, num_features)

    这会移除市场整体影响，让模型专注于学习股票间的相对差异
    """
    if len(X.shape) == 4:
        # 4D case: (batch, seq_len, num_stocks, num_features)
        batch_size, seq_len, num_stocks, num_feat = X.shape
        X_normalized = np.zeros_like(X)

        for t in range(seq_len):
            x_t = X[:, t, :, :]  # (batch, num_stocks, num_features)

            # 每个特征独立在股票间归一化
            for f in range(num_feat):
                x_t_feature = x_t[:, :, f]  # (batch, num_stocks)
                mean_f = x_t_feature.mean(axis=1, keepdims=True)  # (batch, 1)
                std_f = x_t_feature.std(axis=1, keepdims=True)    # (batch, 1)
                std_f = np.maximum(std_f, eps)
                X_normalized[:, t, :, f] = (x_t_feature - mean_f) / std_f

        return X_normalized
    else:
        # 3D case: (batch, seq_len, num_variates)
        batch_size, seq_len, num_variates = X.shape
        X_normalized = np.zeros_like(X)

        if num_features == 1:
            # 原始行为：在所有变量间归一化
            for t in range(seq_len):
                x_t = X[:, t, :]  # (batch, num_variates)
                mean_t = x_t.mean(axis=1, keepdims=True)  # (batch, 1)
                std_t = x_t.std(axis=1, keepdims=True)    # (batch, 1)
                std_t = np.maximum(std_t, eps)
                X_normalized[:, t, :] = (x_t - mean_t) / std_t
        else:
            # 多特征：每个特征独立在股票间归一化
            num_stocks = num_variates // num_features

            for t in range(seq_len):
                x_t = X[:, t, :]  # (batch, num_stocks × num_features)

                # 重塑为 (batch, num_stocks, num_features)
                x_t_reshaped = x_t.reshape(batch_size, num_stocks, num_features)

                # 每个特征独立在股票间归一化
                for f in range(num_features):
                    x_t_feature = x_t_reshaped[:, :, f]  # (batch, num_stocks)
                    mean_f = x_t_feature.mean(axis=1, keepdims=True)  # (batch, 1)
                    std_f = x_t_feature.std(axis=1, keepdims=True)    # (batch, 1)
                    std_f = np.maximum(std_f, eps)
                    x_t_reshaped[:, :, f] = (x_t_feature - mean_f) / std_f

                # 重塑回 (batch, num_stocks × num_features)
                X_normalized[:, t, :] = x_t_reshaped.reshape(batch_size, -1)

        return X_normalized
