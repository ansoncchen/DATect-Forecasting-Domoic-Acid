"""
Simple Temporal Convolutional Network (TCN) regressor.
Uses TimeSeriesDataSet dataloaders for input batches.
"""

from __future__ import annotations

import torch
from torch import nn

from forecasting.torch_forecasting_adapter import make_dataloaders


class _TCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, kernel_size: int):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(channels, hidden_channels, kernel_size, padding=kernel_size - 1))
            layers.append(nn.ReLU())
            channels = hidden_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.network(x)
        # Use last timestep
        last = out[:, :, -1]
        return self.head(last).squeeze(-1)


class TCNRegressor:
    def __init__(
        self,
        max_epochs: int = 20,
        batch_size: int = 64,
        hidden_channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, dataset):
        train_loader, _ = make_dataloaders(dataset, batch_size=self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model based on first batch
        first = next(iter(train_loader))
        x, y = first
        encoder = x["encoder_cont"]  # (batch, encoder_len, features)
        in_channels = encoder.shape[-1]
        self.model = _TCN(in_channels, self.hidden_channels, self.num_layers, self.kernel_size).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.max_epochs):
            for x, y in train_loader:
                encoder = x["encoder_cont"].to(device).permute(0, 2, 1)
                target = y[0] if isinstance(y, (tuple, list)) else y
                target = target[:, 0].to(device)

                optimizer.zero_grad()
                preds = self.model(encoder)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, dataset):
        if self.model is None:
            raise RuntimeError("TCNRegressor model is not fit yet.")
        device = next(self.model.parameters()).device
        _, val_loader = make_dataloaders(dataset, batch_size=self.batch_size)
        self.model.eval()
        preds_all = []
        with torch.no_grad():
            for x, _y in val_loader:
                encoder = x["encoder_cont"].to(device).permute(0, 2, 1)
                preds = self.model(encoder)
                preds_all.append(preds.cpu())
        if preds_all:
            return torch.cat(preds_all, dim=0).numpy().reshape(-1)
        return torch.empty(0).numpy()
