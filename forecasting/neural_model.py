"""
Neural sequence models for DA forecasting.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import config


class DAForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class DAForecastGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]


@dataclass
class SequenceData:
    sequences: np.ndarray
    targets: np.ndarray
    target_dates: List[pd.Timestamp]


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "da",
    sequence_length: int = 8,
    forecast_horizon: int = 1
) -> SequenceData:
    """
    Create leak-free sequences by predicting t + forecast_horizon
    from the prior sequence_length timesteps.
    """
    sequences = []
    targets = []
    target_dates = []

    for site in df["site"].unique():
        site_df = df[df["site"] == site].sort_values("date").reset_index(drop=True)
        if len(site_df) < sequence_length + forecast_horizon:
            continue

        features = site_df[feature_cols].to_numpy(dtype=np.float32)
        target_values = site_df[target_col].to_numpy(dtype=np.float32)
        dates = site_df["date"].to_list()

        start_index = sequence_length - 1
        end_index = len(site_df) - forecast_horizon
        for idx in range(start_index, end_index):
            seq_start = idx - sequence_length + 1
            seq_end = idx + 1
            target_index = idx + forecast_horizon

            sequences.append(features[seq_start:seq_end])
            targets.append(target_values[target_index])
            target_dates.append(dates[target_index])

    sequences = np.array(sequences)
    targets = np.array(targets)

    if config.USE_LOG_TARGET_TRANSFORM:
        targets = np.log1p(targets)

    return SequenceData(sequences=sequences, targets=targets, target_dates=target_dates)


def build_sequence_model(model_type: str, input_size: int):
    if model_type == "lstm":
        return DAForecastLSTM(input_size=input_size)
    if model_type == "gru":
        return DAForecastGRU(input_size=input_size)
    raise ValueError(f"Unknown model_type: {model_type}")


def train_sequence_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "da",
    sequence_length: int = 8,
    forecast_horizon: int = 1,
    model_type: str = "lstm",
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3
) -> Tuple[nn.Module, dict]:
    """
    Train an LSTM/GRU model with a temporal train/validation split.
    """
    sequence_data = prepare_sequences(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )

    if len(sequence_data.sequences) == 0:
        raise ValueError("No sequences available for training")

    # Temporal split based on target date order
    sorted_indices = np.argsort(sequence_data.target_dates)
    sequences = sequence_data.sequences[sorted_indices]
    targets = sequence_data.targets[sorted_indices]

    split_idx = int(len(sequences) * 0.8)
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]

    train_ds = SequenceDataset(train_sequences, train_targets)
    val_ds = SequenceDataset(val_sequences, val_targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_sequence_model(model_type=model_type, input_size=train_sequences.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "sequence_length": sequence_length,
        "forecast_horizon": forecast_horizon
    }
    return model, history
