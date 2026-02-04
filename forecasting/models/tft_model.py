"""
Temporal Fusion Transformer wrapper for regression.
"""

from __future__ import annotations

import torch

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer
    HAS_TFT = True
except ImportError:
    pl = None
    TemporalFusionTransformer = None
    HAS_TFT = False

from forecasting.torch_forecasting_adapter import make_dataloaders
from forecasting.models import get_accelerator_for_lightning


class TFTRegressor:
    """
    Wrapper around PyTorch Forecasting TFT with sklearn-like interface.
    Expects a TimeSeriesDataSet for fit/predict.
    """

    def __init__(
        self,
        max_epochs: int = 20,
        batch_size: int = 64,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        if not HAS_TFT:
            raise ImportError("pytorch-forecasting and pytorch-lightning are required for TFTRegressor.")
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, dataset):
        train_loader, val_loader = make_dataloaders(dataset, batch_size=self.batch_size)
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )

        accelerator = get_accelerator_for_lightning()
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, dataset):
        if self.model is None:
            raise RuntimeError("TFTRegressor model is not fit yet.")
        preds = self.model.predict(dataset, mode="prediction", return_x=False)
        return preds.detach().cpu().numpy().reshape(-1)
