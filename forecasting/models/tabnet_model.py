"""
TabNet regression wrapper.
"""

from __future__ import annotations

import numpy as np

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    TabNetRegressor = None
    HAS_TABNET = False


class TabNetRegressorWrapper:
    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        max_epochs: int = 50,
        batch_size: int = 256,
    ):
        if not HAS_TABNET:
            raise ImportError("pytorch-tabnet is required for TabNetRegressorWrapper.")
        self.model = TabNetRegressor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
        )
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.model.fit(
            X_train=X,
            y_train=y,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            virtual_batch_size=max(32, self.batch_size // 8),
            drop_last=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X)
        return preds.reshape(-1)
