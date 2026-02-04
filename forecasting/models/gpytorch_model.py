"""
GPyTorch regression wrapper.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import gpytorch
    HAS_GP = True
except ImportError:
    torch = None
    gpytorch = None
    HAS_GP = False


if HAS_GP:
    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
else:
    _ExactGPModel = None


class GPyTorchRegressor:
    def __init__(self, training_iters: int = 75, learning_rate: float = 0.1):
        if not HAS_GP:
            raise ImportError("gpytorch and torch are required for GPyTorchRegressor.")
        self.training_iters = training_iters
        self.learning_rate = learning_rate
        self.model = None
        self.likelihood = None
        self.x_mean = None
        self.x_std = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None:
            self.x_mean = x.mean(axis=0, keepdims=True)
            self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        return (x - self.x_mean) / self.x_std

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self._standardize(np.asarray(X, dtype=np.float32))
        ys = np.asarray(y, dtype=np.float32)

        train_x = torch.from_numpy(Xs)
        train_y = torch.from_numpy(ys)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ExactGPModel(train_x, train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.training_iters):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GPyTorchRegressor model is not fit yet.")
        Xs = self._standardize(np.asarray(X, dtype=np.float32))
        test_x = torch.from_numpy(Xs)

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(test_x))
            mean = preds.mean.cpu().numpy()
        return mean.reshape(-1)
