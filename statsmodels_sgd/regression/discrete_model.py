import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statsmodels.api as sm
import pandas as pd


class Logit(nn.Module):
    def __init__(
        self,
        n_features,
        learning_rate=0.1,
        epochs=2000,
        batch_size=32,
        clip_value=5.0,
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.results_ = None
        self.add_constant = True

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, X, y, sample_weight=None, add_constant=True):
        self.add_constant = add_constant
        if add_constant:
            X = sm.add_constant(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        if sample_weight is not None:
            sample_weight = torch.tensor(
                sample_weight, dtype=torch.float32
            ).reshape(-1, 1)

        # Correct initialization of weights
        self.linear = nn.Linear(X.shape[1], 1, bias=False)
        init_weights = torch.zeros(
            1, X.shape[1]
        )  # Changed shape from (X.shape[1], 1)
        self.linear.weight = nn.Parameter(init_weights)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50
        )

        prev_loss = float("inf")
        patience_counter = 0
        best_weights = None
        min_loss = float("inf")

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.weighted_binary_cross_entropy(y_pred, y, sample_weight)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_weights = self.linear.weight.clone().detach()

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_value)
            optimizer.step()
            scheduler.step(loss)

            if abs(loss.item() - prev_loss) < 1e-6:
                patience_counter += 1
                if patience_counter >= 100:
                    break
            else:
                patience_counter = 0
            prev_loss = loss.item()

        if best_weights is not None:
            self.linear.weight.data = best_weights

        self.results_ = {"coef": self.linear.weight.detach().numpy().flatten()}

    def weighted_binary_cross_entropy(self, y_pred, y_true, weights=None):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        if weights is None:
            weights = torch.ones_like(y_true)
        loss = -(
            weights
            * (
                y_true * torch.log(y_pred)
                + (1 - y_true) * torch.log(1 - y_pred)
            )
        ).mean()
        return loss

    def predict(self, X):
        if self.add_constant:
            X = sm.add_constant(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self(X).numpy()

    def summary(self):
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")
        return pd.DataFrame(
            {
                "coef": self.results_["coef"],
                "std err": [np.nan] * len(self.results_["coef"]),
                "z": [np.nan] * len(self.results_["coef"]),
                "P>|z|": [np.nan] * len(self.results_["coef"]),
                "[0.025": [np.nan] * len(self.results_["coef"]),
                "0.975]": [np.nan] * len(self.results_["coef"]),
            },
            index=["const"]
            + [f"x{i}" for i in range(1, len(self.results_["coef"]))],
        )
