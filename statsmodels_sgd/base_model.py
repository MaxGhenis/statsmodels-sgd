import torch.nn as nn
import torch.optim as optim
import numpy as np
from optimizers import SVRG  # SVRG optimizer
from .tools import (
    add_constant,
    calculate_standard_errors,
    calculate_t_p_values,
)

import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch installation path: {torch.__file__}")
except ImportError as e:
    print(f"Error importing torch: {e}")
    print("Installed packages:")
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
    )
    print(result.stdout)


class BaseModel(nn.Module):
    def __init__(
        self,
        n_features,
        learning_rate=0.01,
        epochs=1000,
        batch_size=32,
        clip_value=1.0,
        optimizers = 'sgd',
        n_inner=10, 
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.results_ = None
        # Initializing optimizer based on user choice
        if optimizer == 'svrg':
            self.optimizer = SVRG(lr=learning_rate, n_inner=n_inner)  # SVRG optimizer
        elif optimizer == 'sgd':
            self.optimizer = None  # Existing SGD logic or implement SGD class
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def forward(self, x):
        return self.linear(x)

def fit(self, X, y, sample_weight=None):
    """
    Fit the model to the training data.

    Parameters:
    -----------
    X : torch.Tensor
        Input features of shape (n_samples, n_features).
    y : torch.Tensor
        Target labels of shape (n_samples,).
    sample_weight : torch.Tensor, optional
        Sample weights for weighted loss calculation.

    Raises:
    -------
    ValueError: If input dimensions are inconsistent.
    """
    try:
        # Ensure the model is in training mode
        self.train()

        # Convert inputs to torch tensors if they are not already
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y for linear regression

        # Check input dimensions
        if X.size(0) != y.size(0):
            raise ValueError("Number of samples in X and y must be the same.")

        # Number of samples
        n_samples = X.size(0)

        # Initialize a placeholder for gradients
        grads = None

        for epoch in range(self.epochs):
            for i in range(0, n_samples, self.batch_size):
                # Get mini-batch
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]

                # Zero gradients
                self.linear.zero_grad()

                # Forward pass: Compute predicted y by passing batch_X to the model
                predictions = self.linear(batch_X)

                # Calculate loss (using mean squared error for regression)
                loss = nn.MSELoss()(predictions, batch_y)

                # Backward pass: Compute gradient of the loss with respect to model parameters
                loss.backward()

                # Compute the full gradients if using SVRG
                if isinstance(self.optimizer, SVRG):
                    # Collect full gradients if needed (can implement here)
                    # For now, use the gradients calculated from the backward pass
                    grads = [param.grad for param in self.linear.parameters()]

                # Update weights using the optimizer
                if isinstance(self.optimizer, SVRG):
                    full_grads = grads  # Placeholder for full gradients
                    self.optimizer.update(self.linear.parameters(), [param.grad for param in self.linear.parameters()], full_grads)
                else:
                    # If using standard SGD or another optimizer, implement its update method
                    for param in self.linear.parameters():
                        param.data -= self.learning_rate * param.grad.data
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item()}')

        self.results_ = predictions.detach().numpy()

    except Exception as e:
        print(f"An error occurred during training: {e}")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")

    def summary(self):
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.results_
