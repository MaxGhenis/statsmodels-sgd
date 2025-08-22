"""
Ordinary Least Squares regression with differential privacy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
import statsmodels.api as sm

from ..base_model import BaseModel
from ..tools import (
    calculate_standard_errors,
    calculate_t_p_values,
)
from ..privacy import clip_gradient, add_noise_to_gradient


class OLS(BaseModel):
    """
    Ordinary Least Squares regression with differential privacy via SGD.
    
    This implementation uses stochastic gradient descent with gradient
    clipping and noise addition to provide differential privacy guarantees.
    """
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 32,
        clip_value: float = 1.0,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        track_privacy: bool = True,
    ):
        """
        Initialize OLS model with DP-SGD.
        
        Args:
            n_features: Number of features (including constant if added)
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            batch_size: Batch size for training
            clip_value: Maximum gradient norm for clipping
            noise_multiplier: Scale of noise relative to clip_value
            delta: Target delta for (ε,δ)-DP
            track_privacy: Whether to track privacy budget
        """
        super().__init__(
            n_features=n_features,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            clip_value=clip_value,
            noise_multiplier=noise_multiplier,
            delta=delta,
            track_privacy=track_privacy,
        )
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        add_constant: bool = True
    ) -> "OLS":
        """
        Fit the OLS model using DP-SGD.
        
        Args:
            X: Training features
            y: Training targets
            sample_weight: Optional sample weights
            add_constant: Whether to add constant term
            
        Returns:
            Self for method chaining
        """
        # Add constant if requested
        if add_constant:
            X = sm.add_constant(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        if sample_weight is not None:
            weight_tensor = torch.FloatTensor(sample_weight).reshape(-1, 1)
        else:
            weight_tensor = torch.ones_like(y_tensor)
        
        # Store sample size for privacy accounting
        self.n_samples_ = len(X)
        
        # Initialize optimizer
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            # Create random batches
            indices = torch.randperm(len(X))
            
            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                weight_batch = weight_tensor[batch_indices]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute weighted loss
                losses = weight_batch * (predictions - y_batch) ** 2
                loss = losses.mean()
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping and noise for DP
                with torch.no_grad():
                    for param in self.parameters():
                        if param.grad is not None:
                            # Clip gradient
                            param.grad = clip_gradient(
                                param.grad,
                                self.clip_value
                            )
                            
                            # Add noise for privacy
                            param.grad = add_noise_to_gradient(
                                param.grad,
                                self.noise_multiplier,
                                self.clip_value,
                                mechanism="gaussian"
                            )
                
                # Update weights
                optimizer.step()
                
                # Track privacy budget
                if self.track_privacy:
                    sample_rate = self.batch_size / self.n_samples_
                    self.privacy_accountant.step(
                        noise_multiplier=self.noise_multiplier,
                        sample_rate=sample_rate,
                        sensitivity=self.clip_value
                    )
        
        # Calculate statistics
        with torch.no_grad():
            # Get coefficients
            all_coef = self.linear.weight.data.numpy().flatten()
            intercept = self.linear.bias.data.numpy()[0]
            
            # Arrange params based on whether constant was added
            if add_constant:
                # Skip the first coefficient (for the constant column) and combine it with bias
                actual_intercept = intercept + all_coef[0]  # Combine bias with constant's weight
                params = np.concatenate([[actual_intercept], all_coef[1:]])
            else:
                params = np.concatenate([[intercept], all_coef])
            
            # Calculate residuals and standard errors
            y_pred = self.predict(X)
            residuals = y - y_pred.flatten()
            
            # Get weights and bias for standard error calculation
            all_weights = self.linear.weight.data.numpy().flatten()
            bias = self.linear.bias.data.numpy()[0]
            
            # Calculate standard errors accounting for DP noise
            # Note: X should be without constant for this function
            if add_constant:
                X_no_const = X[:, 1:]  # Remove the constant column
                weights = all_weights[1:]  # Remove the weight for constant (it's in bias)
            else:
                X_no_const = X
                weights = all_weights
                
            std_errors = calculate_standard_errors(
                X_no_const, y, weights, bias, 
                is_logit=False, sample_weight=sample_weight
            )
            
            # Adjust standard errors for DP noise
            if self.track_privacy:
                # Approximate adjustment based on noise level
                noise_adjustment = np.sqrt(1 + self.noise_multiplier ** 2)
                std_errors = std_errors * noise_adjustment
            
            t_values, p_values = calculate_t_p_values(params, std_errors)
            
            # Store results
            self.results_ = {
                "params": params,
                "std_errors": std_errors,
                "t_values": t_values,
                "p_values": p_values,
                "residuals": residuals,
            }
            
            # Add privacy information
            if self.track_privacy:
                privacy_info = self.get_privacy_guarantee()
                self.results_["privacy_epsilon"] = privacy_info["epsilon"]
                self.results_["privacy_delta"] = privacy_info["delta"]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Input features (should include constant term if used in training)
            
        Returns:
            Predictions
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.forward(X_tensor)
            return predictions.numpy()