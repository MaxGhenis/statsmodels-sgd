"""
Base model for differentially private statistical models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any

from .tools import (
    add_constant,
    calculate_standard_errors,
    calculate_t_p_values,
)
from .privacy import PrivacyAccountant, clip_gradient, add_noise_to_gradient


class BaseModel(nn.Module):
    """
    Base class for differentially private models using SGD.
    
    This class provides the foundation for DP-SGD training with
    gradient clipping and noise addition for privacy guarantees.
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
        Initialize the base model.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            batch_size: Batch size for training
            clip_value: Maximum gradient norm (L2)
            noise_multiplier: Noise scale relative to clip_value
            delta: Target delta for (ε,δ)-DP
            track_privacy: Whether to track privacy budget
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.track_privacy = track_privacy
        
        # Initialize privacy accountant
        if self.track_privacy:
            self.privacy_accountant = PrivacyAccountant(
                mechanism="gaussian",
                delta_target=delta
            )
        else:
            self.privacy_accountant = None
        
        self.results_ = None
        self.n_samples_ = None
        self.privacy_spent_ = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.linear(x)
    
    def compute_per_sample_gradients(
        self, 
        X_batch: torch.Tensor, 
        y_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample gradients for privacy-preserving training.
        
        Args:
            X_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Per-sample gradients
        """
        # This needs to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit the model with differential privacy.
        
        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_privacy_guarantee(self) -> Dict[str, float]:
        """
        Get the privacy guarantee of the trained model.
        
        Returns:
            Dictionary with epsilon and delta values
        """
        if self.privacy_accountant is None:
            return {"epsilon": float("inf"), "delta": 0.0}
        
        epsilon = self.privacy_accountant.get_privacy_spent(self.delta)
        return {"epsilon": epsilon, "delta": self.delta}
    
    def summary(self) -> Dict[str, Any]:
        """
        Get model summary including privacy guarantees.
        
        Returns:
            Dictionary with model results and privacy info
        """
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")
        
        summary = self.results_.copy()
        
        # Add privacy information
        if self.track_privacy:
            privacy_info = self.get_privacy_guarantee()
            summary["privacy_epsilon"] = privacy_info["epsilon"]
            summary["privacy_delta"] = privacy_info["delta"]
            summary["noise_multiplier"] = self.noise_multiplier
            summary["clip_value"] = self.clip_value
        
        return summary