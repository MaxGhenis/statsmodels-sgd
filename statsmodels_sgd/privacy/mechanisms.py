"""
Privacy mechanisms for adding noise and clipping gradients.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple


def clip_gradient(
    gradient: torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0
) -> torch.Tensor:
    """
    Clip gradient to have maximum norm.
    
    Args:
        gradient: Gradient tensor
        max_norm: Maximum allowed norm
        norm_type: Type of norm (default: 2.0 for L2)
        
    Returns:
        Clipped gradient
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    
    grad_norm = torch.norm(gradient, p=norm_type)
    clip_factor = torch.clamp(max_norm / (grad_norm + 1e-10), max=1.0)
    
    return gradient * clip_factor


def add_noise_to_gradient(
    gradient: torch.Tensor,
    noise_multiplier: float,
    max_norm: float,
    mechanism: str = "gaussian"
) -> torch.Tensor:
    """
    Add calibrated noise to gradient for differential privacy.
    
    Args:
        gradient: Gradient tensor
        noise_multiplier: Noise scale relative to sensitivity
        max_norm: Maximum gradient norm (sensitivity)
        mechanism: Type of noise ("gaussian" or "laplace")
        
    Returns:
        Noisy gradient
    """
    noise_scale = noise_multiplier * max_norm
    
    if mechanism == "gaussian":
        noise = torch.randn_like(gradient) * noise_scale
    elif mechanism == "laplace":
        # Laplace distribution: scale parameter b = noise_scale
        noise = torch.distributions.Laplace(0, noise_scale).sample(gradient.shape)
        if gradient.is_cuda:
            noise = noise.cuda()
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return gradient + noise


class GaussianMechanism:
    """Gaussian mechanism for differential privacy."""
    
    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float
    ):
        """
        Initialize Gaussian mechanism.
        
        Args:
            sensitivity: L2 sensitivity
            epsilon: Privacy parameter
            delta: Privacy parameter
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        
        # Compute required noise scale
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        """Compute noise standard deviation for target privacy."""
        # From the Gaussian mechanism analysis
        # σ ≥ sensitivity * sqrt(2 * log(1.25/δ)) / ε
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise(
        self,
        value: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Add Gaussian noise to value.
        
        Args:
            value: Value to add noise to
            
        Returns:
            Noisy value
        """
        if isinstance(value, torch.Tensor):
            noise = torch.randn_like(value) * self.sigma
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.normal(0, self.sigma, value.shape)
            return value + noise
        else:
            noise = np.random.normal(0, self.sigma)
            return value + noise
    
    def get_privacy_guarantee(self) -> Tuple[float, float]:
        """Get (ε,δ)-DP guarantee."""
        return self.epsilon, self.delta


class LaplaceMechanism:
    """Laplace mechanism for differential privacy."""
    
    def __init__(
        self,
        sensitivity: float,
        epsilon: float
    ):
        """
        Initialize Laplace mechanism.
        
        Args:
            sensitivity: L1 sensitivity
            epsilon: Privacy parameter
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        
        # Laplace scale parameter
        self.scale = sensitivity / epsilon
    
    def add_noise(
        self,
        value: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Add Laplace noise to value.
        
        Args:
            value: Value to add noise to
            
        Returns:
            Noisy value
        """
        if isinstance(value, torch.Tensor):
            noise = torch.distributions.Laplace(0, self.scale).sample(value.shape)
            if value.is_cuda:
                noise = noise.cuda()
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.laplace(0, self.scale, value.shape)
            return value + noise
        else:
            noise = np.random.laplace(0, self.scale)
            return value + noise
    
    def get_privacy_guarantee(self) -> Tuple[float, float]:
        """Get (ε,δ)-DP guarantee."""
        return self.epsilon, 0.0  # Laplace gives (ε,0)-DP