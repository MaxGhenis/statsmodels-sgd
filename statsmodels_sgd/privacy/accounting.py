"""
Privacy accounting for differential privacy.

Implements privacy accounting using Rényi Differential Privacy (RDP)
and conversion to (ε,δ)-DP following Mironov (2017) and Abadi et al. (2016).
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from scipy import special
from scipy.optimize import minimize_scalar


def compute_epsilon(
    sensitivity: float, sigma: float, delta: float
) -> float:
    """
    Compute epsilon for Gaussian mechanism.
    
    Args:
        sensitivity: L2 sensitivity of the function
        sigma: Standard deviation of Gaussian noise
        delta: Target delta for (ε,δ)-DP
        
    Returns:
        epsilon: Privacy parameter
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("Delta must be in (0, 1)")
    
    # Using the analytic Gaussian mechanism formula
    # ε = (sensitivity / sigma) * sqrt(2 * log(1.25 / delta))
    epsilon = (sensitivity / sigma) * np.sqrt(2 * np.log(1.25 / delta))
    return epsilon


def compute_rdp(
    alpha: float, 
    sigma: float,
    sensitivity: float = 1.0
) -> float:
    """
    Compute RDP for Gaussian mechanism.
    
    Args:
        alpha: RDP order
        sigma: Noise standard deviation
        sensitivity: L2 sensitivity
        
    Returns:
        RDP epsilon at order alpha
    """
    if alpha <= 1:
        raise ValueError("Alpha must be > 1")
    
    # RDP for Gaussian mechanism: ε_α = α * (sensitivity^2) / (2 * sigma^2)
    return alpha * (sensitivity ** 2) / (2 * sigma ** 2)


def rdp_to_dp(
    rdp_budget: np.ndarray,
    alpha_list: List[float],
    delta: float
) -> float:
    """
    Convert RDP to (ε,δ)-DP.
    
    Args:
        rdp_budget: Array of RDP values at different orders
        alpha_list: List of RDP orders
        delta: Target delta
        
    Returns:
        epsilon: Best (ε,δ)-DP guarantee
    """
    if delta <= 0:
        raise ValueError("Delta must be positive")
    
    epsilons = []
    for rdp, alpha in zip(rdp_budget, alpha_list):
        # Conversion formula: ε = rdp + log(1/δ) / (α - 1)
        eps = rdp + np.log(1 / delta) / (alpha - 1)
        epsilons.append(eps)
    
    # Return the tightest bound
    return min(epsilons)


def compute_rdp_sample(
    alpha: float,
    sigma: float,
    sample_rate: float,
    sensitivity: float = 1.0
) -> float:
    """
    Compute RDP with subsampling amplification.
    
    Args:
        alpha: RDP order
        sigma: Noise standard deviation  
        sample_rate: Sampling probability
        sensitivity: L2 sensitivity
        
    Returns:
        RDP with subsampling
    """
    if sample_rate == 0:
        return 0
    
    if sample_rate == 1.0:
        return compute_rdp(alpha, sigma, sensitivity)
    
    # Subsampling amplification for RDP (Wang et al., 2019)
    # Using tighter bounds from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
    rdp_no_sampling = compute_rdp(alpha, sigma, sensitivity)
    
    # For small sampling rates, use the tight amplification bound
    if sample_rate < 0.1:
        # Approximate amplification: RDP ≈ sample_rate^2 * RDP_no_sampling
        # This is valid for small sampling rates
        amplified_rdp = sample_rate ** 2 * rdp_no_sampling * (2 * alpha)
    else:
        # For larger sampling rates, use a more conservative bound
        # Based on Mironov et al. (2019)
        log_moment = np.log(
            (1 - sample_rate) + sample_rate * np.exp(rdp_no_sampling)
        )
        amplified_rdp = log_moment
    
    return amplified_rdp


class PrivacyAccountant:
    """
    Track privacy budget during DP-SGD training.
    
    This class implements privacy accounting using RDP composition
    and converts to (ε,δ)-DP guarantees.
    """
    
    def __init__(
        self,
        mechanism: str = "gaussian",
        delta_target: float = 1e-5,
        sampling: str = "poisson"
    ):
        """
        Initialize privacy accountant.
        
        Args:
            mechanism: Privacy mechanism ("gaussian" or "laplace")
            delta_target: Target delta for (ε,δ)-DP conversion
            sampling: Sampling strategy ("poisson" or "uniform")
        """
        self.mechanism = mechanism
        self.delta_target = delta_target
        self.sampling = sampling
        
        # RDP orders to track
        self.alpha_list = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 
                          32, 64, 128, 256, 512, 1024]
        
        # Initialize RDP budget
        self.rdp_budget = np.zeros(len(self.alpha_list))
        self.steps = 0
        
    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        sensitivity: float = 1.0
    ):
        """
        Account for one training step.
        
        Args:
            noise_multiplier: Ratio of noise std to sensitivity
            sample_rate: Batch sampling rate
            sensitivity: L2 sensitivity (usually clip_norm)
        """
        self.steps += 1
        
        # Compute RDP for this step at each order
        for i, alpha in enumerate(self.alpha_list):
            if self.mechanism == "gaussian":
                rdp_step = compute_rdp_sample(
                    alpha=alpha,
                    sigma=noise_multiplier * sensitivity,
                    sample_rate=sample_rate,
                    sensitivity=sensitivity
                )
            elif self.mechanism == "laplace":
                # Laplace mechanism RDP (simplified)
                rdp_step = sensitivity / (noise_multiplier * sensitivity) 
            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")
            
            # Compose with previous steps (RDP composition: simple addition)
            self.rdp_budget[i] += rdp_step
    
    def get_privacy_spent(self, delta: Optional[float] = None) -> float:
        """
        Get current privacy budget as (ε,δ)-DP.
        
        Args:
            delta: Target delta (uses default if None)
            
        Returns:
            epsilon: Current privacy spent
        """
        if delta is None:
            delta = self.delta_target
            
        if self.steps == 0:
            return 0.0
        
        return rdp_to_dp(
            rdp_budget=self.rdp_budget,
            alpha_list=self.alpha_list,
            delta=delta
        )
    
    def compute_epsilon(
        self,
        sensitivity: float,
        sigma: float,
        delta: float
    ) -> float:
        """
        Compute epsilon for a single application of the mechanism.
        
        Args:
            sensitivity: L2 sensitivity
            sigma: Noise standard deviation
            delta: Target delta
            
        Returns:
            epsilon: Privacy guarantee
        """
        if self.mechanism == "gaussian":
            return compute_epsilon(sensitivity, sigma, delta)
        elif self.mechanism == "laplace":
            # Laplace is (ε,0)-DP
            return sensitivity / sigma
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def compute_composition(
        self,
        n_iterations: int,
        noise_multiplier: float,
        sample_rate: float,
        delta: float,
        sensitivity: float = 1.0
    ) -> float:
        """
        Compute privacy for composed iterations.
        
        Args:
            n_iterations: Number of iterations
            noise_multiplier: Noise multiplier
            sample_rate: Sampling rate
            delta: Target delta
            sensitivity: L2 sensitivity
            
        Returns:
            epsilon: Total privacy cost
        """
        # Reset and compute
        old_budget = self.rdp_budget.copy()
        old_steps = self.steps
        
        self.rdp_budget = np.zeros(len(self.alpha_list))
        self.steps = 0
        
        for _ in range(n_iterations):
            self.step(noise_multiplier, sample_rate, sensitivity)
        
        epsilon = self.get_privacy_spent(delta)
        
        # Restore old state
        self.rdp_budget = old_budget
        self.steps = old_steps
        
        return epsilon
    
    def rdp_to_dp(
        self,
        rdp_budget: np.ndarray,
        alpha_list: List[float],
        delta: float
    ) -> float:
        """
        Convert RDP to (ε,δ)-DP.
        
        Args:
            rdp_budget: RDP values
            alpha_list: RDP orders
            delta: Target delta
            
        Returns:
            epsilon: Best DP guarantee
        """
        return rdp_to_dp(rdp_budget, alpha_list, delta)


def get_privacy_spent(
    accountant: PrivacyAccountant,
    delta: Optional[float] = None
) -> float:
    """
    Helper function to get privacy spent from accountant.
    
    Args:
        accountant: Privacy accountant instance
        delta: Target delta
        
    Returns:
        epsilon: Privacy spent
    """
    return accountant.get_privacy_spent(delta)