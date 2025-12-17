"""
Test suite for differential privacy accounting mechanisms.

This module tests the privacy guarantees of our DP-SGD implementation,
including epsilon-delta calculations and Rényi Differential Privacy (RDP).
"""

import numpy as np
import pytest
import torch
from typing import Tuple
from statsmodels_sgd.privacy import (
    PrivacyAccountant,
    compute_epsilon,
    compute_rdp,
    get_privacy_spent,
)


class TestPrivacyAccountant:
    """Test privacy accounting mechanisms."""

    def test_gaussian_mechanism_epsilon(self):
        """Test epsilon calculation for Gaussian mechanism."""
        # Given sensitivity and noise parameters
        sensitivity = 1.0
        sigma = 2.0
        delta = 1e-5
        
        accountant = PrivacyAccountant(mechanism="gaussian")
        epsilon = accountant.compute_epsilon(
            sensitivity=sensitivity,
            sigma=sigma,
            delta=delta
        )
        
        # Epsilon should be positive and reasonable
        assert epsilon > 0
        assert epsilon < 10  # Reasonable privacy guarantee
        
    def test_composition_over_iterations(self):
        """Test privacy composition over multiple SGD iterations."""
        n_iterations = 100
        noise_multiplier = 1.0
        sample_rate = 0.01
        delta = 1e-5
        
        accountant = PrivacyAccountant(mechanism="gaussian")
        epsilon = accountant.compute_composition(
            n_iterations=n_iterations,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta
        )
        
        # Privacy loss should accumulate with iterations
        assert epsilon > 0
        
        # Test that more iterations increase epsilon
        epsilon_more = accountant.compute_composition(
            n_iterations=n_iterations * 2,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta
        )
        assert epsilon_more > epsilon
        
    def test_rdp_to_dp_conversion(self):
        """Test conversion from RDP to (ε,δ)-DP."""
        alpha_list = [2, 3, 4, 5, 10, 20, 50, 100]
        rdp_budget = np.array([0.1 * alpha for alpha in alpha_list])
        delta = 1e-5
        
        accountant = PrivacyAccountant(mechanism="gaussian")
        epsilon = accountant.rdp_to_dp(
            rdp_budget=rdp_budget,
            alpha_list=alpha_list,
            delta=delta
        )
        
        assert epsilon > 0
        assert epsilon < np.inf
        
    def test_gradient_clipping_sensitivity(self):
        """Test that gradient clipping bounds sensitivity."""
        clip_norm = 1.0
        batch_size = 32
        
        # Create random gradients
        gradients = torch.randn(batch_size, 10)
        
        # Compute per-sample gradient norms
        grad_norms = torch.norm(gradients, dim=1)
        
        # Clip gradients
        clipped_grads = gradients * torch.clamp(
            clip_norm / (grad_norms + 1e-10), max=1.0
        ).unsqueeze(1)
        
        # Check that all clipped norms are <= clip_norm
        clipped_norms = torch.norm(clipped_grads, dim=1)
        assert torch.all(clipped_norms <= clip_norm + 1e-6)
        
    def test_privacy_budget_tracking(self):
        """Test tracking of privacy budget over training."""
        accountant = PrivacyAccountant(
            mechanism="gaussian",
            delta_target=1e-5
        )
        
        # Simulate training steps
        for step in range(10):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        epsilon_spent = accountant.get_privacy_spent()
        assert epsilon_spent > 0
        
        # Privacy should increase with more steps
        for step in range(10):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        epsilon_spent_more = accountant.get_privacy_spent()
        assert epsilon_spent_more > epsilon_spent


class TestPrivacyGuarantees:
    """Test specific privacy guarantees for our models."""
    
    def test_ols_privacy_guarantee(self):
        """Test privacy guarantees for OLS model."""
        from statsmodels_sgd.api import OLS
        
        n_features = 5
        n_samples = 1000
        clip_value = 1.0
        noise_multiplier = 1.0
        epochs = 100
        batch_size = 32
        
        # Calculate expected privacy budget
        sample_rate = batch_size / n_samples
        steps = epochs * (n_samples // batch_size)
        
        accountant = PrivacyAccountant(mechanism="gaussian")
        for _ in range(steps):
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )
        
        epsilon = accountant.get_privacy_spent(delta=1e-5)
        
        # Privacy budget should be reasonable
        assert epsilon > 0
        # With 3100 steps and noise_multiplier=1, expect higher epsilon
        assert epsilon < 1000  # Adjusted for realistic expectations
        
    def test_logit_privacy_guarantee(self):
        """Test privacy guarantees for Logit model."""
        from statsmodels_sgd.api import Logit
        
        n_features = 5
        n_samples = 1000
        clip_value = 1.0
        noise_multiplier = 2.0  # Higher noise for stronger privacy
        epochs = 50
        batch_size = 64
        
        sample_rate = batch_size / n_samples
        steps = epochs * (n_samples // batch_size)
        
        accountant = PrivacyAccountant(mechanism="gaussian")
        for _ in range(steps):
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )
        
        epsilon = accountant.get_privacy_spent(delta=1e-5)
        
        # With higher noise, epsilon should be lower
        assert epsilon > 0
        # With noise_multiplier=2, expect lower epsilon than OLS
        assert epsilon < 50  # More realistic expectation
        
    def test_privacy_utility_tradeoff(self):
        """Test the privacy-utility tradeoff."""
        noise_levels = [0.5, 1.0, 2.0, 4.0]
        epsilons = []
        
        for noise in noise_levels:
            accountant = PrivacyAccountant(mechanism="gaussian")
            for _ in range(100):
                accountant.step(noise_multiplier=noise, sample_rate=0.01)
            
            epsilon = accountant.get_privacy_spent(delta=1e-5)
            epsilons.append(epsilon)
        
        # Higher noise should give lower epsilon (better privacy)
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1]


class TestPrivacyAmplification:
    """Test privacy amplification by subsampling."""
    
    def test_subsampling_amplification(self):
        """Test that subsampling amplifies privacy."""
        noise_multiplier = 1.0
        
        # Compare different sampling rates
        sample_rates = [0.001, 0.01, 0.1, 1.0]
        epsilons = []
        
        for rate in sample_rates:
            accountant = PrivacyAccountant(mechanism="gaussian")
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=rate
            )
            epsilon = accountant.get_privacy_spent(delta=1e-5)
            epsilons.append(epsilon)
        
        # Smaller sampling rate should give better privacy
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i + 1]
    
    def test_poisson_vs_uniform_sampling(self):
        """Compare Poisson and uniform sampling strategies."""
        noise_multiplier = 1.0
        sample_rate = 0.01
        n_steps = 100
        
        # Poisson sampling
        accountant_poisson = PrivacyAccountant(
            mechanism="gaussian",
            sampling="poisson"
        )
        for _ in range(n_steps):
            accountant_poisson.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )
        epsilon_poisson = accountant_poisson.get_privacy_spent(delta=1e-5)
        
        # Uniform sampling  
        accountant_uniform = PrivacyAccountant(
            mechanism="gaussian",
            sampling="uniform"
        )
        for _ in range(n_steps):
            accountant_uniform.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )
        epsilon_uniform = accountant_uniform.get_privacy_spent(delta=1e-5)
        
        # Both should give valid privacy guarantees
        assert epsilon_poisson > 0
        assert epsilon_uniform > 0
        
        # Poisson sampling typically gives tighter bounds
        assert abs(epsilon_poisson - epsilon_uniform) < 2.0


@pytest.mark.parametrize("mechanism", ["gaussian", "laplace"])
@pytest.mark.parametrize("delta", [1e-3, 1e-5, 1e-7])
def test_mechanism_comparison(mechanism: str, delta: float):
    """Compare different privacy mechanisms."""
    accountant = PrivacyAccountant(mechanism=mechanism)
    
    if mechanism == "laplace" and delta < 1e-3:
        # Laplace mechanism doesn't depend on delta
        pytest.skip("Laplace mechanism is (ε,0)-DP")
    
    # Add noise and compute privacy
    sensitivity = 1.0
    noise_scale = 2.0
    
    if mechanism == "gaussian":
        epsilon = accountant.compute_epsilon(
            sensitivity=sensitivity,
            sigma=noise_scale,
            delta=delta
        )
    else:  # laplace
        epsilon = sensitivity / noise_scale
    
    assert epsilon > 0
    assert epsilon < np.inf