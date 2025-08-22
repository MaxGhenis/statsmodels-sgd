"""
Differential privacy accounting and mechanisms for statsmodels-sgd.

This module provides privacy accounting tools for tracking privacy budgets
and computing privacy guarantees for DP-SGD training.
"""

from .accounting import (
    PrivacyAccountant,
    compute_epsilon,
    compute_rdp,
    get_privacy_spent,
)
from .mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    add_noise_to_gradient,
    clip_gradient,
)

__all__ = [
    "PrivacyAccountant",
    "compute_epsilon",
    "compute_rdp",
    "get_privacy_spent",
    "GaussianMechanism",
    "LaplaceMechanism",
    "add_noise_to_gradient",
    "clip_gradient",
]