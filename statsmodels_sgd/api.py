from .regression.linear_model import OLS
from .regression.logit_dp import LogitDP

# Use LogitDP as the default Logit for consistency with OLS having DP support
Logit = LogitDP

# Make commonly used objects available at api level
__all__ = ["OLS", "Logit", "LogitDP"]
