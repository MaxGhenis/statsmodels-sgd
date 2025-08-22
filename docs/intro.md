# Differentially Private Statistical Models via Stochastic Gradient Descent

```{epigraph}
The first implementation providing both differentially private regression coefficients **and** standard errors.
```

## Abstract

We present **statsmodels-sgd**, a Python library that implements differentially private statistical models using stochastic gradient descent (DP-SGD) with a crucial innovation: it provides both privacy-preserving coefficient estimates **and** adjusted standard errors for statistical inference. While existing differential privacy libraries focus solely on point estimates, our work addresses the critical gap in privacy-preserving statistical inference, enabling hypothesis testing and confidence interval construction under differential privacy guarantees.

## Key Contributions

1. **First implementation** combining differential privacy with statistical inference (standard errors, t-statistics, p-values)
2. **Statsmodels-compatible API** making DP accessible to statisticians and economists
3. **Rigorous privacy accounting** using Rényi Differential Privacy with conversion to (ε,δ)-DP
4. **Standard error adjustment** accounting for noise injected by the privacy mechanism
5. **Comprehensive evaluation** of privacy-utility tradeoffs through extensive simulations

## Why This Matters

Existing differential privacy libraries have a critical limitation:

| Library | DP Coefficients | Standard Errors | Statistical Inference |
|---------|----------------|-----------------|----------------------|
| IBM Diffprivlib | ✅ | ❌ | ❌ |
| Google Opacus | ✅ | ❌ | ❌ |
| TensorFlow Privacy | ✅ | ❌ | ❌ |
| **statsmodels-sgd** | ✅ | ✅ | ✅ |

Without standard errors, researchers cannot:
- Test hypotheses about regression coefficients
- Construct confidence intervals
- Assess statistical significance
- Perform valid inference on private data

## Quick Example

```python
import statsmodels_sgd.api as sm_sgd
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 5)
true_coef = np.array([1, 2, 3, 4, 5])
y = X @ true_coef + np.random.randn(1000) * 0.5

# Fit DP-OLS with privacy guarantee
model = sm_sgd.OLS(
    n_features=6,  # 5 features + intercept
    noise_multiplier=1.0,  # Controls privacy level
    clip_value=1.0,  # Gradient clipping threshold
    delta=1e-5  # Target δ for (ε,δ)-DP
)
model.fit(X, y)

# Get results with standard errors
results = model.summary()
print(f"Privacy guarantee: ε = {results['privacy_epsilon']:.2f}")
print(f"Coefficients: {results['params']}")
print(f"Standard errors: {results['std_errors']}")
print(f"P-values: {results['p_values']}")
```

## Installation

```bash
pip install statsmodels-sgd
```

Or install from source:

```bash
git clone https://github.com/PolicyEngine/statsmodels-sgd.git
cd statsmodels-sgd
pip install -e ".[dev,docs]"
```

## Navigation

```{tableofcontents}
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ghenis2024statsmodels_sgd,
  author = {Ghenis, Max},
  title = {statsmodels-sgd: Differentially Private Statistical Models with Standard Errors},
  year = {2024},
  url = {https://github.com/PolicyEngine/statsmodels-sgd}
}
```