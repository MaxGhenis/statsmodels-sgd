# Differentially Private Statistical Models via Stochastic Gradient Descent

```{epigraph}
A practical Python library for differentially private regression with statistical inference.
```

## Abstract

We present **statsmodels-sgd**, a Python library that implements differentially private statistical models using stochastic gradient descent (DP-SGD). The library provides both privacy-preserving coefficient estimates **and** adjusted standard errors for statistical inference, with a familiar statsmodels-like API. While theoretical approaches to DP inference exist, most lack accessible implementations. statsmodels-sgd fills this gap by providing a practical, well-tested solution for researchers who need both privacy guarantees and valid inference.

## Key Contributions

1. **Accessible DP inference** with standard errors, t-statistics, and p-values in Python
2. **Statsmodels-compatible API** making DP accessible to statisticians and economists
3. **Rigorous privacy accounting** using Rényi Differential Privacy with conversion to (ε,δ)-DP
4. **Empirically calibrated standard errors** achieving ~95% confidence interval coverage
5. **Comprehensive documentation** including comparison with alternative approaches

## Why This Matters

Most widely-used DP libraries focus on ML applications without statistical inference {cite}`opacus2021,diffprivlib2019`:

| Library | DP Coefficients | Standard Errors | Ease of Use |
|---------|----------------|-----------------|-------------|
| IBM Diffprivlib {cite}`diffprivlib2019` | ✅ | ❌ | High |
| Meta Opacus {cite}`opacus2021` | ✅ | ❌ | Medium |
| svinfer {cite}`svinfer2024` | ✅ | ✅ | Medium |
| PrivacyUnbiased {cite}`privacyunbiased2024` | ✅ | ✅ | Medium (R) |
| **statsmodels-sgd** | ✅ | ✅ | High |

While principled methods for DP inference exist (see {doc}`chapters/related_work`), they often require:
- Complex analytical derivations {cite}`sheffet2017differentially,evans2024linked`
- Expensive MCMC computation {cite}`bernstein2019bayesian`
- R rather than Python {cite}`privacyunbiased2024`
- Deep DP expertise {cite}`dwork2014algorithmic`

statsmodels-sgd provides a **practical, Python-native solution** with a familiar statsmodels API.

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