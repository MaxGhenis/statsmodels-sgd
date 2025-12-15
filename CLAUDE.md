# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

statsmodels-sgd is a Python package that reimplements statsmodels regression models using stochastic gradient descent (SGD) with gradient clipping for differential privacy. It provides a statsmodels-like API while using PyTorch as the backend for automatic differentiation and optimization.

## Development Commands

### Installation and Setup

```bash
# Install development dependencies
python -m pip install --upgrade pip
python -m pip install pytest black isort flake8

# Install PyTorch (CPU-only version used in CI)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install package in development mode
python -m pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest statsmodels_sgd/tests/test_ols.py
pytest statsmodels_sgd/tests/test_logit.py

# Run with coverage
pytest --cov=statsmodels_sgd
```

### Linting and Formatting

```bash
# Lint with flake8 (strict - stops on syntax errors)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Lint with flake8 (warnings only)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code with black
black statsmodels_sgd/

# Sort imports with isort
isort statsmodels_sgd/
```

## Architecture

### Core Design Pattern

All models follow a consistent three-layer architecture:

1. **API Layer** (`api.py`): Exports public-facing classes (`OLS`, `Logit`)
2. **Model Layer** (`regression/`): Implements specific regression models
3. **Base Layer** (`base_model.py`): Provides shared functionality (currently minimal, models inherit from `nn.Module` directly)

### Model Implementation Pattern

Each model (OLS, Logit) is implemented as a PyTorch `nn.Module` with:

- **Constructor**: Defines hyperparameters (learning_rate, epochs, batch_size, clip_value) and a single `nn.Linear` layer
- **fit()**: Implements SGD training loop with gradient clipping via `torch.nn.utils.clip_grad_value_()`
- **predict()**: Returns predictions using the trained linear layer
- **forward()**: Defines the forward pass (standard PyTorch pattern)

### Key Architectural Decisions

**Gradient Clipping for Differential Privacy**: Every model implements gradient clipping using PyTorch's `clip_grad_value_()`. The `clip_value` parameter controls the maximum gradient magnitude, which is essential for differential privacy guarantees.

**Sample Weighting**: Both OLS and Logit support sample weights through custom loss functions:
- OLS uses `weighted_mse_loss()` with weights applied element-wise
- Logit uses weighted binary cross-entropy

**Statsmodels Compatibility**: The API intentionally mimics statsmodels for easy migration:
- Similar class names (`OLS`, `Logit`)
- Similar method signatures (`fit()`, `predict()`)
- Uses statsmodels `add_constant()` for adding intercept terms

**Optimizer Differences**:
- OLS: Uses vanilla SGD with constant learning rate
- Logit: Uses Adam optimizer with StepLR scheduler (lr decay) and implements early stopping with patience mechanism

### Module Organization

```
statsmodels_sgd/
├── api.py                     # Public API - import OLS and Logit from here
├── base_model.py              # Base class (prints PyTorch debug info on import)
├── tools.py                   # Utilities: add_constant, calculate_standard_errors, calculate_t_p_values
├── regression/
│   ├── linear_model.py        # OLS implementation
│   └── discrete_model.py      # Logit implementation with early stopping
└── tests/
    ├── test_ols.py            # Tests OLS against statsmodels WLS
    └── test_logit.py          # Tests Logit against statsmodels GLM
```

## Important Implementation Details

### PyTorch Backend Debugging

The `base_model.py` module prints extensive debugging information on import:
- Python executable path and version
- PyTorch version and installation path
- Full pip package list if PyTorch import fails

This is intentional for troubleshooting installation issues but should be considered if import-time side effects are problematic.

### Model-Specific Behaviors

**OLS**:
- Default hyperparameters: lr=0.01, epochs=1000, batch_size=32, clip_value=1.0
- Automatically adds constant term unless `add_constant=False` is specified in `fit()`
- No early stopping - always runs for full epochs

**Logit**:
- Default hyperparameters: lr=0.1, epochs=2000, batch_size=32, clip_value=5.0
- Uses Adam optimizer (not SGD) with StepLR scheduler (step_size=100, gamma=0.9)
- Implements early stopping with patience=50
- Tracks best weights during training and restores them
- Returns results as pandas DataFrame via `summary()`

### Testing Strategy

Tests compare SGD-based models against statsmodels baselines:
- `test_ols_vs_statsmodels_with_weights()`: Compares against WLS with tolerance
- `test_logit_vs_statsmodels_with_weights()`: Compares against GLM with tolerance

Both tests use weighted samples to validate the weighting implementation. Tests use `numpy.testing.assert_allclose()` with `rtol` (relative tolerance) to handle SGD convergence variability.

### PyTorch Installation

CI/CD and local development use CPU-only PyTorch from the official index:
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This avoids CUDA dependencies and reduces installation size.

## Package Structure

- **Entry point**: Users import via `statsmodels_sgd.api` (e.g., `import statsmodels_sgd.api as sm_sgd`)
- **Dependencies**: numpy, pandas, statsmodels (for utilities), torch (for SGD)
- **Python version**: Requires Python 3.8+
- **License**: MIT (Copyright 2024 PolicyEngine)

## Documentation

Documentation is built with Jupyter Book and located in `statsmodels_sgd/docs/`:
- `installation.md`: Installation guide
- `ols-example.md`: OLS usage example
- `logit-example.md`: Logit usage example
- `cps-asec-example.md`: Real-world example with CPS-ASEC data
