# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **statsmodels-sgd**, a reimplementation of statsmodels using stochastic gradient descent (SGD) with gradient clipping for differential privacy. It provides PyTorch-based implementations of common statistical models that mimic the statsmodels API but use SGD optimization instead of analytical solutions.

## Architecture

### Core Design Pattern

All models follow a consistent three-layer architecture:

1. **API Layer** (`api.py`): Exports public-facing classes (`OLS`, `Logit`)
2. **Model Layer** (`regression/`): Implements specific regression models
3. **Base Layer** (`base_model.py`): Provides shared functionality (currently minimal, models inherit from `nn.Module` directly)

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

## Development Commands

### Installation

```bash
# Install development dependencies
python -m pip install --upgrade pip
python -m pip install pytest black isort flake8

# Install PyTorch (CPU-only version used in CI)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install package in development mode
python -m pip install -e ".[dev]"

# Or install from GitHub
pip install git+https://github.com/PolicyEngine/statsmodels-sgd.git
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest statsmodels_sgd/tests/test_ols.py
pytest statsmodels_sgd/tests/test_logit.py

# Run specific test function
pytest statsmodels_sgd/tests/test_ols.py::test_ols_fit_predict_with_weights

# Run with verbose output
pytest -v

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

# Run all formatters (typical before commit)
black statsmodels_sgd/ && isort statsmodels_sgd/
```

## Important Implementation Notes

### Model Usage Pattern

```python
import statsmodels_sgd.api as sm_sgd
import statsmodels.api as sm

# Important: Models need n_features specified at init time
# Add 1 for constant term if using add_constant
model = sm_sgd.OLS(n_features=X.shape[1] + 1)

# Add constant to X before fitting (mimics statsmodels)
X_with_const = sm.add_constant(X)
model.fit(X_with_const, y, sample_weight=sample_weight)

# Predict also needs constant added
y_pred = model.predict(X_with_const)
```

### SGD Parameters

All models accept these SGD-specific parameters:
- `learning_rate`: Default 0.01
- `epochs`: Default 1000
- `batch_size`: Default 32
- `clip_value`: Gradient clipping value for differential privacy (default 1.0)

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

## CI/CD Configuration

GitHub Actions workflow (`.github/workflows/ci-cd.yaml`) runs on push and PR:
1. Python 3.12 environment
2. Installs PyTorch CPU version for CI efficiency
3. Runs flake8 linting (max line length 127)
4. Executes pytest suite

## Dependencies

### Core Dependencies
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `statsmodels`: API compatibility and comparison
- `torch`: SGD optimization engine

### Development Dependencies
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting

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

## CRITICAL: Accuracy and Verification Requirements

### NEVER Present Hypothetical Results as Real
When working with simulations, experiments, or data analysis:

1. **Code Execution Status - ALWAYS be explicit:**
   - "This code runs and produces these actual results: [paste real output]"
   - "This is a template/framework that needs implementation"
   - NEVER say "The results show X" unless code actually ran and produced X

2. **Simulation Results - STRICT rules:**
   - If notebook/code doesn't run: "This notebook provides the framework but cannot run because [specific missing pieces]"
   - If presenting example output: "This is a MOCKUP of expected output format, not actual results"
   - NEVER fabricate specific numbers (e.g., "epsilon = 3.8, coverage = 94%") without actual execution

3. **Feature Claims - Verification required:**
   - Before claiming "this implements X": Verify the feature is fully implemented and working
   - Before claiming "first/only implementation": Verify the unique feature actually works
   - If uncertain: "This aims to implement X but needs verification"

4. **Results and Data - Red flags to catch:**
   - Specific metrics without execution: "MSE = 0.21" vs "MSE would be calculated"
   - Performance claims without benchmarks: "Achieves 95% coverage" vs "Targets 95% coverage"
   - Comparison results without running comparisons: "Outperforms baseline" vs "Expected to compare favorably"

### Implementation Status Tracking
When describing this repository's features:
- **Partially Implemented**: Privacy accounting exists but needs integration with models
- **Not Implemented**: Standard error adjustment for DP noise
- **Framework Only**: Simulation notebooks are templates, not running code

### Scientific Integrity
**Fabricating results, even unintentionally, can destroy careers and credibility.**
- Always distinguish between theoretical expectations and actual results
- Never present hypothetical numbers as real findings
- When unsure, explicitly state: "Cannot verify without running code"
