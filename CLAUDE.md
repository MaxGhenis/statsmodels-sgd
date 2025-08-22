# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **statsmodels-sgd**, a reimplementation of statsmodels using stochastic gradient descent (SGD) with gradient clipping for differential privacy. It provides PyTorch-based implementations of common statistical models that mimic the statsmodels API but use SGD optimization instead of analytical solutions.

## Architecture and Key Components

### Core Structure
- **Base Model** (`statsmodels_sgd/base_model.py`): Abstract base class using PyTorch's `nn.Module` that provides the SGD training framework with gradient clipping
- **Regression Models** (`statsmodels_sgd/regression/`):
  - `linear_model.py`: OLS (Ordinary Least Squares) implementation
  - `discrete_model.py`: Logit model implementation for binary classification
- **API** (`statsmodels_sgd/api.py`): Main entry point exposing `OLS` and `Logit` classes
- **Tools** (`statsmodels_sgd/tools.py`): Utility functions for adding constants, calculating standard errors, t-values, and p-values

### Key Design Patterns
- Models inherit from `BaseModel` which extends `torch.nn.Module`
- All models support weighted samples via `sample_weight` parameter
- Models use SGD with gradient clipping (`clip_value` parameter) for differential privacy
- API mimics statsmodels interface but with `n_features` required at initialization
- Models require explicit constant term addition (using `statsmodels.api.add_constant()`)

## Development Commands

### Installation
```bash
# Install package in development mode with all dependencies
pip install -e ".[dev]"

# Or install specific to use
pip install git+https://github.com/PolicyEngine/statsmodels-sgd.git
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest statsmodels_sgd/tests/test_ols.py

# Run specific test function
pytest statsmodels_sgd/tests/test_ols.py::test_ols_fit_predict_with_weights

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=statsmodels_sgd
```

### Code Quality
```bash
# Format code with black
black statsmodels_sgd/

# Sort imports
isort statsmodels_sgd/

# Lint with flake8
flake8 statsmodels_sgd/ --max-line-length=127

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

### Testing Approach
Tests compare outputs with standard statsmodels implementations:
- OLS compared with `statsmodels.api.WLS` (when using weights)
- Logit compared with `statsmodels.api.Logit`
- Tests use relative tolerance (rtol=0.1) due to SGD's stochastic nature

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

## Project Configuration

The project uses both `pyproject.toml` configurations:
- Standard Python project metadata under `[project]`
- Poetry configuration under `[tool.poetry]` (appears to be legacy/duplicate)
- Build system uses setuptools with `setuptools.build_meta` backend

## CRITICAL: Accuracy and Verification Requirements

### NEVER Present Hypothetical Results as Real
When working with simulations, experiments, or data analysis:

1. **Code Execution Status - ALWAYS be explicit:**
   - ✅ "This code runs and produces these actual results: [paste real output]"
   - ⚠️ "This is a template/framework that needs implementation"
   - ❌ NEVER say "The results show X" unless code actually ran and produced X

2. **Simulation Results - STRICT rules:**
   - If notebook/code doesn't run: "This notebook provides the framework but cannot run because [specific missing pieces]"
   - If presenting example output: "This is a MOCKUP of expected output format, not actual results"
   - NEVER fabricate specific numbers (e.g., "ε = 3.8, coverage = 94%") without actual execution

3. **Feature Claims - Verification required:**
   - Before claiming "this implements X": Verify the feature is fully implemented and working
   - Before claiming "first/only implementation": Verify the unique feature actually works
   - If uncertain: "This aims to implement X but needs verification"

4. **Results and Data - Red flags to catch:**
   - Specific metrics without execution: "MSE = 0.21" ❌ vs "MSE would be calculated" ✅
   - Performance claims without benchmarks: "Achieves 95% coverage" ❌ vs "Targets 95% coverage" ✅
   - Comparison results without running comparisons: "Outperforms baseline" ❌ vs "Expected to compare favorably" ✅

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