"""Test standard error calibration for proper coverage."""

import numpy as np
import statsmodels_sgd.api as sm_sgd
import statsmodels.api as sm
from scipy import stats

def test_coverage_calibration():
    """Test that coverage is approximately 95%, not 100%."""
    np.random.seed(42)
    
    # Run simulations to check coverage
    n_sims = 200
    true_beta = 2.0
    coverage_rates = []
    
    for noise_mult in [0.5, 1.0, 2.0]:
        in_ci_count = 0
        
        for _ in range(n_sims):
            # Generate data
            X = np.random.randn(200, 1)
            y = true_beta * X.squeeze() + np.random.randn(200) * 0.5
            
            # Fit model
            model = sm_sgd.OLS(
                n_features=2, 
                noise_multiplier=noise_mult,
                epochs=30,
                track_privacy=True
            )
            model.fit(X, y)
            summary = model.summary()
            
            # Check if true parameter is in CI
            coef = summary['coefficients'][1]  # Skip intercept
            se = summary['std_errors'][1]
            
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            if ci_lower <= true_beta <= ci_upper:
                in_ci_count += 1
        
        coverage = in_ci_count / n_sims
        coverage_rates.append(coverage)
        print(f"Noise {noise_mult}: Coverage = {coverage:.1%}")
    
    avg_coverage = np.mean(coverage_rates)
    print(f"Average coverage: {avg_coverage:.1%}")
    
    # Should be close to 95%, not 100%
    assert 0.90 <= avg_coverage <= 0.985, f"Coverage {avg_coverage:.1%} outside expected range"
    
    return avg_coverage


def test_se_inflation_reasonable():
    """Test that SE inflation is reasonable, not excessive."""
    np.random.seed(42)
    
    X = np.random.randn(300, 2)
    y = np.random.randn(300)
    
    # Get baseline SEs from standard OLS
    X_const = sm.add_constant(X)
    ols = sm.OLS(y, X_const).fit()
    baseline_se = np.mean(ols.bse)
    
    # Test different noise levels
    noise_levels = [0.5, 1.0, 2.0, 3.0]
    inflations = []
    
    for noise_mult in noise_levels:
        model = sm_sgd.OLS(
            n_features=3,
            noise_multiplier=noise_mult,
            epochs=30,
            track_privacy=True
        )
        model.fit(X, y)
        summary = model.summary()
        
        mean_se = np.mean(summary['std_errors'])
        inflation = mean_se / baseline_se
        inflations.append(inflation)
        print(f"Noise {noise_mult}: SE inflation = {inflation:.1f}x")
    
    # Check that inflation is reasonable
    # Should be significant but not excessive
    assert inflations[0] < 5, f"Low noise inflation {inflations[0]:.1f}x too high"
    assert inflations[-1] < 20, f"High noise inflation {inflations[-1]:.1f}x too high"
    
    # Should increase with noise
    assert all(inflations[i] <= inflations[i+1] for i in range(len(inflations)-1)), \
        "SE inflation should increase with noise"
    
    return inflations


if __name__ == "__main__":
    print("Testing SE calibration...")
    coverage = test_coverage_calibration()
    print(f"\nFinal coverage: {coverage:.1%}")
    
    print("\nTesting SE inflation...")
    inflations = test_se_inflation_reasonable()
    print(f"Inflation range: {min(inflations):.1f}x to {max(inflations):.1f}x")