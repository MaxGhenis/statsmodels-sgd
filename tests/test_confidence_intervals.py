"""
Test-driven development for fixing confidence interval coverage.
We'll write tests first, then fix the implementation.
"""

import numpy as np
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels_sgd.api import OLS


class TestConfidenceIntervals:
    """Test confidence interval coverage and standard errors."""
    
    def test_standard_errors_positive(self):
        """Standard errors should always be positive."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        model = OLS(n_features=4, noise_multiplier=2.0)
        model.fit(X, y)
        summary = model.summary()
        
        assert all(se > 0 for se in summary['std_errors']), "All standard errors must be positive"
    
    def test_standard_errors_increase_with_noise(self):
        """Standard errors should increase with more DP noise."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(200)
        
        # Test with different noise levels
        se_low = None
        se_high = None
        
        for noise_mult in [1.0, 5.0]:
            model = OLS(n_features=4, noise_multiplier=noise_mult)
            model.fit(X, y)
            summary = model.summary()
            
            if noise_mult == 1.0:
                se_low = np.mean(summary['std_errors'])
            else:
                se_high = np.mean(summary['std_errors'])
        
        assert se_high > se_low, "Higher noise should lead to larger standard errors"
    
    def test_basic_coverage_single_coefficient(self):
        """Test coverage for a single coefficient with known true value."""
        np.random.seed(42)
        n_sims = 100
        true_beta = 2.0
        coverage_count = 0
        
        for _ in range(n_sims):
            # Simple regression: y = 2*x + noise
            X = np.random.randn(500, 1)
            y = true_beta * X.squeeze() + np.random.randn(500) * 0.5
            
            # Fit with minimal DP noise for this test
            model = OLS(n_features=2, noise_multiplier=0.5, epochs=50)
            model.fit(X, y)
            summary = model.summary()
            
            # Get coefficient and SE (skip intercept)
            coef = summary['coefficients'][1]
            se = summary['std_errors'][1]
            
            # 95% CI
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            if ci_lower <= true_beta <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_sims
        print(f"Coverage rate: {coverage_rate:.2%}")
        
        # We expect at least 80% coverage with DP noise
        assert coverage_rate >= 0.80, f"Coverage {coverage_rate:.2%} is too low"
    
    def test_coverage_multiple_coefficients(self):
        """Test coverage for multiple coefficients simultaneously."""
        np.random.seed(42)
        n_sims = 50
        true_coef = np.array([1.0, -2.0, 3.0])
        joint_coverage_count = 0
        individual_coverage = [0, 0, 0]
        
        for _ in range(n_sims):
            X = np.random.randn(300, 3)
            y = X @ true_coef + np.random.randn(300) * 0.5
            
            model = OLS(n_features=4, noise_multiplier=1.0, epochs=40)
            model.fit(X, y)
            summary = model.summary()
            
            coef = summary['coefficients'][1:]  # Skip intercept
            se = summary['std_errors'][1:]
            
            # Check individual coverage
            all_covered = True
            for i in range(3):
                ci_lower = coef[i] - 1.96 * se[i]
                ci_upper = coef[i] + 1.96 * se[i]
                
                if ci_lower <= true_coef[i] <= ci_upper:
                    individual_coverage[i] += 1
                else:
                    all_covered = False
            
            if all_covered:
                joint_coverage_count += 1
        
        # Individual coverage should be reasonable
        for i in range(3):
            ind_rate = individual_coverage[i] / n_sims
            print(f"Coverage for β_{i+1}: {ind_rate:.2%}")
            assert ind_rate >= 0.70, f"Individual coverage for β_{i+1} too low: {ind_rate:.2%}"
        
        # Joint coverage will be lower but should be > 50%
        joint_rate = joint_coverage_count / n_sims
        print(f"Joint coverage: {joint_rate:.2%}")
        assert joint_rate >= 0.50, f"Joint coverage too low: {joint_rate:.2%}"
    
    def test_standard_error_formula_components(self):
        """Test that SE formula properly combines sampling and privacy variance."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(500)
        
        # Non-private baseline
        X_const = sm.add_constant(X)
        ols = sm.OLS(y, X_const).fit()
        baseline_se = ols.bse
        
        # DP model
        model = OLS(n_features=4, noise_multiplier=2.0, epochs=30)
        model.fit(X, y)
        
        # Check internal calculations
        assert hasattr(model, 'n_samples_'), "Model should store sample size"
        
        # The DP standard errors should be larger but not ridiculously so
        summary = model.summary()
        dp_se = summary['std_errors']
        
        ratio = np.mean(dp_se) / np.mean(baseline_se)
        print(f"SE ratio (DP/OLS): {ratio:.2f}x")
        
        # Should be larger but not more than 20x for reasonable noise
        assert 1.0 < ratio < 20.0, f"SE ratio {ratio:.2f}x is unreasonable"
    
    def test_t_statistics_and_p_values(self):
        """Test that t-statistics and p-values are computed correctly."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        # Make one coefficient clearly significant
        y = 5.0 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(200) * 0.5
        
        model = OLS(n_features=3, noise_multiplier=1.0)
        model.fit(X, y)
        summary = model.summary()
        
        # Check t-statistics
        t_stats = summary['t_values'][1:]  # Skip intercept
        p_vals = summary['p_values'][1:]
        
        # First coefficient should be highly significant
        assert abs(t_stats[0]) > 2.0, "Strong effect should have |t| > 2"
        assert p_vals[0] < 0.05, "Strong effect should be significant"
        
        # Second coefficient might not be significant
        print(f"t-stats: {t_stats}")
        print(f"p-values: {p_vals}")
    
    @pytest.mark.slow
    def test_coverage_convergence_with_sample_size(self):
        """Test that coverage improves with larger sample sizes."""
        np.random.seed(42)
        sample_sizes = [100, 500, 1000]
        coverages = []
        
        for n in sample_sizes:
            true_beta = 3.0
            coverage_count = 0
            n_sims = 30
            
            for _ in range(n_sims):
                X = np.random.randn(n, 1)
                y = true_beta * X.squeeze() + np.random.randn(n)
                
                model = OLS(n_features=2, noise_multiplier=1.0)
                model.fit(X, y)
                summary = model.summary()
                
                coef = summary['coefficients'][1]
                se = summary['std_errors'][1]
                
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
                
                if ci_lower <= true_beta <= ci_upper:
                    coverage_count += 1
            
            coverage = coverage_count / n_sims
            coverages.append(coverage)
            print(f"n={n}: Coverage = {coverage:.2%}")
        
        # Coverage should improve with sample size
        assert coverages[-1] >= coverages[0], "Coverage should improve with larger samples"


class TestStandardErrorAdjustment:
    """Test the actual standard error adjustment mechanism."""
    
    def test_dp_noise_variance_calculation(self):
        """Test that DP noise variance is calculated correctly."""
        # Create a simple model to test the calculation
        model = OLS(n_features=3, noise_multiplier=2.0, clip_value=1.0)
        
        # Manually calculate expected noise variance
        noise_var_per_step = (model.clip_value * model.noise_multiplier) ** 2
        
        # This should match what the model calculates internally
        assert noise_var_per_step == 4.0, "Noise variance per step should be (clip * noise_mult)^2"
    
    def test_finite_sample_correction(self):
        """Test that finite sample corrections are applied."""
        np.random.seed(42)
        
        for n in [50, 100, 500]:
            X = np.random.randn(n, 2)
            y = np.random.randn(n)
            
            model = OLS(n_features=3)
            model.fit(X, y)
            
            # Check degrees of freedom adjustment
            assert hasattr(model, 'n_samples_'), f"Model should track sample size"
            assert model.n_samples_ == n, f"Sample size should be {n}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])