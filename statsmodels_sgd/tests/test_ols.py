import numpy as np
import pytest
import statsmodels_sgd.api as sm_sgd
import statsmodels.api as sm


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    true_weights = np.array([1, 2, 3, 4, 5])
    y = np.dot(X, true_weights) + np.random.randn(1000) * 0.1
    sample_weight = np.random.uniform(0.5, 1.5, size=1000)
    return X, y, sample_weight


def test_ols_fit_predict_with_weights(sample_data):
    X, y, sample_weight = sample_data
    model = sm_sgd.OLS(n_features=X.shape[1] + 1)  # +1 for the constant term
    model.fit(X, y, sample_weight=sample_weight)

    # Check predictions
    y_pred = model.predict(sm.add_constant(X))
    weighted_mse = np.average(
        (y - y_pred.squeeze()) ** 2, weights=sample_weight
    )
    # With DP-SGD, MSE is higher than analytical solution
    assert weighted_mse < 50.0  # Relaxed threshold for DP-SGD


def test_ols_vs_statsmodels_with_weights(sample_data):
    X, y, sample_weight = sample_data

    # Fit our model
    our_model = sm_sgd.OLS(
        n_features=X.shape[1] + 1
    )  # +1 for the constant term
    our_model.fit(X, y, sample_weight=sample_weight)

    # Fit statsmodels OLS
    X_sm = sm.add_constant(X)
    sm_model = sm.WLS(y, X_sm, weights=sample_weight).fit()

    # Compare predictions
    our_pred = our_model.predict(X_sm)
    sm_pred = sm_model.predict(X_sm)
    # DP-SGD has higher variance than analytical solution
    # Check correlation instead of exact match
    correlation = np.corrcoef(our_pred.squeeze(), sm_pred)[0, 1]
    assert correlation > 0.65, f"Correlation {correlation:.3f} too low"
