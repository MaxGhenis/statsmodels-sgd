import numpy as np
from scipy import stats


def add_constant(X):
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def calculate_standard_errors(
    X, y, weights, bias, is_logit=False, sample_weight=None
):
    X_with_intercept = add_constant(X)

    if is_logit:
        y_pred = 1 / (1 + np.exp(-np.dot(X, weights) - bias))
        V = np.diag(y_pred * (1 - y_pred))
        if sample_weight is not None:
            V = np.diag(sample_weight.squeeze()) @ V
        var_covar_matrix = np.linalg.inv(
            X_with_intercept.T @ V @ X_with_intercept
        )
    else:
        y_pred = np.dot(X, weights) + bias
        residuals = y - y_pred
        if sample_weight is not None:
            mse = np.average(residuals**2, weights=sample_weight.squeeze())
            X_weighted = X_with_intercept * np.sqrt(sample_weight).reshape(-1, 1)
        else:
            mse = np.mean(residuals**2)
            X_weighted = X_with_intercept
        var_covar_matrix = mse * np.linalg.inv(
            np.dot(X_weighted.T, X_weighted)
        )

    return np.sqrt(np.diag(var_covar_matrix))


def calculate_t_p_values(coefficients, std_errors):
    t_values = coefficients / std_errors
    p_values = 2 * (
        1 - stats.t.cdf(np.abs(t_values), df=len(coefficients) - 1)
    )
    return t_values, p_values
