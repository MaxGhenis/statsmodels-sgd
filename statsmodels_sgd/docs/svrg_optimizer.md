# Stochastic Variance Reduced Gradient (SVRG) Optimizer

## Overview

The **Stochastic Variance Reduced Gradient (SVRG)** optimizer is an advanced optimization technique used for minimizing the variance of gradient estimates in stochastic gradient descent. By periodically computing full-batch gradients and using them to adjust mini-batch gradients, SVRG provides a more stable and efficient convergence, especially useful for large datasets and deep learning models.

## Key Features
- **Variance Reduction**: SVRG reduces gradient variance, leading to more stable parameter updates.
- **Improved Convergence**: It converges faster than standard SGD for many large-scale learning tasks.
- **Inner Loop Updates**: The algorithm performs a series of "inner loop" updates based on mini-batches, followed by an update using a "full gradient."

## Parameters

| Parameter     | Type    | Description                                                                                     |
|---------------|---------|-------------------------------------------------------------------------------------------------|
| `lr`          | `float` | Learning rate for gradient updates. Default is `0.01`.                                         |
| `n_inner`     | `int`   | Number of inner loop updates before recalculating the full gradient. Default is `10`.           |

## Usage Example

The SVRG optimizer is defined in `optimizers.py`. Below is an example of how to use it with a custom model.

```python
from statsmodels_sgd.optimizers import SVRG

# Initialize the SVRG optimizer
lr = 0.01
n_inner = 10
svrg_optimizer = SVRG(lr=lr, n_inner=n_inner)

# Example parameters and gradients
params = [np.array([1.0, 2.0])]
grads = [np.array([0.5, 0.5])]
full_grads = [np.array([0.3, 0.3])]

# Perform an update
svrg_optimizer.update(params, grads, full_grads)
```