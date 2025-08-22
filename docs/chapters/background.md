# Background and Motivation

## The Privacy-Inference Gap

Statistical inference is fundamental to empirical research. Researchers need not just point estimates but also measures of uncertainty—standard errors, confidence intervals, and p-values—to draw valid conclusions. However, when working with sensitive data, privacy constraints often force a difficult choice: either sacrifice privacy or sacrifice inference.

## Current State of Differential Privacy Tools

The landscape of differential privacy tools reveals a critical gap:

### Machine Learning Focus
Most existing DP libraries {cite}`opacus2021,diffprivlib2019` are designed for machine learning applications where:
- The goal is prediction accuracy, not coefficient interpretation
- Standard errors are not needed for deployment
- Privacy guarantees focus on the trained model, not statistical inference

### Limited Statistical Functionality
Current implementations provide:
- ✅ Private point estimates (coefficients)
- ❌ Standard errors
- ❌ Hypothesis tests
- ❌ Confidence intervals

This limitation severely restricts the use of differential privacy in:
- Economics research
- Social science studies
- Policy analysis
- Medical research
- Any domain requiring statistical inference

## Why Standard Errors Matter Under DP

When we add noise for differential privacy, we fundamentally change the distribution of our estimates. Consider the OLS estimator:

$$\hat{\beta}_{DP} = \hat{\beta}_{OLS} + \text{Noise}$$

The variance of this estimator is:

$$\text{Var}(\hat{\beta}_{DP}) = \text{Var}(\hat{\beta}_{OLS}) + \text{Var}(\text{Noise})$$

Without accounting for the added noise variance, standard errors will be incorrect, leading to:
- Invalid hypothesis tests
- Incorrect confidence intervals
- Misleading p-values
- False conclusions about statistical significance

## The Innovation: DP-SGD with Statistical Inference

Our approach addresses this gap by:

1. **Using DP-SGD for training**: Gradient clipping and noise addition provide formal privacy guarantees
2. **Tracking privacy budget**: Rigorous accounting using Rényi DP {cite}`mironov2017renyi`
3. **Adjusting standard errors**: Accounting for the variance introduced by privacy mechanisms
4. **Maintaining familiar API**: Statsmodels-like interface for ease of adoption

## Theoretical Foundation

### Differential Privacy Definition
A randomized algorithm $\mathcal{M}$ satisfies $(ε, δ)$-differential privacy if for all datasets $D$ and $D'$ differing in one record, and all subsets $S$ of outputs:

$$P[\mathcal{M}(D) \in S] \leq e^ε \cdot P[\mathcal{M}(D') \in S] + δ$$

### DP-SGD Algorithm
The DP-SGD algorithm {cite}`abadi2016deep` achieves differential privacy through:

1. **Gradient Clipping**: Bound the influence of any single sample
   $$g_i^{clipped} = g_i \cdot \min\left(1, \frac{C}{||g_i||_2}\right)$$

2. **Noise Addition**: Add calibrated Gaussian noise
   $$\tilde{g} = \frac{1}{|B|}\sum_{i \in B} g_i^{clipped} + \mathcal{N}(0, \sigma^2 C^2 I)$$

3. **Privacy Amplification**: Subsampling provides additional privacy
   $$ε_{total} = f(ε_{step}, \text{sampling rate}, \text{iterations})$$

### Standard Error Adjustment

For OLS under DP-SGD, we propose the adjusted standard error formula:

$$SE_{DP}(\hat{\beta}_j) = SE_{OLS}(\hat{\beta}_j) \cdot \sqrt{1 + \frac{\sigma^2_{noise}}{\sigma^2_{data}}}$$

Where:
- $SE_{OLS}$ is the traditional standard error
- $\sigma^2_{noise}$ is the variance from DP noise
- $\sigma^2_{data}$ is the data variance

This adjustment ensures valid inference under differential privacy.

## Use Cases and Applications

### Economics and Social Science
- Estimating treatment effects on sensitive outcomes
- Labor market analysis with protected characteristics
- Income and wealth regression with privacy guarantees

### Healthcare and Medicine
- Clinical trial analysis with patient privacy
- Epidemiological studies on sensitive conditions
- Healthcare utilization models

### Policy Analysis
- Program evaluation with administrative data
- Census-based research with formal privacy
- Education outcomes with student privacy

## Comparison with Alternatives

| Method | Privacy | Coefficients | Std Errors | Efficiency |
|--------|---------|--------------|------------|------------|
| Output Perturbation | ✅ | ✅ | ❌ | High |
| Objective Perturbation | ✅ | ✅ | ❌ | Medium |
| Sufficient Statistics | ✅ | ✅ | ❌ | High |
| **DP-SGD (Ours)** | ✅ | ✅ | ✅ | Medium |

While other methods may be more efficient for point estimates alone, only our approach provides the complete statistical inference framework needed for research.

## Summary

The gap between differential privacy and statistical inference has limited the adoption of privacy-preserving methods in empirical research. By providing both private estimates and valid standard errors, statsmodels-sgd enables researchers to:

- Maintain rigorous privacy guarantees
- Perform valid statistical inference
- Use familiar statsmodels-like syntax
- Apply DP to real research problems

The next chapter details our methodology for achieving these goals.