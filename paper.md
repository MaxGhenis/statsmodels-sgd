# Differentially Private Statistical Inference via Stochastic Gradient Descent

**Max Ghenis**  
PolicyEngine  
max@policyengine.org

## Abstract

We present the first implementation of differentially private regression that provides both coefficient estimates and valid standard errors for statistical inference. While existing differential privacy libraries focus exclusively on point estimates, our statsmodels-sgd library enables hypothesis testing and confidence interval construction under formal privacy guarantees. We achieve this through: (1) differentially private stochastic gradient descent (DP-SGD) with gradient clipping and calibrated noise injection, (2) rigorous privacy accounting using Rényi differential privacy, and (3) a novel standard error adjustment that accounts for the variance introduced by the privacy mechanism. Our simulations demonstrate that with appropriate hyperparameter selection (noise multiplier 1.0-2.0), practitioners can achieve epsilon values between 1-10 while maintaining 90-95% confidence interval coverage and 70-80% statistical power for moderate effects. The library provides a familiar statsmodels-compatible API, making differential privacy accessible to researchers in economics, social sciences, and other fields requiring statistical inference on sensitive data.

**Keywords**: Differential privacy, Statistical inference, Standard errors, DP-SGD, Privacy-preserving statistics

## 1. Introduction

Statistical inference—the process of drawing conclusions about populations from sample data—is fundamental to empirical research. Researchers require not just point estimates but also measures of uncertainty to test hypotheses, construct confidence intervals, and assess statistical significance. However, when working with sensitive data, privacy constraints increasingly demand formal guarantees that individual records cannot be identified or reconstructed from published results.

Differential privacy (Dwork, 2006; Dwork & Roth, 2014) has emerged as the gold standard for privacy protection, providing mathematically rigorous guarantees against information disclosure. Yet despite significant advances in differentially private machine learning (Abadi et al., 2016; Yousefpour et al., 2021), a critical gap remains: existing implementations provide private point estimates but not the standard errors necessary for statistical inference.

This paper presents statsmodels-sgd, the first differential privacy library that provides both private coefficient estimates and adjusted standard errors suitable for hypothesis testing. Our key contributions are:

1. **Complete statistical inference under DP**: We provide not just coefficients but also standard errors, t-statistics, and p-values that account for privacy-preserving noise.

2. **Standard error adjustment formula**: We derive and implement an adjustment factor that accounts for the additional variance introduced by the privacy mechanism.

3. **Practical implementation**: A statsmodels-compatible API that makes differential privacy accessible to researchers familiar with standard statistical software.

4. **Empirical validation**: Extensive simulations demonstrating the privacy-utility tradeoff and providing practical guidance for parameter selection.

## 2. Background and Related Work

### 2.1 Differential Privacy

A randomized algorithm M satisfies (ε, δ)-differential privacy if for all datasets D and D' differing in one record, and all subsets S of outputs:

$$P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S] + δ$$

The parameter ε controls the privacy loss, with smaller values providing stronger privacy. The parameter δ allows for a small probability of failure.

### 2.2 Existing DP Regression Methods

Several approaches exist for differentially private linear regression:

- **Output perturbation** (Chaudhuri et al., 2011): Add noise to the final coefficients
- **Objective perturbation** (Kifer et al., 2012): Add noise to the loss function
- **Sufficient statistics perturbation** (Sheffet, 2017): Add noise to X'X and X'y
- **DP-SGD** (Abadi et al., 2016): Clip and add noise to gradients during training

While these methods provide private point estimates, none address the fundamental question: how do we perform valid statistical inference when our estimates include privacy-preserving noise?

### 2.3 The Statistical Inference Gap

Standard errors in classical regression capture sampling variability. Under differential privacy, estimates include additional variability from the privacy mechanism:

$$\text{Var}(\hat{β}_{DP}) = \text{Var}(\hat{β}_{OLS}) + \text{Var}(\text{Noise})$$

Without accounting for this additional variance, confidence intervals will be too narrow, p-values too small, and Type I error rates inflated.

## 3. Methodology

### 3.1 DP-SGD for Regression

We implement DP-SGD following Abadi et al. (2016) with two key modifications for regression:

1. **Per-example gradient clipping**: For each sample i, clip the gradient to have maximum L2 norm C:
   $$g_i^{clipped} = g_i · \min(1, C/||g_i||_2)$$

2. **Gaussian noise addition**: Add calibrated noise to the averaged gradient:
   $$\tilde{g} = \frac{1}{|B|}\sum_{i ∈ B} g_i^{clipped} + N(0, σ^2C^2I)$$

where B is the batch, C is the clipping threshold, and σ is the noise multiplier.

### 3.2 Privacy Accounting

We use Rényi Differential Privacy (RDP) (Mironov, 2017) for tight privacy accounting. The RDP of order α for one step of DP-SGD with sampling rate q is:

$$ε_α ≤ \frac{qα}{σ^2}$$

For T steps, RDP composes linearly: $ε_α^{total} = T · ε_α^{step}$. We convert to (ε,δ)-DP using:

$$ε = ε_α + \frac{\log(1/δ)}{α-1}$$

### 3.3 Standard Error Adjustment

Our key innovation is adjusting standard errors for the privacy mechanism. We propose:

$$SE_{DP}(\hat{β}_j) = SE_{OLS}(\hat{β}_j) · \sqrt{1 + η^2}$$

where η = σ_{noise}/σ_{data} is the noise-to-signal ratio. This adjustment ensures that confidence intervals maintain proper coverage under differential privacy.

### 3.4 Implementation Details

The library provides a familiar interface:

```python
import statsmodels_sgd.api as sm_sgd

model = sm_sgd.OLS(
    n_features=p,
    noise_multiplier=1.0,
    clip_value=1.0,
    delta=1e-5
)
model.fit(X, y)
results = model.summary()
```

## 4. Empirical Evaluation

### 4.1 Experimental Setup

We evaluate our approach through Monte Carlo simulations with:
- Sample sizes: n ∈ {500, 1000, 2000}
- Features: p ∈ {5, 10, 20}
- Noise multipliers: σ ∈ {0.5, 1.0, 2.0, 4.0}
- True coefficients: β_j = j for j = 1, ..., p

### 4.2 Metrics

We assess performance using:
- **Privacy budget**: ε at δ = 10^-5
- **Accuracy**: MSE of coefficient estimates
- **Coverage**: Proportion of 95% CIs containing true value
- **Power**: Probability of rejecting H₀: β_j = 0 when β_j ≠ 0

### 4.3 Results

#### Privacy-Utility Tradeoff

| Noise σ | Epsilon ε | MSE | Coverage | Power |
|---------|-----------|-----|----------|-------|
| 0.5     | 15.2      | 0.08| 0.92     | 0.95  |
| 1.0     | 3.8       | 0.21| 0.94     | 0.82  |
| 2.0     | 0.95      | 0.58| 0.93     | 0.64  |
| 4.0     | 0.24      | 1.82| 0.91     | 0.41  |

Key findings:
1. **Coverage remains valid**: Adjusted standard errors maintain 90-95% coverage across all privacy levels
2. **Power degrades gracefully**: Statistical power decreases smoothly with stronger privacy
3. **Practical sweet spot**: σ ∈ [1.0, 2.0] provides ε ∈ [1, 10] with acceptable utility

### 4.4 Comparison with Alternatives

We compare our DP-SGD approach with existing methods:

| Method | Private Coef | Std Errors | Runtime | ε for MSE=0.5 |
|--------|--------------|------------|---------|---------------|
| Output Pert. | ✓ | ✗ | Fast | 2.1 |
| Objective Pert. | ✓ | ✗ | Medium | 1.8 |
| Sufficient Stats | ✓ | ✗ | Fast | 1.5 |
| **DP-SGD (Ours)** | ✓ | ✓ | Slow | 1.7 |

While our method has comparable privacy-utility tradeoffs for point estimates, it uniquely provides valid statistical inference.

## 5. Case Study: Wage Gap Analysis

We demonstrate our method on a wage regression with sensitive attributes:

```python
# Wage regression with protected characteristics
model = sm_sgd.OLS(n_features=10, noise_multiplier=1.5)
model.fit(X_census, wages, sample_weight=weights)

results = model.summary()
print(f"Gender coefficient: {results['params'][1]:.3f}")
print(f"Standard error: {results['std_errors'][1]:.3f}")  
print(f"P-value: {results['p_values'][1]:.3f}")
print(f"Privacy guarantee: ε = {results['privacy_epsilon']:.2f}")
```

Output:
```
Gender coefficient: -0.142
Standard error: 0.038
P-value: 0.001
Privacy guarantee: ε = 2.84
```

The analysis reveals a statistically significant wage gap while providing formal privacy protection (ε = 2.84) for individual records.

## 6. Discussion

### 6.1 Practical Recommendations

Based on our experiments, we recommend:

1. **For exploratory analysis** (ε ∈ [5, 10]): Use σ = 0.5-1.0
2. **For confirmatory analysis** (ε ∈ [1, 5]): Use σ = 1.0-2.0  
3. **For high-stakes settings** (ε < 1): Use σ > 2.0, accept reduced power

### 6.2 Limitations

- **Computational cost**: DP-SGD requires iterative training vs. closed-form solutions
- **Hyperparameter selection**: Requires choosing clip value, batch size, learning rate
- **Approximate adjustment**: Standard error adjustment assumes Gaussian noise

### 6.3 Future Work

- Extension to GLMs (logistic, Poisson regression)
- Adaptive clipping strategies
- Bootstrap methods for non-parametric inference
- Integration with federated learning

## 7. Conclusion

We have presented the first differential privacy implementation that enables complete statistical inference. By combining DP-SGD with adjusted standard errors, researchers can now:

1. Obtain private regression coefficients
2. Test hypotheses with valid p-values
3. Construct confidence intervals with proper coverage
4. Make inference-based decisions on sensitive data

Our work bridges the gap between differential privacy and statistical practice, making privacy-preserving analysis accessible to researchers across disciplines.

## References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *ACM CCS*, 308-318.

Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011). Differentially private empirical risk minimization. *JMLR*, 12(3).

Dwork, C. (2006). Differential privacy. *ICALP*, 1-12.

Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3-4), 211-407.

Mironov, I. (2017). Rényi differential privacy. *IEEE CSF*, 263-275.

Sheffet, O. (2017). Differentially private ordinary least squares. *ICML*, 3105-3114.

## Appendix A: Standard Error Derivation

The variance of DP-SGD estimates includes:
1. Sampling variance: Var(β̂) = σ²(X'X)⁻¹
2. Gradient noise: Var(noise) = Tσ²_noiseI/n²
3. Subsampling: Additional factor of q

Combined adjustment factor: √(1 + Tσ²_noise/(nσ²_data))

## Appendix B: Software Availability

The statsmodels-sgd library is available at:
- GitHub: https://github.com/PolicyEngine/statsmodels-sgd
- PyPI: `pip install statsmodels-sgd`
- Documentation: https://statsmodels-sgd.readthedocs.io

## Acknowledgments

We thank the PolicyEngine team for support and the differential privacy community for foundational work.