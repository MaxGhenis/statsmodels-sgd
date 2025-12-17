# Related Work: DP Linear Regression with Inference

This chapter surveys existing approaches to differentially private linear regression, with a focus on methods that provide statistical inference (standard errors, confidence intervals). We compare these approaches and discuss their tradeoffs.

## The Inference Gap in DP Regression

Most differential privacy libraries focus on point estimates without uncertainty quantification {cite}`diffprivlib2019,opacus2021`. This is a critical gap because:

- Researchers need standard errors to test hypotheses
- Confidence intervals are essential for policy recommendations
- P-values determine statistical significance
- Without inference, results cannot be properly interpreted

Recent research has begun to address this gap through several approaches {cite}`king2024dpd,ferrando2024bootstrap`.

## Methods for DP Regression with Inference

### 1. Noisy Sufficient Statistics with Analytical Variance

**Algorithm:**
1. Compute sufficient statistics: $X'X$ and $X'y$
2. Add calibrated Gaussian or Laplace noise to each
3. Solve: $\hat{\beta} = (X'X + \text{noise}_1)^{-1}(X'y + \text{noise}_2)$
4. Derive variance analytically

**Variance Decomposition:**

The total variance decomposes into sampling variance plus privacy noise variance {cite}`evans2024linked`:

$$\text{Var}(\hat{\beta}_{DP}) = \Sigma^{OLS} + \omega^2(W'W)^{-1}(I + \Sigma_0 + \Sigma_1 + \Sigma_2)(W'W)^{-1}$$

Where:
- $\Sigma^{OLS}$ is the classical OLS variance
- $\omega^2$ captures the privacy noise scale
- Additional terms account for noise in $X'X$ and $X'y$

**Advantages:**
- Closed-form standard errors
- Computationally efficient
- Theoretically grounded

**Disadvantages:**
- Requires data bounds (or privacy leak)
- Can be numerically unstable with small eigenvalues
- Tight bounds require careful analysis

**Implementation:** IBM's diffprivlib {cite}`diffprivlib2019` implements the coefficient estimation but not the variance formulas.

### 2. Noise-Aware Bayesian Inference

{cite:t}`bernstein2019bayesian` proposed modeling the DP noise as part of the generative process.

**Algorithm:**
1. Release noisy sufficient statistics with known noise distribution
2. Specify prior: $\beta \sim \mathcal{N}(\mu_0, \Sigma_0)$
3. Include noise model: $\tilde{S} = S + \mathcal{N}(0, \sigma^2_{DP})$
4. Use MCMC (Gibbs sampling) to sample from posterior
5. Posterior provides both point estimates and credible intervals

**Key Innovation:**

The posterior properly accounts for both:
- Uncertainty from finite sample size
- Uncertainty from privacy mechanism noise

**Variance Formula:**

For conjugate priors, the posterior variance is:

$$\text{Var}(\beta | \tilde{S}) = \left(\Sigma_0^{-1} + \frac{1}{\sigma^2 + \sigma^2_{DP}} X'X\right)^{-1}$$

**Advantages:**
- Proper uncertainty quantification
- Full posterior distribution
- Can incorporate prior knowledge

**Disadvantages:**
- Computational cost (MCMC)
- Limited to exponential families
- Requires specifying priors

### 3. Measurement Error Correction

{cite:t}`king2024dpd` treat DP noise as classical measurement error and apply correction techniques.

**Key Insight:**

DP data release creates:
$$X_{observed} = X_{true} + \text{Noise}$$

This is exactly the measurement error model studied for decades in statistics.

**Correction Method:**

1. Estimate the noise variance from the DP mechanism
2. Apply bias correction to regression coefficients
3. Inflate standard errors to account for additional uncertainty

**Result:**

> "Statistically valid linear regression estimates...with appropriately larger standard errors"

**Implementations:**
- **PrivacyUnbiased**: R package
- **svinfer**: Python package from Meta

**Advantages:**
- Builds on well-established theory
- Familiar framework for statisticians
- Practical implementations available

**Disadvantages:**
- May require assumptions about noise structure
- Limited to certain DP mechanisms

### 4. Parametric Bootstrap for DP

{cite:t}`ferrando2024bootstrap` propose a parametric bootstrap that accounts for both sampling and privacy noise.

**Algorithm:**
1. Fit DP regression to get $\hat{\beta}_{DP}$
2. For $b = 1, \ldots, B$:
   - Generate bootstrap sample: $y^*_b = X\hat{\beta}_{DP} + \epsilon^*$
   - Add simulated DP noise: $\hat{\beta}^*_b = \text{DP-Fit}(X, y^*_b)$
3. Compute bootstrap confidence intervals

**Key Feature:**

The bootstrap naturally captures both sources of variability without requiring analytical variance formulas.

**Advantages:**
- Flexible - works with any DP mechanism
- No closed-form variance needed
- Can provide any quantile of the distribution

**Disadvantages:**
- Computationally expensive (many model fits)
- May consume additional privacy budget if not careful
- Requires correct specification of noise distribution

### 5. Asymptotic Theory for DP Estimators

Recent work derives the asymptotic distribution of DP estimators, enabling classical inference.

**Result** {cite}`wang2018revisiting`:

Under regularity conditions, for the Gaussian mechanism:

$$\sqrt{n}(\hat{\beta}_{DP} - \beta) \xrightarrow{d} \mathcal{N}\left(0, \sigma^2(X'X)^{-1} + \Sigma_{DP}\right)$$

Where $\Sigma_{DP}$ is the variance from the privacy mechanism.

**Standard Error Formula:**

$$SE(\hat{\beta}_{j, DP}) = \sqrt{\frac{\hat{\sigma}^2}{n \cdot \text{Var}(X_j)} + \text{Var}_{DP}(\hat{\beta}_j)}$$

**Advantages:**
- Simple closed-form
- Classical interpretation
- Fast computation

**Disadvantages:**
- Requires large samples
- May not hold for all DP mechanisms
- Finite-sample coverage may be poor

## Comparison of Methods

| Method | Theoretical Foundation | Computation | Finite Sample | Implementation |
|--------|----------------------|-------------|---------------|----------------|
| Noisy Sufficient Stats | Strong | Fast | Moderate | Partial |
| Bayesian Inference | Strong | Slow (MCMC) | Good | Limited |
| Measurement Error | Established | Fast | Moderate | Available |
| Bootstrap | Simulation-based | Slow | Good | Limited |
| Asymptotic Theory | Strong | Fast | Poor | None |
| **DP-SGD (This work)** | Heuristic | Medium | Calibrated | Full |

## Our Approach: DP-SGD with Empirical Calibration

statsmodels-sgd takes a pragmatic approach:

1. **Use DP-SGD** for coefficient estimation
2. **Track coefficient trajectory** during training
3. **Combine empirical and theoretical** variance estimates
4. **Calibrate** adjustment factors via simulation

**Variance Estimation:**

We use a weighted combination of:
- Empirical variance from coefficient trajectory
- Theoretical inflation based on noise parameters

$$\hat{SE}_{DP} = \hat{SE}_{OLS} \cdot \left(\alpha \cdot \sqrt{1 + \frac{\text{Var}_{empirical}}{\hat{SE}_{OLS}^2}} + (1-\alpha) \cdot f(\sigma, T, n)\right)$$

**Trade-offs:**

| Aspect | Our Approach | Principled Methods |
|--------|-------------|-------------------|
| Theoretical guarantees | Limited | Strong |
| Practical coverage | ~95% (calibrated) | Varies |
| Computational cost | Low | Often high |
| Implementation complexity | Low | High |
| Flexibility | High (any DP-SGD) | Limited |

## Recommendations

### For Research Requiring Formal Guarantees

Use **noisy sufficient statistics** with analytical variance:
- Strongest theoretical foundation
- Exact variance formulas available
- Best for peer-reviewed publications

### For Practical Applications

Consider:
- **svinfer/PrivacyUnbiased** for measurement error correction
- **statsmodels-sgd** for ease of use with calibrated inference
- **Bayesian methods** when you have prior information

### For Maximum Flexibility

Use **bootstrap methods**:
- Works with any DP mechanism
- No analytical derivations needed
- But computationally expensive

## Open Problems

Several challenges remain in DP inference:

1. **Finite-sample guarantees**: Most theory is asymptotic
2. **Heteroskedasticity**: Handling non-constant variance under DP
3. **Model selection**: Information criteria under DP
4. **Multiple testing**: Adjusting for many hypotheses
5. **Robustness**: Performance under model misspecification

## Conclusion

Differentially private inference is an active research area with multiple viable approaches. While no method is universally optimal, the choice depends on:

- Theoretical rigor requirements
- Computational constraints
- Available implementations
- Specific use case needs

statsmodels-sgd prioritizes practical usability with calibrated inference, while acknowledging that more theoretically grounded methods exist for applications requiring formal guarantees.
