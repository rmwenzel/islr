---
layout: page
title: 6. Linear Model Selection and Regularization
---

{% katexmm %}

<div class="toc">
  <ul class="toc-item">
    <li><span><a href="#linear-model-selection-and-regularization" data-toc-modified-id="Linear-Model-Selection-and-Regularization-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Linear Model Selection and Regularization</a></span>
      <ul class="toc-item">
        <li><span><a href="#subset-selection" data-toc-modified-id="Subset-Selection-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Subset Selection</a></span>
          <ul class="toc-item">
            <li><span><a href="#best-subset-selection" data-toc-modified-id="Best-Subset-Selection-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Best Subset Selection</a></span>
              <ul class="toc-item">
                <li><span><a href="#advantages" data-toc-modified-id="Advantages-6.1.1.1"><span class="toc-item-num">6.1.1.1&nbsp;&nbsp;</span>Advantages</a></span></li>
                <li><span><a href="#disadvantages" data-toc-modified-id="Disadvantages-6.1.1.2"><span class="toc-item-num">6.1.1.2&nbsp;&nbsp;</span>Disadvantages</a></span></li>
              </ul>
            </li>
            <li><span><a href="#stepwise-selection" data-toc-modified-id="Stepwise-Selection-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>Stepwise Selection</a></span>
              <ul class="toc-item">
                <li><span><a href="#forward-stepwise-selection" data-toc-modified-id="Forward-Stepwise-Selection-6.1.2.1"><span class="toc-item-num">6.1.2.1&nbsp;&nbsp;</span>Forward Stepwise Selection</a></span></li>
                <li><span><a href="#backward-stepwise-selection" data-toc-modified-id="Backward-Stepwise-Selection-6.1.2.2"><span class="toc-item-num">6.1.2.2&nbsp;&nbsp;</span>Backward Stepwise Selection</a></span></li>
                <li><span><a href="#hybrid-approaches" data-toc-modified-id="Hybrid-Approaches-6.1.2.3"><span class="toc-item-num">6.1.2.3&nbsp;&nbsp;</span>Hybrid Approaches</a></span></li>
              </ul>
            </li>
            <li><span><a href="#choosing-the-optimal-model" data-toc-modified-id="Choosing-the-Optimal-Model-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>Choosing the Optimal Model</a></span>
              <ul class="toc-item">
                <li><span><a href="#cp-aic-bic-and-adjusted-r2" data-toc-modified-id="$C_p$,-AIC,-BIC-and-Adjusted-$R^2$-6.1.3.1"><span class="toc-item-num">6.1.3.1&nbsp;&nbsp;</span>$C_p$, AIC, BIC and Adjusted $R^2$</a></span></li>
                <li><span><a href="#validation-and-cross-validation" data-toc-modified-id="Validation-and-Cross-Validation-6.1.3.2"><span class="toc-item-num">6.1.3.2&nbsp;&nbsp;</span>Validation and Cross-Validation</a></span></li>
              </ul>
            </li>
          </ul>
        </li>
        <li><span><a href="#shrinkage-methods" data-toc-modified-id="Shrinkage-Methods-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Shrinkage Methods</a></span>
          <ul class="toc-item">
            <li><span><a href="#ridge-regression" data-toc-modified-id="Ridge-Regression-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Ridge Regression</a></span></li>
            <li><span><a href="#the-lasso" data-toc-modified-id="The-Lasso-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>The Lasso</a></span></li>
            <li><span><a href="#selecting-the-tuning-parameter" data-toc-modified-id="Selecting-the-Tuning-Parameter-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>Selecting the Tuning Parameter</a></span></li>
          </ul>
        </li>
        <li><span><a href="#dimension-reduction-methods" data-toc-modified-id="Dimension-Reduction-Methods-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Dimension Reduction Methods</a></span>
          <ul class="toc-item">
            <li><span><a href="#principal-components-regression" data-toc-modified-id="Principal-Components-Regression-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Principal Components Regression</a></span></li>
            <li><span><a href="#partial-least-squares" data-toc-modified-id="Partial-Least-Squares-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Partial Least Squares</a></span></li>
          </ul>
        </li>
        <li><span><a href="#considerations-in-high-dimensions" data-toc-modified-id="Considerations-in-High-Dimensions-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Considerations in High Dimensions</a></span>
          <ul class="toc-item">
            <li><span><a href="#high-dimensional-data" data-toc-modified-id="High-Dimensional-Data-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>High-Dimensional Data</a></span></li>
            <li><span><a href="#what-goes-wrong-in-high-dimensions?" data-toc-modified-id="What-Goes-Wrong-in-High-Dimensions?-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>What Goes Wrong in High Dimensions?</a></span></li>
            <li><span><a href="#regression-in-high-dimensions" data-toc-modified-id="Regression-in-High-Dimensions-6.4.3"><span class="toc-item-num">6.4.3&nbsp;&nbsp;</span>Regression in High Dimensions</a></span></li>
            <li><span><a href="#interpreting-results-in-high-dimensions" data-toc-modified-id="Interpreting-Results-in-High-Dimensions-6.4.4"><span class="toc-item-num">6.4.4&nbsp;&nbsp;</span>Interpreting Results in High Dimensions</a></span></li>
          </ul>
        </li>
        <li><span><a href="#footnotes" data-toc-modified-id="Footnotes-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Footnotes</a></span></li>
      </ul>
    </li>
  </ul>
</div>

Alternatives to the least squares fitting procedures can yield 
better

- prediction accuracy
- model interpretability

## Subset Selection

Methods for selecting a subset of the predictors to improve test performance

### Best Subset Selection

###### Algorithm: Best Subset Selection (BSS) for linear regression 


1. Let $\mathcal{M}_0$ denote the null model[^1]
2. For $1 \leqslant k \leqslant p$:
    1. Fit all $\binom{p}{k}$ linear regression models with $k$ predictors
    2. Let $\mathcal{M}_k = \underset{\text{models}}{\text{argmin}}\ RSS$
3. Choose the best model $\mathcal{M}_i, 1 \leqslant i \leqslant p$ based on estimated test error [^2]

For logistic regression, in step 2.A., let $\mathcal{M}_k = \underset{\text{models}}{\text{argmin}}\ D(y, \hat{y})$ where $D(y, \hat{y})$  is the ***deviance***[^3] of the model

#### Advantages

- Slightly faster than brute force. Model evaluation is $O(p)$ as opposed to $O(2^p)$ for brute force.
- Conceptually simple

#### Disadvantages

- Still very slow. Fitting is $O(2^p)$ as for brute force
- Overfitting and high variance of coefficient estimates when $p$ is large

### Stepwise Selection

#### Forward Stepwise Selection

##### Algorithm: ***Forward Stepwise Selection*** (FSS) for linear regression[^4]

1. Let $\mathcal{M}_0$ denote the null model
2. For $0 \leqslant k \leqslant p - 1$:
    1. Fit all $p-k$ linear regression models that augment model $\mathcal{M}_k$ with one additional predictor
    2. Let $\mathcal{M}_{k+1} = \underset{\text{models}}{\text{argmin}}\ RSS$
3. Choose the best model $\mathcal{M}_i, 1 \leqslant i \leqslant p$ based on estimated test error

##### Advantages

- Faster than BSS. Fitting is $O(p^2)$ and evaluation is $O(p)$
- Can be applied in the high-dimensional setting $n < p$

##### Disadvantages

- Evaluation is more challenging since it compares models with different numbers of predictors.
- Searches less of the parameter space, hence may be suboptimal

#### Backward Stepwise Selection

##### Algorithm: Backward Stepwise Selection (BKSS) for linear regression [^5] is

1. Let $\mathcal{M}_p$ denote the full model [^6]
2. For $k = p, p-1, \dots, 1$:
    1. Fit all $k$ linear regression models of $k-1$ predictors that contain all but one of the predictors in $\mathcal{M}_k$.
    2. Let $\mathcal{M}_{k-1} = \underset{\text{models}}{\text{argmin}}\ RSS$
3. Choose the best model $\mathcal{M}_i, 1 \leqslant i \leqslant p$ based on estimated test error 

##### Advantages

- As fast as FSS

##### Disadvantages

- Same disadvantages as FSS
- Cannot be used when $n < p$

#### Hybrid Approaches

Other approaches exist which may add variables sequentially (as with FSS) but may also remove variables (as with BSS). These methods strike a balance between optimality (e.g. BSS) and speed (FSS/BSS)

### Choosing the Optimal Model

Two common approaches to estimating the test error:

1. Estimate indirectly by adjusting the training error to account for overfitting bias
2. Estimate directly using a validation approach

#### $C_p$, AIC, BIC and Adjusted $R^2$

- Train MSE underestimates test MSE and decreases as $p$ increases, so it cannot be used to select from models with different numbers of predictors. However we may adjust the training error to account for the model size, and use this to estimate the test MSE

- For least squares models, the ***$C_p$*** estimate[^7] of the test MSE for a model with $d$ predictors is
$$ C_p = \frac{1}{n}(RSS + 2d\hat{\sigma}^2) $$

  where $\hat{\sigma} = \hat{\mathbb{V}}(\epsilon)$.

- For maximum likelihood models[^8], the ***Akaike Information Criterion*** (AIC) estimate of the test MSE is

$$ AIC = \frac{1}{n\hat{\sigma}^2}(RSS + 2d\hat{\sigma}^2) $$

- For least squares models, the ***Bayes Information Criterion*** (BIC) estimate[^9] of the test MSE is

$$ BIC = \frac{1}{n}(RSS + \log(n)d\hat{\sigma}^2) $$

- For least squares models, the ***adjusted $R^2$*** statistic[^10] is

$$AdjR^2 = 1 - \frac{RSS/(n - d - 1)}{TSS/(n - 1)}$$


#### Validation and Cross-Validation

- Instead using adjusted training error to estimate test error indirectly, we can directly estimate using validation or cross-validation
- In the past this was computationally prohibitive but advances in computation have made this method very attractive.
- In this approach, we can select a model using the ***one-standard-error*** rule, i.e. selecting the model for which the estimated standard error is within one standard error of the $p$ vs. error curve.

## Shrinkage Methods

Methods for constraining or ***regularizing*** the coefficient estimates, i.e. ***shrinking*** them towards zero. This can significantly reduce their variance.

### Ridge Regression

- Ridge regression introduces an $L^2$-penalty[^11] for the training error and estimates
$$\hat{\beta}^R = RSS+\lambda\|\tilde{\beta}\|_2^2$$
    where $\lambda$ is a ***tuning parameter***[^12] and $\tilde{\beta} = (\beta_1, \dots, \beta_p)$[^13].

- The term $\lambda\|\beta\|_2^2$ is called a ***shrinkage penalty*** 

- Selecting a good value for $\lambda$ is critical, see section 6.2.3

- Standardizing the predictors $X_i \mapsto \frac{X_i - \mu_i}{s_i}$ is advised.

##### Advantages

- Takes advantage of bias-variance tradeoff by decreasing flexibility [^14] thus decreasing variance.
- Preferable to least squares in situations when the latter has high variance (close to linear relationship, $p \lesssim n$
- In contrast to least squares, works when $p > n$ 

##### Disadvantages

- Lower variance means higher bias.
- Will not eliminate any predictors which can be an issue for interpretation when $p$ is large.

### The Lasso

- Lasso regression introduces an $L^1$-penalty [^15] for the training error and estimates

$$\hat{\beta}^R = RSS+\lambda\|\tilde{\beta}\|^2_1$$

##### Advantages

- Same advantages as ridge regression.
- Improves over ridge regression by yielding ***sparse models*** (i.e. performs variable selection) when $\lambda$ is sufficiently large

##### Disadvantages

- Lower variance means higher bias.

##### Another Formulation for Ridge Regression and the Lasso

- Ridge Regression is equivalent to the quadratic optimization problem:

$$ \begin{aligned}
\min&\ RSS + \|\tilde{\beta}\|_2\\
\text{s.t.}&\ \| \tilde{\beta} \|_2^2 \leqslant s \\
\end{aligned} $$

- Lasso Regression is equivalent to the quadratic optimization problem:

$$ \begin{aligned}
\min&\ RSS + \|\tilde{\beta}\|_1\\
\text{s.t.}&\ \| \tilde{\beta} \|_1 \leqslant s \\
\end{aligned} $$

##### Bayesian Interpretation for Ridge and Lasso Regression

Given Gaussian errors, and simple assumptions on the prior $p(\beta)$, ridge and lasso regression emerge as solutions

- If the $\beta_i \sim \text{Normal}(0, h(\lambda))$ iid for some function $h=h(\lambda)$ then the ***posterior mode*** for $\beta$ (i.e. $\underset{\beta}{\text{argmax}} p(\beta| X, Y)$) is the ridge regression solution

- If the $\beta_i \sim \text{Laplace}(0, h(\lambda))$ iid then the posterior mode is the lasso regression solution.

### Selecting the Tuning Parameter

Compute the cross-validation error $CV_{(n),i}$ for for a "grid" (evenly-spaced discrete set) of values $\lambda_i$, and choose

$$ \lambda = \underset{i}{\text{argmin}\ CV_{(n),i}}$$

## Dimension Reduction Methods

- ***Dimension reduction*** methods transform the predictors $X_1, \dots, X_p$ into a smaller set of predictors 
  $Z_1, \dots, Z_M$, $M < p$.
  
- When $p >> n$, $M << p$ can greatly reduce the variance of the coefficient estimates.

- In this section we consider linear transformations 

  $$Z_m = \sum_{j = 1}^p \phi_{jm}X_j$$
  
  and a least squares regression model
  
  $$ Y = \mathbf{Z}\theta + \epsilon $$
  
  where $\mathbf{Z} = (1,Z_1, \dots, Z_M)$

### Principal Components Regression

***Principal Components Analysis*** is a popular unsupervised approach [^16] that can be used for dimensional reduction

##### An Overview of Principal Components Analysis

- The ***principal components*** of a data matrix $n\times p$ matrix $\mathbf{X}$ can be seen ([among many different perspectives](https://en.wikipedia.org/wiki/Principal_component_analysis)) as the [right singular eigenvectors](https://en.wikipedia.org/wiki/Principal_component_analysis#Computing_PCA_using_the_covariance_method) $v_1, \dots, v_p$ of the $p\times p$ sample covariance matrix $C$, i.e. [the eigenvectors of $C^{\top}C$](https://en.wikipedia.org/wiki/Singular_value_decomposition)) ordered by decreasing absolute value of the corresponding eigenvalues.

- Let $\sigma_1^2,\dots, \sigma_k^2$ be the singular values of $C$ ([the squares of the eigenvalues of $C^{\top}C$](https://people.cs.pitt.ed0u/~milos/courses/cs3750-Fall2007/lectures/PCA.pdf)) and let $v_1, \dots, v_p$ be the corresponding eigenvectors of $C$. Then $\sigma_i^2$ is the variance of the data along the direction $v_i$, and $\sigma_1^2$ is the direction of maximal variance.

##### The Principal Components Regression Approach

- ***Principal Components Regression*** takes $Z_1,\dots, Z_M$ to be the first $M$ principal components of $\mathbf{X}$ and then fits a least squares model on these components. 
- The assumption is that, since the principal components correspond to the directions of greatest variation of the data, they show the most association with $Y$. Furthermore, they are ordered by decreasing magnitude of association.
- Typically $M$ is chosen by cross-validation.

###### Advantages

- If the assumption holds then the least squares model on $Z_1, \dots, Z_M$ will perform better than $X_1, \dots, X_p$, since it will contain most of the information related to the response [^17], and by choosing $M << p$ we can mitigate overfitting.

- Decreased variance of coefficient estimates relative to OLS regression

###### Disadvantages

- Is not a feature selection method, since each $Z_i$ is a linear function of the predictors

###### Recommendations

- Data should usually be standarized prior to finding the principal components.

### Partial Least Squares

A supervised dimension reduction method which [proceeds roughly as follows](https://en.wikipedia.org/wiki/Partial_least_squares_regression)

- Standardize the variables
- Compute $Z_1$ by setting $\phi_{j1} = \hat{\beta_j}$ the ordinary least squares estimate
- For $ 1 < m < M$, $Z_m$ is determined by
    - Adjust the data $X_j = \epsilon_j$ where $\epsilon_j$ is the residual from regression of $Z_{m - 1}$ onto $X_j$
    - Compute $Z_m$ in the same fashion as $Z_1$ on the adjusted data
    
As with PCR, $M$ is chosen by cross-validation

##### Advantages

- Decreased variance of coefficient estimates relative to OLS regression
- Supervised dimension reduction may reduce bias

##### Disadvantages

- May increase variance relative to PCR (which is unsupervised). 
- May be no better than PCR in practice

## Considerations in High Dimensions

### High-Dimensional Data

Low dimensional means $p << n$, high dimensional is $p \gtrsim n$

### What Goes Wrong in High Dimensions?

- If $p \gtrsim n$, then linear models will create a perfect fit, hence overfit (usually badly)

- $C_p$, $AIC$, $BIC$, and $R^2$ approaches don't work in well in this setting

### Regression in High Dimensions

1. Regularization or shrinkage plays a key role in high-dimensional problems.
2. Appropriate tuning parameter selection is crucial for good predictive performance.
3. The test error tends to increase as the dimensionality of the problem increases if the additional features aren't truly associated with the response (the curse of dimensionality)

### Interpreting Results in High Dimensions

- Multicollinearity problem is maximal in high dimensional setting

- This makes interpretation difficult, since models obtained from highly multicollinear data fail to identify which features are "preferred"

- Care must be taken to measure performance [^18]

___
## Footnotes


[^1]: This is the model that predicts $\hat{y} = \overline{y}$, i.e. $\hat{\beta_i} = 0$ for $i > 1$ and $\hat{\beta_0} = \overline{y}$.

[^2]: Estimates of test error can come from CV, $C_p (AIC)$, $BIC$ or adjusted $R^2$

[^3]: Here $D(y, \hat{y}) = -2\log(p(y\ |\ \hat{\beta})$ where $\hat{\beta}$ is the MLE for $\beta$. The author's definition of deviance can be found in the comment on the [Wikipedia entry](https://en.wikipedia.org/wiki/Deviance_(statistics)) if $\hat{\theta}_0$.

[^4]: As with BSS, we can use FSS for logistic regression by replacing $RSS$ with the deviance in step 2B.

[^5]: As with BSS, we can use BackSS for logistic regression by replacing $RSS$ with the deviance in step 2B.

[^6]: Here full means contains all $p$ predictors.

[^7]: Thus $C_p$ is RSS plus a penalty which depends on the number of predictors and the estimate of the error variance. One can show that if $\hat{\sigma}^2$ is unbiased then then $C_p$ is unbiased.

[^8]: For Gaussian errors, the least squares estimate is the maximumlikelihood estimate so in that case $C_p$ and $AIC$ are proportional.

[^9]: The BIC places a heavier penalty than $C_p$ when $n > 7$ due to the $\log(n)d\hat{\sigma}^2$ term. The book says this means BIC places a heavier penalty than $C_p$ on models with many variables although this isn't clear. It would seem it places a penalty on large numbers of observation (unless somehow larger numbers of observations are correlated with larger numbers of predictors).

[^10]: $C_p, AIC$ and $BIC$ are all estimates of the test $MSE$ so smaller values are better. By contrast, larger values of adjusted $R^2$, but this is equivalent to minimizing $RSS/(n - d - 1)$ which likely can be thought of as a test MSE estimate.

[^11]: Here $L^2$ is a reference to the [$L^p$ norm](https://en.wikipedia.org/wiki/Lp_space) (denoted $\| \cdot \|_2$) when $p=2$ (see also p216), which is just the standard Euclidean norm.

[^12]: The tuning parameter $\lambda$ is actually a Lagrange multiplier used to turn a constrained optimization problem into an unconstrained optimization problem -- see [above](#another-formulation-for-ridge-regression-and-the-lasso)

[^13]: We use $\tilde{\beta}$ instead of $\beta$ because we don't want to shrink the intercept $\beta_0$. If the data have been centered about their mean then $\hat{\beta}_0 = \overline{y}$

[^14]: Flexibility decreases because the shrinkage penalty effectively decreases the size of the parameter space?

[^15]: The [$L^1$ norm](https://en.wikipedia.org/wiki/Taxicab_geometry) is $\|\tilde{\beta}\|_1 = \sum_{i=1}^p |\beta_p|$

[^16]: Unsupervised since it only takes the predictors $\mathbf{X}$ and not the response $\mathbf{Y}$ as input.

[^17]: Under certain assumptions, PCA is an [optimal dimension reduction method from an information theoretic perspective](https://en.wikipedia.org/wiki/Principal_component_analysis#PCA_and_information_theory)

[^18]: For example SSE, $p$-values, and $R^2$ statistics from the training data are useless in this setting. Thus it is important to e.g. evaluate performance on an independent test set or use [resampling methods](#Resampling-Methods)
    
{% endkatexmm %}
