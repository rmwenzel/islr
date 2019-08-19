---
layout: page
title: 4. Logistic Regression
---

{% katexmm %}

<h2>Table of Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#logistic-regression" data-toc-modified-id="Logistic-Regression-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Logistic Regression</a></span><ul class="toc-item"><li><span><a href="#an-overview-of-classification" data-toc-modified-id="An-Overview-of-Classification-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>An Overview of Classification</a></span></li><li><span><a href="#why-not-linear-regression?" data-toc-modified-id="Why-Not-Linear-Regression?-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Why Not Linear Regression?</a></span></li><li><span><a href="#logistic-regression" data-toc-modified-id="Logistic-Regression-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Logistic Regression</a></span><ul class="toc-item"><li><span><a href="#the-logistic-model" data-toc-modified-id="The-Logistic-Model-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>The Logistic Model</a></span></li><li><span><a href="#estimating-the-regression-coefficients" data-toc-modified-id="Estimating-the-Regression-Coefficients-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Estimating the Regression Coefficients</a></span></li><li><span><a href="#making-predictions" data-toc-modified-id="Making-Predictions-4.3.3"><span class="toc-item-num">4.3.3&nbsp;&nbsp;</span>Making Predictions</a></span></li><li><span><a href="#multiple-logistic-regression" data-toc-modified-id="Multiple-Logistic-Regression-4.3.4"><span class="toc-item-num">4.3.4&nbsp;&nbsp;</span>Multiple Logistic Regression</a></span></li><li><span><a href="#logistic-regression-for-more-than-two-response-classes" data-toc-modified-id="Logistic-Regression-for-more-than-two-response-classes-4.3.5"><span class="toc-item-num">4.3.5&nbsp;&nbsp;</span>Logistic Regression for more than two response classes</a></span></li></ul></li><li><span><a href="#linear-discriminant-analysis" data-toc-modified-id="Linear-Discriminant-Analysis-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Linear Discriminant Analysis</a></span><ul class="toc-item"><li><span><a href="#bayes-theorem-for-classification" data-toc-modified-id="Bayes-Theorem-for-Classification-4.4.1"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>Bayes Theorem for Classification</a></span></li><li><span><a href="#linear-discriminant-analysis-for-p=1" data-toc-modified-id="Linear-Discriminant-Analysis-for-p=1-4.4.2"><span class="toc-item-num">4.4.2&nbsp;&nbsp;</span>Linear Discriminant Analysis for p=1</a></span></li><li><span><a href="#linear-discriminant-analysis-for-p->-1" data-toc-modified-id="Linear-Discriminant-Analysis-for-p->-1-4.4.3"><span class="toc-item-num">4.4.3&nbsp;&nbsp;</span>Linear Discriminant Analysis for p &gt; 1</a></span></li><li><span><a href="#quadratic-discriminant-analysis" data-toc-modified-id="Quadratic-Discriminant-Analysis-4.4.4"><span class="toc-item-num">4.4.4&nbsp;&nbsp;</span>Quadratic Discriminant Analysis</a></span></li></ul></li><li><span><a href="#a-comparison-of-classification-methods" data-toc-modified-id="A-Comparison-of-Classification-Methods-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>A Comparison of Classification Methods</a></span></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Footnotes</a></span></li></ul></li></ul></div>


## An Overview of Classification

- In ***classification*** we consider paired data $(\mathbf{X}, Y)$, where $Y$ is a ***qualitative variable***, that is, a finite random variable.

- The values $Y$ takes are called ***classes***

## Why Not Linear Regression?

Because a linear regression model implies an ordering on the values of the response and in general there is no natural ordering on the values of a qualitative variable

## Logistic Regression

### The Logistic Model

- Consider a quantitiative predictor $X$ binary response variable $Y \in {0,1}$

- We want to model the conditional probability of $Y=1$ given $X$

$$ P(X) := P\left(Y=1 | X\right)$$

- We model $P(X)$ with the ***logistic function***

$$ P(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} $$

- The logistic model can be considered a linear model for the ***log-odds*** or ***logit***

$$\log\left(\frac{P(X)}{1 - P(X)}\right) = \beta_0 + \beta_1 X$$

### Estimating the Regression Coefficients

- The likelihood function for the logistic regression parameter $\boldsymbol{\beta} = (\beta_0, \beta_1)$ is

$$
\begin{aligned}
\ell(\boldsymbol{\beta}) &= \prod_{i = 1}^n p(x_i)\\
&= \prod_{i: y_i = 1}p(x_i) \prod_{i: y_i = 0} (1 - p(x_i))
\end{aligned}
$$

- The maximum likelihood estimate (MLE) for the regression parameter is

$$ \hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}\in \mathbb{R}^2}{\text{argmax}\,} \ell(\boldsymbol{\beta})$$

- There isn't a closed form solution for $\hat{\boldsymbol{\beta}}$ so [it must be found using numerical methods](https://en.wikipedia.org/wiki/Logistic_regression#Maximum_likelihood_estimation)

### Making Predictions


- The MLE $\hat{\boldsymbol{\beta}}$ results in an estimate for the conditional probability $\hat{P}(X)$ which can be used to predict the class $Y$

### Multiple Logistic Regression

- Multiple logistic regression considers the case of multiple predictors $\mathbf{X} = (X_1,\dots, X_p)^\top$.

- If we write the predictors as $\mathbf{X} = (1, X_1, \dots, X_p)^\top$, and the parameter $\boldsymbol(\beta) = (\beta_0, \dots, \beta_p)^\top$ then multiple logistic regression models

$$p(X) = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{X})}{1 + \exp(\boldsymbol{\beta}^\top \mathbf{X})} $$

### Logistic Regression for more than two response classes

- This isn't used often (a [softmax](https://en.wikipedia.org/wiki/Softmax_function) is often used)

## Linear Discriminant Analysis

This is a method for modeling the conditional probability of a qualitative response $Y$ given quantitative predictors $\mathbf{X}$ when $Y$ takes more than two values. It is useful because:

- Parameter estimates for logistic regression are suprisingly unstable when the classes are well separated, but LDA doesn't have this problem

- If $n$ is small and the $X_i$ are approximately normal in the classes (i.e. the conditional $X_i | Y = k$ is approximately normal) LDA is more stable

- LDA can accomodate more than two clases

### Bayes Theorem for Classification

- Consider a quantitiative input $\mathbf{X}$ and qualitative response $Y \in {1, \dots K}$. 

- Let $\pi_k := \mathbb{P}(Y = k)$ be the prior probability that $Y=k$, let $p_k(x) := \mathbb{P}(Y = k\ |\ X = x)$ be the posterior probability that $Y = k$, and let $f_k(x):= \mathbb{P}(X = x\ |\ Y = k)$. Then Bayes' theorem says:

$$ p_k(x) = \frac{\pi_k f_k(x)}{\sum_{l}\pi_l f_l(x)} $$

- We can form an estimate $\hat{p}_k(x)$ for $p_k(x)$ with estimates of $\pi_k$ and $f_k(x)$ for each k, and for $x$ predicts [^1]  

$$\hat{y} = \underset{1 \leqslant k \leqslant K}{\text{argmax}\,} \hat{p}_k(x) $$


### Linear Discriminant Analysis for p=1

- Assume that the conditional $ X| Y = k \sim \text{Normal}(\mu_k, \sigma_k^2)$ and that the variances are equal across classes $\sigma_1^2 = \cdots = \sigma_K^2 = \sigma^2$.

- The Bayes classifier  predicts $Y = k$ where $p_k(x)$ is largest or equivalently

$$
\begin{aligned}
\hat{y}
&= \underset{1 \leqslant k \leqslant K}{\text{argmax}\ } \delta_k(x)\\
\end{aligned}
$$
    where
$$ \delta_k(x) := \left(\frac{\mu_k}{\sigma^2}\right) - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)$$
    is the ***discriminant function***[^2]

- The LDA classifier [^3] estimates the parameters 

$$
\begin{aligned}
\hat{\mu}_k &= \frac{1}{n_k}\sum_{i: y_i = k} x_i\\
\hat{\sigma}_k &= \frac{1}{n-K} \sum_{k = 1}^K \sum_{i: y_i = k} \left(x_i - \hat{\mu}_k\right)^2
\end{aligned}
$$

Where $n_k$ is the number of observations in class $k$ [^4] and predicts 
 
$$ \begin{aligned}
\hat{y} &= \underset{1 \leqslant k \leqslant K}{\text{argmax}\ } \hat{\sigma}_k(x)) \\
\end{aligned} $$


### Linear Discriminant Analysis for p > 1

- Assume that the conditional $(\mathbf{X}| Y = k) \sim \text{Normal}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ and that the covariance matrices are equal across classes $\boldsymbol{\Sigma}_1 = \cdots = \boldsymbol{\Sigma}_K = \boldsymbol{\Sigma}$.

- The discriminant functions are

$$\sigma_k(x) = x^\top\boldsymbol{\Sigma}^{-1}\mu_k - \frac{1}{2}\mu_k {\Sigma}^{-1} \mu_k + \log(\pi_k)$$

- LDA estimates $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$ [^5] componentwise 
$$ \begin{aligned}
(\hat{\mu}_k)_j &= \frac{1}{n_k}\sum_{i: y_i = k} x_{ij}\\
(\hat{\sigma}_k)_j &= \frac{1}{n-K} \sum_{k = 1}^K \sum_{i: y_i = k} \left(x_{ij} - (\hat{\mu}_k)_j\right)^2
\end{aligned} $$
    for $1 \leqslant j \leqslant p$ as above and predicts
$$ \begin{aligned}
\hat{y} &= \underset{1 \leqslant k \leqslant K}{\text{argmax}\ } \hat{\sigma}_k(x)) \\
\end{aligned} $$
    as above

- Confusion matrices help analyze misclassifications for an LDA model [^6]

- The Bayes decision boundary may not be agreeable in every context so sometimes a different decsision boundary (threshold) is used. 

- An ROC curve is useful for vizualising true vs false positives over different decision thresholds in the binary response case.

### Quadratic Discriminant Analysis

- Assume that the conditional $(\mathbf{X}| Y = k) \sim \text{Normal}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ but assume that the covariance matrices $\boldsymbol{\Sigma}_k$ are not equal across classes 

- The discriminant functions are now quadratic in $x$

$$\sigma_k(x) = x^\top\boldsymbol{\Sigma}_k^{-1}\mu_k - \frac{1}{2}\mu_k {\Sigma}_k^{-1} \mu_k + \log(\pi_k)$$

- QDA has more degrees of freedom than LDA [^7] so generally has lower bias but higher variance.

## A Comparison of Classification Methods

- So far our classification methods are KNN, logistic regression (LogReg), LDA and QDA

- LDA and LogReg both produce linear decision boundaries. They often give similar performance results, although LDA tends to outperform when the conditionals $X | Y = k$ are normally distributed, and not when they aren't

- As a non-parametric approach, KNN produces a non-linear decision boundary, so tends to outperform LDA and LogReg when the true decision boundary is highly non-linear. It doesn't help with selecting important predictors

- With a quadratic decision boundary, QDA is a compromise between the non-linear KNN and the linear LDA/LogReg

___
## Footnotes

[^1]: Recall that that the Bayes classifier predicts
    $$\hat{y} = \underset{1 \leqslant k \leqslant K}{\text{argmax}\,} p_k(x) $$
     So we can think of LDA as an esimate of the Bayes Classifier

[^2]: The Bayes decision boundary corresponds to the parameter values $\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)$, $\boldsymbol{\sigma} = (\sigma_1, \dots, \sigma_K)$ such that $\delta_k(x) = \delta_j(x)$ for all $1 \leqslant j,k \leqslant K$. The Bayes classifier assigns a class $y$ to an input $x$ based on where $x$ falls with respect to this boundary.

[^3]: For the case $K=2$, this is equivalent to assigning $x$ to class 1 if $2x(\mu_1 - \mu_2) > \mu_1^2 - \mu_2^2$.

[^4]: The functions $\hat{\delta}_k(x)$ are called ***discriminant*** functions, and since they're linear in $x$, the method is called linear discriminant analysis.

[^5]: $\hat{\mu}_k$ is the average of all observation inputs in the $k$-th class, and $\hat{\sigma}_k$ is a weighted average of the samples variances over the $K$ classes.

[^6]: False positives and false negatives in the binary case.
[^7]: $K\binom{p}{2}$ for QDA versus $\binom{p}{2}$ for LDA.


{% endkatexmm %}