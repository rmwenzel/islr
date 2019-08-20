---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

<h2>Table of Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#resampling-methods" data-toc-modified-id="Resampling-Methods-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Resampling Methods</a></span><ul class="toc-item"><li><span><a href="#cross-validation" data-toc-modified-id="Cross-validation-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Cross-validation</a></span><ul class="toc-item"><li><span><a href="#the-validation-set-approach" data-toc-modified-id="The-Validation-Set-Approach-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>The Validation Set Approach</a></span></li><li><span><a href="#leave-one-out-cross-validation" data-toc-modified-id="Leave-One-Out-Cross-Validation-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Leave-One-Out Cross Validation</a></span></li><li><span><a href="#k-fold-cross-validation" data-toc-modified-id="k-fold-Cross-Validation-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span> k-fold Cross-Validation </a></span></li><li><span><a href="#bias-variance-tradeoff-for-k-fold-cross-validation" data-toc-modified-id="Bias-Variance-Tradeoff-for-k-fold-Cross-Validation-5.1.4"><span class="toc-item-num">5.1.4&nbsp;&nbsp;</span>Bias-Variance Tradeoff for k-fold Cross Validation </a></span></li><li><span><a href="#cross-validation-on-classification-problems" data-toc-modified-id="Cross-Validation-on-Classification-Problems-5.1.5"><span class="toc-item-num">5.1.5&nbsp;&nbsp;</span>Cross-Validation on Classification Problems</a></span></li></ul></li><li><span><a href="#the-bootstrap" data-toc-modified-id="The-Bootstrap-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>The Bootstrap</a></span></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Footnotes</a></span></li></ul></li></ul></div>


- ***Resampling methods***  involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model

- Two of the most commonly used resampling methods are ***cross-validation*** and the bootstrap

- Resampling methods can be useful in ***model assessment***, the process of evaluating a model's performance, or in ***model selection***, the process of selecting the proper level of flexibility.

## Cross-validation

### The Validation Set Approach

- Randomly divide the data into a ***training set*** and ***validation set***. The model is fit on the training set and its prediction performance on the test set provides an estimate of overall performance. 

- In the case of a quantitative response, the prediction performance is measured by the mean-squared-error. The validation estimates the "true" $\text{MSE}$ with the mean-squared error $\text{MSE}_{validation}$ computed on the validation set.

##### Advantages

- conceptual simplicity
- ease of implementation
- low computational resources

##### Disadvantages

- the validation estimate is highly variable - it is highly dependent on the train/validation set split
- since the model is trained on a subset of the dataset, it may tend to overestimate the test error rate if it was trained on the entire dataset

### Leave-One-Out Cross Validation

Given paired observations $\mathcal{D} = \{(x_1, y_1), \dots, (x_n, y_n)\}$, for each $1 \leqslant i \leqslant n$:
- Divide the data $\mathcal{D}$ into a training set $\mathcal{D}_{i.} = \mathcal{D}\ \{(x_i, y_i)\}$ and a validation set $\{(x_i, y_i)\}$.
- Train a model $\mathcal{M}_i$ on $\mathcal{D}_{i.}$ and use it to predict $\hat{y}_i$.
- The LOOCV estimate for $\text{MSE}_{test}$ is
    
    $$CV_{(n)} = \frac{1}{n}\sum_{i=1}^n \text{MSE}_i$$
    
   where $\text{MSE}_i = (y_i - \hat{y}_i)$[^1]

##### Advantages

- approximately unbiased
- deterministic - doesn't depend on a random train/test split.
- computationally fast in least squares regression
   $$CV_{(n)} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2$$
   where $h_i$ is the [leverage](#High-Leverage-Points) of point i

##### Disdvantages

- Computationally expensive[^2] in general

### k-fold Cross-Validation

Given paired observations $\mathcal{D} = \{(x_1, y_1), \dots, (x_n, y_n)\}$, divide the data $\mathcal{D}$ into $K$ ***folds*** (sets) $\mathcal{D}_1, \dots, \mathcal{D}_K$ of roughly equal size.[^3] Then for each $1 \leqslant k \leqslant K$:
    
- Train a model on $\mathcal{M}_k$ on $\cup_{j\neq k} \mathcal{D}_{j}$ and validate on $\mathcal{D}_k$.
- The $k$-fold CV estimate for $\text{MSE}_{test}$ is
    
    $$CV_{(k)} = \frac{1}{k}\sum_{i=1}^k \text{MSE}_k$$
    
   where $\text{MSE}_k$ is the mean-squared-error on the validation set $\mathcal{D}_k$

##### Advantages

- computationally faster than $LOOCV$ if $k > 1$
- less variance than validation set approach or LOOCV

##### Disdvantages

- more biased than LOOCV if $k > 1$.

### Bias-Variance Tradeoff for k-fold Cross Validation

As $k \rightarrow n$, bias $\downarrow$ but variance $\uparrow$

### Cross-Validation on Classification Problems

In the classification setting, we define the LOOCV estimate

$$CV_{(n)} = \frac{1}{n}\sum_{i=1}^n \text{Err}_i$$

where $\text{Err}_i = I(y_i \neq \hat{y}_i)$. The $k$-fold CV and validation error rates are defined analogously.

## The Bootstrap

The bootstrap is a method for estimating the standard error of a statistic [^4] or statistical learning process. In the case of an estimator $\hat{S}$ for a statistic $S$ proceeds as follows:

Given a dataset $\mathcal{D}$ with $|\mathcal{D}=n|$, for $1 \leqslant i \leqslant B$:
- Create a bootstrap dataset $\mathcal{D}^\ast_i$ by sampling uniformly $n$ times from $\mathcal{D}$
- Calculate the statistic $S$ on $\mathcal{D}^\ast_i$ to get a bootstrap estimate $S^\ast_i$ of $S$

Then the bootstrap estimate for the $\mathbf{se}(S)$ the sample standard deviation of the boostrap estimates $S^\ast_1, \dots, S^\ast_B$:

$$\hat{se}(\hat{S}) = \sqrt{\frac{1}{B-1} \sum_{i = 1}^ B \left(S^\ast_i - \overline{S^\ast}\right)^2}$$

___
## Footnotes

[^1]: $\text{MSE}_i$ is just the mean-squared error of the model $\mathcal{M}_i$ on the validation set $\{(x_i, y_i)\}$. It is an approximately unbiased estimator of $\text{MSE}_{test}$ but it has high variance. But as the average of the $\text{MSE}_i$, $CV_{(n)}$ has much lower variance.

$CV_{(n)}$ is sometimes called the LOOCV error rate -- it can be seen as the average error rate over the singleton validation sets $\{(x_i, y_i)\}$

[^2]: Specifically $O(n * \text{model fit time})$

[^3]: LOOCV is then $k$-fold CV in the case $k=n$. Analogous, $CV_{k}$ is sometimes called the $k$-fold CV error rate, the average error over the folds.

[^4]: Recall a statistic $S$ is just a function of a sample $S = S(X_1,\dots, X_n)$

{% endkatexmm %}
