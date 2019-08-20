---
layout: page
title: 2. Statistical Learning
---

{% katexmm %}

<h2>Table of Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#statistical-learning" data-toc-modified-id="Statistical-Learning-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Statistical Learning</a></span><ul class="toc-item"><li><span><a href="#what-is-statistical-learning" data-toc-modified-id="What-is-Statistical-Learning?-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>What is Statistical Learning?</a></span><ul class="toc-item"><li><span><a href="#why-estimate-f" data-toc-modified-id="Why-Estimate-f?-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Why Estimate f?</a></span></li><li><span><a href="#how-to-estimate-f" data-toc-modified-id="How-to-Estimate-f?-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>How to Estimate f?</a></span></li><li><span><a href="#accuracy-vs-interpretability" data-toc-modified-id="Accuracy-vs.-Interpretability-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Accuracy vs. Interpretability</a></span></li><li><span><a href="#supervised-vs-unsupervised-learning" data-toc-modified-id="Supervised-vs.-Unsupervised-Learning-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>Supervised vs. Unsupervised Learning</a></span></li><li><span><a href="#regression-vs-classification" data-toc-modified-id="Regression-vs.-Classification-2.1.5"><span class="toc-item-num">2.1.5&nbsp;&nbsp;</span>Regression vs. Classification</a></span></li></ul></li><li><span><a href="#assessing-model-accuracy" data-toc-modified-id="Assessing-Model-Accuracy-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Assessing Model Accuracy</a></span><ul class="toc-item"><li><span><a href="#measuring-quality-of-fit" data-toc-modified-id="Measuring-Quality-of-Fit-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Measuring Quality of Fit</a></span></li><li><span><a href="#the-bias-variance-tradeoff" data-toc-modified-id="The-Bias-Variance-Tradeoff-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>The Bias-Variance Tradeoff</a></span></li><li><span><a href="#the-classification-setting" data-toc-modified-id="The-Classification-Setting-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>The Classification Setting</a></span></li></ul></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Footnotes</a></span></li></ul></li></ul></div>

## What is Statistical Learning?

Given paired data  $(X, Y)$, assume a relationship between $X$[^0] and $Y$ modeled by

$$ Y = f(X) + \epsilon $$

where $f:\mathbb{R}^p \rightarrow \mathbb{R}$ is a function and $\epsilon$ is a random error term with $\mathbb{E}(\epsilon) = 0$.

***Statistical learning*** is a set of approaches for estimating $f$[^1]

### Why Estimate f?

##### Prediction

- We may want to ***predict*** the output $Y$ from an estimate $\hat{f}$ of $f$. The predicted value for a given $Y$ is then $$ \hat{Y} = \hat{f}(X)$$. In prediction, we often treat $f$ as a ***black-box***

- The mean squared-error[^2] $\mathbf{mse}(\hat{Y})=\mathbb{E}(Y-\hat{Y})^2$ is a good measure of the accuracy of $\hat{Y}$ as a predictor for $Y$.

- One can write

$$ \mathbf{mse}(\hat{Y}) = (f(X) - \hat{f}(X))^2 + \mathbb{V}(\epsilon) $$

These two terms are known as the ***reducible error*** and ***irreducible error***, respectively[^3]

##### Inference

- Instead of predicting $Y$ from $X$, we may be more interested how $Y$ changes as a function of $X$. In inference, we usually do not treat $f$ as a black box. 

Examples of important inference questions:

- *Which predictors have the largest influence on the response?*
- *What is the relationship between the response and each predictor?*
- *Is f linear or non-linear?*

### How to Estimate f?

##### Parametric methods

Steps for parametric method:

1. Assume a parametric model for $f$, that is assume a specific functional form[^4]

$$f = f(X, \boldsymbol{\beta}) $$

for some vector of ***parameters*** $\boldsymbol{\beta} = (\beta_1,\dots,\beta_p)^T$  

2. Use the training data to ***fit*** or ***train*** the model, that is to choose $\beta_i$ such that 

$$Y \approx f(X, \boldsymbol{\beta})$$

##### Non-parametric methods

These methods make no assumptions about the functional form of $f$.

### Accuracy vs. Interpretability

- In inference, generally speaking the more flexible the method, the less interpretable.

- In prediction, generally speaking the more flexible the method, the less accurate

### Supervised vs. Unsupervised Learning

- In ***supervised learning***, training data consists of pairs $(X, Y)$ where $X$ is a vector of predictors and $Y$ a response. Prediction and inference are supervised learning problems, and the response variable (or the relationship between the response and the predictors) *supervises* the analysis of model 

- In ***unsupervised learning***, training data lacks a response variable.

### Regression vs. Classification

- Problems with a quantitative response ($Y\in S \subseteq \mathbb{R}$) tend to be called ***regression*** problems

- Problems with a qualitative, or categorical response ($Y \in \{y_1, \dots, y_n\})$ tend to be called ***classification*** problems

## Assessing Model Accuracy

There is no free lunch in statistics

### Measuring Quality of Fit

- To evaluate the performance of a method on a data set, we need measure model accuracy (how well predictions match observed data). 

- In regression, the most common measure is the ***mean-squared error***

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{f}(x_i))^2$$

  where $y_i$ and $\hat{f}(x_i)$ are the $i$ true and predicting   
  responses, respectively.

- We are usually not interested in minimizing MSE with respect to training data but rather to test data. 

- There is no guarantee low training MSE will translate to low test MSE. 

- Having low training MSE but high test MSE is called ***overfitting***

### The Bias-Variance Tradeoff

- For a given $x_0$, the expected [^5] MSE can be written

$$ \begin{aligned}
\mathbb{E}\left[y_0 - \hat{f}(x_0))^2\right] 
&= )\mathbb{E}\left[\hat{f}(x) \right] - f(x))^2 + \mathbb{E}\left[\hat{f}(x_0) - \mathbb{E}\left[\hat{f}(x_0)\right])^2\right] + \mathbb{E}\left[\epsilon - \mathbb{E}[\epsilon])^2\right]\\
&= \mathbf{bias}^2)\hat{f}(x_0))) + \mathbb{V})\hat{f}(x_0)) + \mathbb{V}(\epsilon)
\end{aligned} $$


- A good method minimizes variance and bias simultaneously. 

- As a general rule, these quantities are inversely proportional. More flexible methods have lower bias but higher variance, while less flexible methods have the opposite. This is the ***bias-variance tradeoff*** 

- In practice the mse, variance and bias cannot be calculated exactly but one must keep the bias-variance tradeoff in mind.

### The Classification Setting

- In the classification setting, the most common measure of model accuracy is the ***error rate*** [^6]

$$\frac{1}{n}\sum_{i=1}^n I(y_i \neq \hat{y}_i)$$ 
- As with the regression, we are interested in minimizing the test error rate, not the training error rate.

##### The Bayes Classifier

- Given $K$ classes, the ***Bayes Classifier*** predicts

$$ \hat{y_0} = \underset{1\leqslant j \leqslant K}{\text{argmax}\,} \mathbb{P}(Y=j\ |\ X = x_0)$$

- The set of points 
$$\{x_0\in\mathbb{R}^p\ |\ \mathbb{P}(Y=j\ |\ X = x_0) = \mathbb{P})Y=k\ |\ X = x_0)\ \text{for all}\ 1\leqslant j,k \leqslant K\}$$

    is called the ***Bayes decision boundary***

- The test error rate of the Bayes classifier is the ***Bayes error rate***, which is minimal among classifiers. It is given by

$$ 1 - \mathbb{E})\underset{j}{\max} \mathbb{P}(Y=j\ |\ X))$$

- The Bayes classifier is optimal, but in practice we don't know $\mathbb{P})Y\ |\ X)$.

##### K-Nearest Neighbors

- The ***K-nearest neighbors*** classifier works by estimating $\mathbb{P})Y\ |\ X)$ as follows.

1. Given $K\geqslant 1$ and $x_0$,  find the set of points
$$ \mathcal{N}_0 = \{K\ \text{nearest points to}\ x_0\}\subseteq\mathbb{R}^p $$
2. For each class $j$ set 
$$ \mathbb{P}(Y=j\ |\ X) = \frac{1}{K}\sum_{x_i\in\mathcal{N}_0}I(y_i = j)$$
3. Predict

$$ \hat{y_0} = \underset{1\leqslant j \leqslant K}{\text{argmax}\,} \mathbb{P}(Y=j\ |\ X = x_0)$$

---
## Footnotes

[^0]: Here $X=(X_1,\dots, X_p)^T$ is a vector.

[^1]: Reading the rest of the chapter, one realized this is the situation for *supervised* learning, which is the vast majority of this book is concerned with. 

[^2]: This is usual definition of the mean squared-error of $\hat{Y}$ as an estimator of the (non-parametric) quantity $Y=f(X)$.

[^3]: We can in principle control the reducible error by improving the estimate $\hat{f}$, but we cannot control the irreducible error.

[^4]: For example, a simple but popular assumption is that f is linear in both the parameters and the features, that is: 
$$f(X) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$
This is linear regression.

[^5]: Here the random variable is $\hat{f}(x_0)$, so the average is taken over all data sets

[^6]: This is just the proportion of misclassified observations.

{% endkatexmm %}
