---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#support-vector-machines" data-toc-modified-id="Support-Vector-Machines-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Support Vector Machines</a></span><ul class="toc-item"><li><span><a href="#maximal-margin-classifier" data-toc-modified-id="Maximal-Margin-Classifier-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Maximal Margin Classifier</a></span><ul class="toc-item"><li><span><a href="#what-is-a-hyperplane?" data-toc-modified-id="What-Is-a-Hyperplane?-9.1.1"><span class="toc-item-num">9.1.1&nbsp;&nbsp;</span>What Is a Hyperplane?</a></span></li><li><span><a href="#classification-using-a-separating-hyperplane" data-toc-modified-id="Classification-Using-a-Separating-Hyperplane-9.1.2"><span class="toc-item-num">9.1.2&nbsp;&nbsp;</span>Classification Using a Separating Hyperplane</a></span></li><li><span><a href="#the-maximal-margin-classifier" data-toc-modified-id="The-Maximal-Margin-Classifier-9.1.3"><span class="toc-item-num">9.1.3&nbsp;&nbsp;</span>The Maximal Margin Classifier</a></span></li><li><span><a href="#construction-of-the-maximal-margin-classifier" data-toc-modified-id="Construction-of-the-Maximal-Margin-Classifier-9.1.4"><span class="toc-item-num">9.1.4&nbsp;&nbsp;</span>Construction of the Maximal Margin Classifier</a></span></li><li><span><a href="#the-non-separable-case" data-toc-modified-id="The-Non-separable-Case-9.1.5"><span class="toc-item-num">9.1.5&nbsp;&nbsp;</span>The Non-separable Case</a></span></li></ul></li><li><span><a href="#support-vector-classifiers" data-toc-modified-id="Support-Vector-Classifiers-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Support Vector Classifiers</a></span><ul class="toc-item"><li><span><a href="#overview-of-the-support-vector-classifier" data-toc-modified-id="Overview-of-the-Support-Vector-Classifier-9.2.1"><span class="toc-item-num">9.2.1&nbsp;&nbsp;</span>Overview of the Support Vector Classifier</a></span></li><li><span><a href="#details-of-the-support-vector-classifier" data-toc-modified-id="Details-of-the-Support-Vector-Classifier-9.2.2"><span class="toc-item-num">9.2.2&nbsp;&nbsp;</span>Details of the Support Vector Classifier</a></span></li></ul></li><li><span><a href="#support-vector-machines" data-toc-modified-id="Support-Vector-Machines-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Support Vector Machines</a></span><ul class="toc-item"><li><span><a href="#classification-with-non-linear-decision-boundaries" data-toc-modified-id="Classification-with-Non-Linear-Decision-Boundaries-9.3.1"><span class="toc-item-num">9.3.1&nbsp;&nbsp;</span>Classification with Non-Linear Decision Boundaries</a></span></li><li><span><a href="#the-support-vector-machine" data-toc-modified-id="The-Support-Vector-Machine-9.3.2"><span class="toc-item-num">9.3.2&nbsp;&nbsp;</span>The Support Vector Machine</a></span></li></ul></li><li><span><a href="#svms-with-more-than-two-classes" data-toc-modified-id="SVMs-with-More-than-Two-Classes-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>SVMs with More than Two Classes</a></span><ul class="toc-item"><li><span><a href="#one-versus-one-classification" data-toc-modified-id="One-Versus-One-Classification-9.4.1"><span class="toc-item-num">9.4.1&nbsp;&nbsp;</span>One-Versus-One Classification</a></span></li><li><span><a href="#one-versus-all-classification" data-toc-modified-id="One-Versus-All-Classification-9.4.2"><span class="toc-item-num">9.4.2&nbsp;&nbsp;</span>One-Versus-All Classification</a></span></li></ul></li><li><span><a href="#relationship-to-logistic-tegression" data-toc-modified-id="Relationship-to-Logistic-Regression-9.5"><span class="toc-item-num">9.5&nbsp;&nbsp;</span>Relationship to Logistic Regression</a></span></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-9.6"><span class="toc-item-num">9.6&nbsp;&nbsp;</span>Footnotes</a></span></li></ul></li></ul></div>

## Maximal Margin Classifier

### What Is a Hyperplane?

- A ***hyperplane*** in $\mathbb{R}^p$ is an affine subspace of dimension $p-1$. Every hyperplane is the set of solutions $X$ to $\beta^\top X = 0$ for some $\beta\in\mathbb{R}^p$. 
- A hyperplane $\beta^\top X = 0$ partitions $\mathbb{R}^p$ into two halfspaces:

$$H_+ = \{X\in\mathbb{R}^p\ |\ \beta^\top X > 0\}$$
$$H_- = \{X\in\mathbb{R}^p\ |\ \beta^\top X > 0\}$$
  
  corresponding to either side of the plane, or equivalently, 

$$H_+ = \{X\in\mathbb{R}^p\ |\ \text{sgn}(\beta^\top X) =  1\}$$
$$H_- = \{X\in\mathbb{R}^p\ |\ \text{sgn}(\beta^\top X) =  -1\}$$

### Classification Using a Separating Hyperplane

- Given data $(x_i, y_i)$, $i = 1,\dots n$ with response classes $y_i \in \{ \pm 1\}$, a hyperplane $\beta^\top X = 0$ is ***separating*** if 

$$\text{sgn}(\beta^\top x_i) = y_i$$ 

for all $i$.
- Given a separating hyperplane, we may predict 

$$\hat{y}_i = \text{sgn}(\beta^\top x_i)$$

### The Maximal Margin Classifier

- Separating hyperplanes are not unique (if one exists then uncountably many exist). A natural choice is the ***maximal margin hyperplane*** (or ***optimal separating hyperplane***)

- The ***margin*** is the minimal perpendicular distance to the hyperplane over the sample points
$$ M = \underset{i}{\min}\{\ ||x_i - P x_i||\ \}$$
  where $P$ is the projection matrix onto the hyperplane.

- The points $(x_i, y_i)$ "on the margin" (where $||x_i - P x_i|| = M$) are called ***support vectors***

### Construction of the Maximal Margin Classifier

The maximal margin classifier is the solution to the optimization problem:

$$ \begin{aligned}
\underset{\boldsymbol{\beta}}{\text{argmax}}&\ M\\
\text{subject to}&\ ||\,\boldsymbol{\beta}\,|| = 1\\
& \mathbf{y}^\top(X\boldsymbol{\beta}) \geqslant \mathbf{M}\\
\end{aligned} $$

  where $\mathbf{M} = (M, \dots, M) \in \mathbb{R}^n$ [^1]


### The Non-separable Case

- The maximal margin classifier is a natural classifier, but a separating hyperplane is not guaranteed to exist

- If a separating hyperplane doesn't exist, we can choose an "almost" separating hyperplane by using a "soft" margin.

## Support Vector Classifiers

### Overview of the Support Vector Classifier

- Separating hyperplanes don't always exist, and even if they do, they may be undesirable. 

- The distance to the hyperplane can be thought of as a measure of confidence in the classification. For very small margins, the separating hyperplane is very sensitive to individual observations -- we have low confidence in the classification of nearby observations.

- In these situations, we may prefer a hyperplane that doesn't perfectly separate in the interest of:
    - Greater robustness to individual observations
    - Better classification of most of the training observations

- This is achieved by the ***support vector classifier*** or ***soft margin classifier*** [^2]


### Details of the Support Vector Classifier

- The support vector classifier is the solution to the optimization problem:

$$ \begin{aligned}
\underset{\boldsymbol{\beta}}{\text{argmax}}&\ M\\
\text{subject to}&\ ||\,\boldsymbol{\beta}\,|| = 1\\
& y_i(\boldsymbol{\beta}^\top x_i) \geqslant M(1-\epsilon_i)\\
& \epsilon_i \geqslant 0\\
& \sum_i \epsilon_i \leqslant C
\end{aligned} $$


  where $C \geqslant 0$ is a tuning parameter, $M$ is the margin, and the $\epsilon_i$ are slack variables [3].

- Observations on the margin or on the wrong side of the margin are called ***support vectors***

## Support Vector Machines

### Classification with Non-Linear Decision Boundaries

- The support vector classifier is a natural choice for two response classes when the class boundary is linear, but may perform poorly when the boundary is non-linear.

- Non-linear transformations of the features will lead to a non-linear class boundary, but enlarging the feature space too much can lead to intractable computations.

- The support vector machine enlarges the feature space in a way which is computationally efficient.

### The Support Vector Machine

- It can be shown that:
  - the linear support vector classifier is a model of the form $$f(x) = \beta_0 + \sum_{i = 1}^n \alpha_i \langle x, x_i\rangle $$
  - the parameter estimates $\hat{\alpha}_i, \hat{\beta}_0$ can be computed from the $\binom{n}{2}$ inner products $\langle x, x_i \rangle$

- The support vector machine is a model of the form 
$$f(x) = \beta_0 + \sum_{i = 1}^n \alpha_i K(x, x_i) $$
where $K$ is a ***kernel function*** [^4]

- Popular kernels [^5] are
    - The ***polynomial kernel***
    $$K(x_i, x_i') = (1 + x_i^\top x_i')^d$$
    - The ***radial kernel***
    $$K(x_i, x_i') = \exp(-\gamma\,||x_i - x_i'||^2)$$
    

## SVMs with More than Two Classes

### One-Versus-One Classification

This approach works as follows:
  1. Fit $\binom{K}{2}$ SVMs, one for each pair of classes $k,k'$ encoded as $\pm 1$, respectively.
  2. For each observation $x$, classify using each of the predictors in 1, and let $N_k$ be the number of times $x$ was assigned to class $k$.
  3. Predict 
  $$ \hat{f}(x) = \underset{k}{\text{argmax}}\, N_k$$

### One-Versus-All Classification

This approach works as follows:
  1. Fit $K$ SVMs, comparing each class $k$ to other $K-1$ classes, encoded as $\pm 1$, respectively. Let $\beta_k (\beta_{0k}, \dots, \beta_{pk})$ be resulting parameters.
  2. Predict 
  $$\hat{f}(x) = \underset{k}{\text{argmax}}\, \beta_k^\top x$$

## Relationship to Logistic Regression

- The optimization problem leading to the support vector classifier can be rewritten as
$$\underset{\beta}{\text{argmin}}\left(\sum_{i = 1}^n \max\{0, 1 - y_i(\beta^\top x_i)\} + \lambda\,||\beta||^2\right)$$
 where $\lambda \geqslant 0 $ is a tuning parameter [^6].
- The hinge loss [^7] is very similar to the logistic regression loss, so both methods tend to give similar results. However, SVMs tend to perform better when the classes are well separated, while logistic regression tends to perform better when they are not.

___
## Footnotes

[^1]: The constraint $|| \boldsymbol{\beta} || = 1$ ensures that the perpendicular distance $||x_i - P x_i||$ is given by $y_i(\beta^\top x_i)$.

[^2]: Sometimes the maximal margin and support vector classifiers are called "hard margin" and "soft margin" support vector classifiers, respectively.

[^3]: For each $i$, if $\epsilon_i = 0$ the $i$-th observation is on the correct side of the margin. If $\epsilon_i > 0$ then it is on the wrong side of the margin, and if $\epsilon_i > 1$ then it is on the wrong side of the hyperplane. The parameter $C$ is a "margin violation tolerance" -- it bounds the $\epsilon_i$ and thus the number/size of margin violations. Greater $C$ implies greater tolerance. The case $C = 0$ is the maximal margin hyperplane.

[^4]: In this context a kernel function is [a positive-definite kernel]('https://en.wikipedia.org/wiki/Positive-definite_kernel'). Among other things, it is a generalization of an inner product (every inner product $\langle x, y \rangle$ is a kernel function), and is one way of quantifying similarity between points. In the context of statistical and machine learning, a [kernel method]('https://en.wikipedia.org/wiki/Kernel_method') is one which makes use of the "kernel trick". The kernel function $K(x_i, x_i')$ encodes the similarity of the observations $x_i, x_i'$ in a transformed feature space, but it is more computationally efficient to compute the $\binom{n}{k}$ kernels themselves than to transform the data. The kernel fits a support vector classifier (hence a linear classification boundary) in the transformed feature space, which corresponds to a non-linear boundary in the original feature space. 

[^5]: The <a href='https://en.wikipedia.org/wiki/Polynomial_kernel'>polynomial kernel </a> is effectively the inner product on the space of $d$-degree polynomials in the features $X_j$. The <a href='https://en.wikipedia.org/wiki/Radial_basis_function_kernel'>radial kernel</a> is a similarity measure in an infinite dimensional feature space.

[^6]: This is another instance of a general form of a "regularized loss" or "loss + penalty" $$\underset{\beta}{\text{argmin}}L(\mathbf{X}, \mathbf{y}, \beta) + \lambda P(\beta)$$ where the loss function $L(\mathbf{X}, \mathbf{y}, \beta)$ quantifies how well the parameter model with parameter $\beta$ fits the data $(\mathbf{X}, \mathbf{y})$, and $P(\beta)$ is a penalty function controlled by $\lambda$.

[^7]: In this case $$L(\mathbf{X}, \mathbf{y}, \beta) = \sum_{i = 1}^n \{0, 1 - y_i(\beta^\top x_i\}$$ is called the ***hinge loss***.

{% endkatexmm %}