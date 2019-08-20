---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Conceptual Exercises

<div class="toc"><ul class="toc-item"><li><span><a href="#exercise-1-minimize-the-weighted-sum-of-two-random-variables" data-toc-modified-id="Exercise-1:-Minimize-the-weighted-sum-of-two-random-variables-1">Exercise 1: Minimize the weighted sum of two random variables</a></span></li><li><span><a href="#exercise-2-derive-the-probability-an-observation-appears-in-a-bootstrap-sample" data-toc-modified-id="Exercise-2:-Derive-the-probability-an-observation-appears-in-a-bootstrap-sample-2">Exercise 2: Derive the probability an observation appears in a bootstrap sample</a></span><ul class="toc-item"><li><span><a href="#a" data-toc-modified-id="a.-2.1">a.</a></span></li><li><span><a href="#b" data-toc-modified-id="b.-2.2">b.</a></span></li><li><span><a href="#c" data-toc-modified-id="c.-2.3">c.</a></span></li><li><span><a href="#d" data-toc-modified-id="d.-2.4">d.</a></span></li><li><span><a href="#e" data-toc-modified-id="e.-2.5">e.</a></span></li><li><span><a href="#f" data-toc-modified-id="f.-2.6">f.</a></span></li><li><span><a href="#g" data-toc-modified-id="g.-2.7">g.</a></span></li><li><span><a href="#h" data-toc-modified-id="h.-2.8">h.</a></span></li><li><span><a href="#Exercise-3:-$k$-fold-Cross-Validation" data-toc-modified-id="Exercise-3:-$k$-fold-Cross-Validation-2.9">Exercise 3: $k$-fold Cross Validation</a></span></li><li><span><a href="#exercise-4-estimate-the-standard-deviation-of-a-predicted-reponse" data-toc-modified-id="Exercise-4:-Estimate-the-standard-deviation-of-a-predicted-reponse-2.10">Exercise 4: Estimate the standard deviation of a predicted reponse</a></span></li></ul></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-3">Footnotes</a></span></li></ul></div>

## Exercise 1: Minimize the weighted sum of two random variables

*Using basic statistical properties of the variance, as well as single- variable calculus, derive (5.6). In other words, prove that α given by (5.6) does indeed minimize Var$(\alpha X + (1 − \alpha)Y)$*

Using properties of variance we have

$$\text{Var}(\alpha X + (1 - \alpha) Y) = \alpha^2\sigma^2_X + (1 - \alpha)^2\sigma^2_Y + 2\alpha(1-\alpha)\sigma_{XY}$$

Taking the derivative with respect to $\alpha$, set to zero 

$$2\alpha\sigma^2_X - 2(1 - \alpha)\sigma^2_Y + 2(1-2\alpha)\sigma_{XY} = 0$$

solve for $\alpha$ to find

$$\alpha = \frac{\sigma^2_Y - \sigma_{XY}}{\sigma^2_X + \sigma^2_Y - 2\sigma_{XY}}$$



## Exercise 2: Derive the probability an observation appears in a bootstrap sample

### a.

*What is the probability that the first bootstrap observation is not the jth observation from the original sample? Justify your answer.*

$$
\begin{aligned}
P(\text{first bootstrap observation is not}\ j-\text{th observation}) &= \\
&= 1 - P(\text{first bootstrap observation is}\ j-\text{th observation})\\
&= 1 - \frac{1}{n}
\end{aligned}
$$

Since the boostrap observations are chosen uniformly at random

### b.

*What is the probability that the second bootstrap observation is not the jth observation from the original sample?*

The probability is still $1 - \frac{1}{n}$ since the bootstrap samples are drawn with replacement

### c.

Let 

$$A = \text{the}\ j-\text{th observation is not in the bootstrap sample}$$
$$A_k = \text{the}\ k-\text{th bootstrap observation is not the}\ j-\text{th observation}$$

Then since the bootstrap observations are drawn uniformly at random the $A_k$ are independent and $P(A_k) = 1- \frac{1}{n}$ hence

$$
\begin{aligned}
Pa. &= P\left(\cap_{k = 1}^n A_k\right)\\
&= \prod_{k = 1}^n P(A_k)\\
&= \prod_{k = 1}^n \left(1 - \frac{1}{n}\right)\\
&= \left(1 - \frac{1}{n}\right)^n
\end{aligned}
$$

### d.

We have

$$A^c = \text{the}\ j-\text{th observation is in the bootstrap sample}$$

So 

$$P(A^c) = 1 - Pa. = 1 - (1 - \frac{1}{n})^n$$

When $n=5$, $P(A^c) =$


```python
1 - (1 - 1/5)**5
```




    0.6723199999999999



### e.

When $n=100, Pa.$ is


```python
1 - (1 - 1/100)**100
```




    0.6339676587267709



### f.

When $n=10^4, Pa.$ is


```python
1 - (1 - 1/10e4)**10e4
```




    0.6321223982317534



### g.


```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('seaborn-white')
```


```python
x = np.arange(1, 100000, 1)
y = 1 - (1 - 1/x)**x

plt.plot(x, y, color='r')
```




    [<matplotlib.lines.Line2D at 0x11d180860>]




![png]({{site.baseurl}}/assets/images/ch05_conceptual_exercises_22_1.png)


The probability rapidly drops to around $\frac{2}{3}$


```python
x = np.arange(1, 10, 1)
y = 1 - (1 - 1/x)**x

plt.plot(x, y, color='r')
```




    [<matplotlib.lines.Line2D at 0x121806ac8>]




![png]({{site.baseurl}}/assets/images/ch05_conceptual_exercises_24_1.png)


then slowly asymptotically approaches [the limit](https://en.wikipedia.org/wiki/Exponential_function#Overview)

$$ \underset{n \rightarrow \infty}{\lim} 1 - (1 - \frac{1}{n})^n = 1 - e^{-1} \approx 0.6321$$

### h.


```python
data = np.arange(1, 101, 1)

sum([4 in np.random.choice(data, size=100, replace=True) for i in range(10000)])/10000
```




    0.6308



Very close to the expected value of 


```python
1 - (1 - 1/100)**100
```




    0.6339676587267709



### Exercise 3: $k$-fold Cross Validation

See section 5.1.3 in the [notes](../../Notes.ipynb)

### Exercise 4: Estimate the standard deviation of a predicted reponse

Suppose given $(X, Y)$ we predict $\hat{Y}$. This is an estimator [^0]. To estimate its standard error using data $(x_1, y_1), \dots, (x_n, y_n)$ use the "plug-in" estimator [^1].

$$\hat{se}(\hat{Y}) = \sqrt{\frac{1}{n} \sum_{i = 1}^ n \left(\hat{y}_i - \overline{\hat{y}}\right)^2}$$

where $\hat{y}_i$ is the predicted value for $x_i$ and $\overline{\hat{y}}$ is the mean predicted value.

In other words, use the sample standard deviation of the predicted values.

## Footnotes

[^1]:  An estimator is a statistic (a function of the data) used to estimate a population quantity -- it is a random variable corresponding to the statistical learning method we use and dependent on the observed data.

[^2]: The plug-in estimator is a statistical functional -- it is used to estimate a population functional (i.e. some function of the population distribution). It is so called because it estimates the population functional by plugging in the empirical distribution [Wasserman Ch 7](https://www.stat.cmu.edu/~larry/all-of-statistics/). In this case, the population functional is the "true" standard error $se(\hat{Y})$.

<p>
</p>

<div id="foot0"> 0.
<a href="#ref0">↩</a>
</div>

<p>
</p>

<div id="foot1"> 1. 
<a href="#ref1">↩</a>
</div>

{% endkatexmm %}