---
layout: page
title: 3. Linear Regression
---

{% katexmm %}
# Exercise 12: Simple regression without an intercept

<div class="toc"><ul class="toc-item"><li><span><a href="#a-when-is-hat-beta-the-same-when-we-switch-predictor-and-response" data-toc-modified-id="a.-When-is-hat-beta-the-same-when-we-switch-predictor-and-response?-1">a. When is $\hat{\beta}$ the same when we switch predictor and response?</a></span></li><li><span><a href="#b-generate-a-counterexample" data-toc-modified-id="b.-Generate-a-counterexample-2">b. Generate a counterexample</a></span></li><li><span><a href="#c-generate-an-example" data-toc-modified-id="c.-Generate-an-example-3">c. Generate an example</a></span></li></ul></div>

## a. When is $\hat{\beta}$ the same when we switch predictor and response?

From equation (3.38), $\hat{\beta}$ is the same in both cases when

$$\frac{\sum_i x_iy_i}{\sum_i x_i^2} = \frac{\sum_i y_ix_i}{\sum_i y_i^2}$$

The numerators are always equal, so $\hat{\beta}$ is the same iff

$$\sum_i x_i^2 = \sum_i y_i^2$$

## b. Generate a counterexample

This is fairly easy to do, since with overwhelming probability

$$\sum_i x_i^2 \neq \sum_i y_i^2$$


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

x, y = np.random.normal(size=100), np.random.normal(size=100)

model_1 = sm.OLS(y, x).fit()
model_2 = sm.OLS(x, y).fit()
```


```python
model_1.params[0] == model_2.params[0]
```




    False



## c. Generate an example

This is fairly easy to do cheesily by letting `x` = `y`


```python
x = np.random.normal(size=100)
y = x

model_1 = sm.OLS(y, x).fit()
model_2 = sm.OLS(x, y).fit()
```


```python
model_1.params[0] == model_2.params[0]
```




    True

{% endkatexmm %}
