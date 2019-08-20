---
layout: page
title: 3. Linear Regression
---

{% katexmm %}
# Exercise 11: The t-statistic for null hypothesis in simple linear regression with no intercept

<div class="toc"><ul class="toc-item"><li><span><a href="#generate-the-data" data-toc-modified-id="Generate-the-data-1">Generate the data</a></span></li><li><span><a href="#a-simple-regression-of-y-onto-x-with-no-intercept" data-toc-modified-id="a.-Simple-regression-of-y-onto-x-with-no-intercept-2">a. Simple regression of <code>y</code> onto <code>x</code> with no intercept</a></span></li><li><span><a href="#b-simple-regression-of-x-onto-y-with-no-intercept" data-toc-modified-id="b.-Simple-regression-of-x-onto-y-with-no-intercept-3">b. Simple regression of <code>x</code> onto <code>y</code> with no intercept</a></span></li><li><span><a href="#c-the-relationship-between-the-models" data-toc-modified-id="c.-The-relationship-between-the-models-4">c. The relationship between the models</a></span></li><li><span><a href="#d-a-formula-for-the-t-statistic" data-toc-modified-id="d.-A-formula-for-the-t-statistic-5">d. A formula for the t-statistic</a></span></li><li><span><a href="#e-why-the-t-statistic-is-the-same-for-both-models" data-toc-modified-id="e.-Why-the-t-statistic-is-the-same-for-both-models-6">e. Why the t-statistic is the same for both models</a></span></li><li><span><a href="#f-repeat-for-simple-regression-with-an-intercept" data-toc-modified-id="f.-Repeat-for-simple-regression-with-an-intercept-7">f. Repeat for simple regression with an intercept</a></span></li></ul></div>

## Generate the data

First we generate paired data `(x,y)` according to

$$ Y = 2X + \epsilon$$


```python
import numpy as np

np.random.seed(1)
x = np.random.normal(size=100)
y = 2*x + np.random.normal(size=100)
```

## a. Simple regression of `y` onto `x` with no intercept


```python
import statsmodels.api as sm

model_1 = sm.OLS(y, x).fit()
model_1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.798</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.796</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   391.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 04 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>3.46e-36</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:12</td>     <th>  Log-Likelihood:    </th> <td> -135.67</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   273.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    99</td>      <th>  BIC:               </th> <td>   275.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>    2.1067</td> <td>    0.106</td> <td>   19.792</td> <td> 0.000</td> <td>    1.896</td> <td>    2.318</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.880</td> <th>  Durbin-Watson:     </th> <td>   2.106</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.644</td> <th>  Jarque-Bera (JB):  </th> <td>   0.554</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.172</td> <th>  Prob(JB):          </th> <td>   0.758</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.119</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The coefficient estimate is


```python
model_1.params
```




    array([2.10674169])



which is very close to the true value of $2$.

The standard error is


```python
model_1.bse
```




    array([0.10644517])



The t-statistic is


```python
model_1.tvalues
```




    array([19.79180199])



which has p-value


```python
model_1.pvalues
```




    array([3.45737574e-36])



This is an incredibly small p-value, so we have good grounds to reject the null hypothesis and accept the alternative hypothesis $H_a: \beta \neq 0$

## b. Simple regression of `x` onto `y` with no intercept


```python
import statsmodels.api as sm

model_2 = sm.OLS(x, y).fit()
model_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.798</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.796</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   391.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 04 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>3.46e-36</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:56:12</td>     <th>  Log-Likelihood:    </th> <td> -49.891</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>   101.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    99</td>      <th>  BIC:               </th> <td>   104.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>     <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th> <td>    0.3789</td> <td>    0.019</td> <td>   19.792</td> <td> 0.000</td> <td>    0.341</td> <td>    0.417</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.476</td> <th>  Durbin-Watson:     </th> <td>   2.166</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.788</td> <th>  Jarque-Bera (JB):  </th> <td>   0.631</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.115</td> <th>  Prob(JB):          </th> <td>   0.729</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.685</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The coefficient estimate is


```python
model_2.params
```




    array([0.37890442])



which is somewhat close to the true value of $0.5$.

The standard error is


```python
model_2.bse
```




    array([0.01914451])



The t-statistic is


```python
model_2.tvalues
```




    array([19.79180199])



which has p-value


```python
model_2.pvalues
```




    array([3.45737574e-36])



This is identical to the p-value for `model_1`, so we again have grounds to reject the null hypothesis and accept the alternative hypothesis $H_a: \beta \neq 0$

## c. The relationship between the models

In both cases there is a linear relationship between the two variables. Since the first model has the form

$$Y = 2X + \epsilon$$

the second model has the form

$$X = \frac{1}{2}Y - \frac{1}{2}\epsilon$$

In both cases, the regression detected the linear relationship with high confidence, and found good estimates for the coefficient.

It seems remarkable that the t-statistics were identical for both models, since the first is


```python
model_1.params[0]/model_1.bse[0]
```




    19.79180198709121



while the second


```python
model_2.params[0]/model_2.bse[0]
```




    19.79180198709121



## d. A formula for the t-statistic

Now it makes sense!

## e. Why the t-statistic is the same for both models

From d., we see that the t-statistic only depends on `x`, `y` and the sample size, and its symmetric with respect to  the substitution `x` $\mapsto$ `y`

## f. Repeat for simple regression with an intercept


```python
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({'x' :x, 'y': y})

model_1 = smf.ols('y ~ x', data=df).fit()
model_2 = smf.ols('x ~ y', data=df).fit()
```


```python
model_1.tvalues[1] == model_2.tvalues[1]
```




    True

{% endkatexmm %}
