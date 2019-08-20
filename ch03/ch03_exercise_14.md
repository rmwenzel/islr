---
layout: page
title: 3. Linear Regression
---

{% katexmm %}

# Exercise 14: The collinearity problem

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generating-the-data" data-toc-modified-id="a.-Generating-the-data-1">a. Generating the data</a></span></li><li><span><a href="#b-correlation-among-predictors" data-toc-modified-id="b.-Correlation-among-predictors-2">b. Correlation among predictors</a></span></li><li><span><a href="#c-fitting-a-full-ols-linear-regression-model" data-toc-modified-id="c.-Fitting-a-full-OLS-linear-regression-model-3">c. Fitting a full OLS linear regression model</a></span></li><li><span><a href="#d-fitting-an-ols-linear-regression-model-on-the-first-predictor" data-toc-modified-id="d.-Fitting-an-OLS-linear-regression-model-on-the-first-predictor-4">d. Fitting an OLS linear regression model on the first predictor </a></span></li><li><span><a href="#e-fitting-an-ols-linear-regression-model-on-the-second-predictor" data-toc-modified-id="e.-Fitting-an-OLS-linear-regression-model-on-the-second-predicto">e. Fitting an OLS linear regression model on the second predictor </a></span></li><li><span><a href="#f-do-these-results-contradict-each-other" data-toc-modified-id="f.-Do-these-results-contradict-each-other?-6">f. Do these results contradict each other?</a></span></li><li><span><a href="#g-adding-a-mismeasured-observation" data-toc-modified-id="g.-Adding-a-mismeasured-observation-7">g. Adding a mismeasured observation</a></span></li></ul></div>

## a. Generating the data

In this problem the data are $(\mathbf{X}, Y)$ where $\mathbf{X}=(X_1, X_2)$ and

- $X_1 \sim \text{Uniform}(0,1)$
- $X_2 = \frac{1}{2}X_2 + \frac{1}{100}Z$ where $Z\sim \text{Normal}(0,1)$
- $Y = 2 + 2X_1 + \frac{3}{10} X_2 + \epsilon$

The regression coefficients are

$$(\beta_0, \beta_1, \beta_2) = (2, 2, \frac{3}{10})$$


```python
import numpy as np
import pandas as pd

# set seed state for reproducibility
np.random.seed(0)

# generate data
x1 = np.random.uniform(size=100)
x2 = 0.5*x1 + np.random.normal(size=100)/100
y = 2 + 2*x1 + 0.3*x2

# collect data in dataframe
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y':y})
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.548814</td>
      <td>0.262755</td>
      <td>3.176454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.715189</td>
      <td>0.366603</td>
      <td>3.540360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.602763</td>
      <td>0.306038</td>
      <td>3.297338</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.544883</td>
      <td>0.257079</td>
      <td>3.166890</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.423655</td>
      <td>0.226710</td>
      <td>2.915323</td>
    </tr>
  </tbody>
</table>
</div>



## b. Correlation among predictors

The correlation matrix is


```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x1</th>
      <td>1.000000</td>
      <td>0.997614</td>
      <td>0.999988</td>
    </tr>
    <tr>
      <th>x2</th>
      <td>0.997614</td>
      <td>1.000000</td>
      <td>0.997936</td>
    </tr>
    <tr>
      <th>y</th>
      <td>0.999988</td>
      <td>0.997936</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



so the correlation among the predictors is approximately


```python
round(data.corr().x1.x2, 3)
```




    0.998



The scatterplot is


```python
import seaborn as sns

sns.scatterplot(data.x1, data.x2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17257358>



## c. Fitting a full OLS linear regression model


```python
import statsmodels.formula.api as smf

full_model = smf.ols('y ~ x1 + x2', data=data).fit()
full_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.237e+30</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:03:46</td>     <th>  Log-Likelihood:    </th> <td>  3206.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>  -6406.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    97</td>      <th>  BIC:               </th> <td>  -6398.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.0000</td> <td> 5.67e-16</td> <td> 3.53e+15</td> <td> 0.000</td> <td>    2.000</td> <td>    2.000</td>
</tr>
<tr>
  <th>x1</th>        <td>    2.0000</td> <td> 1.47e-14</td> <td> 1.36e+14</td> <td> 0.000</td> <td>    2.000</td> <td>    2.000</td>
</tr>
<tr>
  <th>x2</th>        <td>    0.3000</td> <td> 2.94e-14</td> <td> 1.02e+13</td> <td> 0.000</td> <td>    0.300</td> <td>    0.300</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 4.521</td> <th>  Durbin-Watson:     </th> <td>   0.105</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.104</td> <th>  Jarque-Bera (JB):  </th> <td>   3.208</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.288</td> <th>  Prob(JB):          </th> <td>   0.201</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.338</td> <th>  Cond. No.          </th> <td>    128.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



We find the estimators $(\hat{\beta_0}, \hat{\beta_1}, \hat{\beta_2})$ are


```python
full_model.params
```




    Intercept    2.0
    x1           2.0
    x2           0.3
    dtype: float64



which is identical to $(\beta_0, \beta_1, \beta_2)$.

The p-values for the estimators are


```python
full_model.pvalues
```




    Intercept    0.0
    x1           0.0
    x2           0.0
    dtype: float64



So we can definitely reject the null hypotheses 

$$H_{0}^i: \beta_i = 0\qquad{0 \leqslant i \leqslant 2}$$

## d. Fitting an OLS linear regression model on the first predictor


```python
x1_model = smf.ols('y ~ x1', data=data).fit()
x1_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>4.215e+06</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>7.26e-229</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:03:46</td>     <th>  Log-Likelihood:    </th> <td>  439.40</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>  -874.8</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    98</td>      <th>  BIC:               </th> <td>  -869.6</td> 
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
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.0007</td> <td>    0.001</td> <td> 3450.157</td> <td> 0.000</td> <td>    2.000</td> <td>    2.002</td>
</tr>
<tr>
  <th>x1</th>        <td>    2.1498</td> <td>    0.001</td> <td> 2052.992</td> <td> 0.000</td> <td>    2.148</td> <td>    2.152</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.746</td> <th>  Durbin-Watson:     </th> <td>   2.083</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.003</td> <th>  Jarque-Bera (JB):  </th> <td>   4.097</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.138</td> <th>  Prob(JB):          </th> <td>   0.129</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.047</td> <th>  Cond. No.          </th> <td>    4.30</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



In this case, $\hat{\beta}_0 = \beta_0 = 2$ but $\hat{\beta}_1 = 2.1498$ is a little off. 

However, the p-value for $\hat{\beta_1}$ are still zero, so we again reject the null hypothesis $H_0: \beta_1 = 0$.

## e. Fitting an OLS linear regression model on the second predictor


```python
x2_model = smf.ols('y ~ x2', data=data).fit()
x2_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.996</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.996</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>2.366e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.14e-118</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:03:46</td>     <th>  Log-Likelihood:    </th> <td>  180.49</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   100</td>      <th>  AIC:               </th> <td>  -357.0</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    98</td>      <th>  BIC:               </th> <td>  -351.8</td> 
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
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    1.9957</td> <td>    0.008</td> <td>  257.091</td> <td> 0.000</td> <td>    1.980</td> <td>    2.011</td>
</tr>
<tr>
  <th>x2</th>        <td>    4.2860</td> <td>    0.028</td> <td>  153.832</td> <td> 0.000</td> <td>    4.231</td> <td>    4.341</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12.365</td> <th>  Durbin-Watson:     </th> <td>   2.101</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.002</td> <th>  Jarque-Bera (JB):  </th> <td>   4.200</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.139</td> <th>  Prob(JB):          </th> <td>   0.122</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.035</td> <th>  Cond. No.          </th> <td>    7.33</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



In this case, $\hat{\beta}_0 = \beta_0 = 2$ but $\hat{\beta}_2 = 4.2860$ is way off. 

However, the p-value for $\hat{\beta_1}$ are still zero, so we again reject the null hypothesis $H_0: \beta_1 = 0$.

## f. Do these results contradict each other?

Given that the authors want us to consider the correlation between the predictors, I think the intent of this question is to encourage us to wonder why, if $X_1, X_2$ are so strongly correlated, did the $X_1$ model do such a great job of estimating $\beta_1$ while the $X_2$ model did a poor job of estimating $\beta_2$?

It's not clear how to answer this question.

## g. Adding a mismeasured observation

First we add the new, mismeasured observation


```python
new_row = pd.DataFrame({'x1': [0.1], 'x2': [0.8], 'y':[6]})
data = pd.concat([data, new_row]).reset_index()
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>index</th>
      <th>x1</th>
      <th>x2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>96</td>
      <td>0.586513</td>
      <td>0.285222</td>
      <td>3.258593</td>
    </tr>
    <tr>
      <th>97</th>
      <td>97</td>
      <td>0.020108</td>
      <td>0.003158</td>
      <td>2.041163</td>
    </tr>
    <tr>
      <th>98</th>
      <td>98</td>
      <td>0.828940</td>
      <td>0.409915</td>
      <td>3.780854</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>0.004695</td>
      <td>0.002523</td>
      <td>2.010148</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>0.100000</td>
      <td>0.800000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now we refit the models from c.-e..


```python
full_model = smf.ols('y ~ x1 + x2', data=data).fit()
x1_model = smf.ols('y ~ x1', data=data).fit()
x2_model = smf.ols('y ~ x2', data=data).fit()
```


```python
full_model.params
```




    Intercept    1.991550
    x1          -0.337428
    x2           4.975187
    dtype: float64




```python
x1_model.params
```




    Intercept    2.115685
    x1           1.984495
    dtype: float64




```python
x2_model.params
```




    Intercept    1.968755
    x2           4.419615
    dtype: float64



Now the paramter estimates of both the full model and $X_2$-only model are off, while the $X_1$-only model is pretty close. 


```python
full_model.pvalues
```




    Intercept    1.926622e-133
    x1            1.614723e-16
    x2            4.878009e-90
    dtype: float64




```python
x1_model.pvalues
```




    Intercept    3.830438e-51
    x1           5.772966e-28
    dtype: float64




```python
x2_model.pvalues
```




    Intercept    9.791075e-121
    x2           3.652059e-102
    dtype: float64



Both models are still really confident in rejecting the null hypothesis. 

We'll do a scatter plot to see if the new observation is an outlier.


```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
```


```python
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(data['x1'], data['x2'], data['y'])
```




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x1a24db2c50>




![png]({{site.baseurl}}/assets/images/ch03_exercise_14_41_1.png)


That new observation sure looks like an outlier. If we look at fitted-vs-residuals plots


```python
# fitted-vs-residuals for the full model
sns.regplot(full_model.fittedvalues, full_model.resid/full_model.resid.std(),
            lowess=True,
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('fitted values')
plt.ylabel('studentized resid')
```




    Text(0,0.5,'studentized resid')




![png]({{site.baseurl}}/assets/images/ch03_exercise_14_43_1.png)



```python
# fitted-vs-residuals for the X1 model
sns.regplot(x1_model.fittedvalues, x1_model.resid/x1_model.resid.std(),
            lowess=True,
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('fitted values')
plt.ylabel('studentized resid')
```




    Text(0,0.5,'studentized resid')




![png]({{site.baseurl}}/assets/images/ch03_exercise_14_44_1.png)



```python
# fitted-vs-residuals for the X2 model
sns.regplot(x2_model.fittedvalues, x2_model.resid/x2_model.resid.std(),
            lowess=True,
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('fitted values')
plt.ylabel('studentized resid')
```




    Text(0,0.5,'studentized resid')




![png]({{site.baseurl}}/assets/images/ch03_exercise_14_45_1.png)


All three plots show a clear outlier. The $X_1$-only and $X_2$-only models also show high standardized residual values of $\approx 6.5$ and $\approx 10$ respectively, for the outlier.

Now we look at some leverage plots


```python
# scatterplot of leverage vs studentized residuals
axes = sns.regplot(full_model.get_influence().hat_matrix_diag, full_model.resid/full_model.resid.std(), 
            lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('leverage')
plt.ylabel('studentized resid')

# plot Cook's distance contours for D = 0.5, D = 1
x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 50)
plt.plot(x, np.sqrt(0.5*(1 - x)/x), color='red', linestyle='dashed')
plt.plot(x, np.sqrt((1 - x)/x), color='red', linestyle='dashed')
```

    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:11: RuntimeWarning: invalid value encountered in sqrt
      # This is added back by InteractiveShellApp.init_path()
    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:12: RuntimeWarning: invalid value encountered in sqrt
      if sys.path[0] == '':





    [<matplotlib.lines.Line2D at 0x1a264b3748>]




![png]({{site.baseurl}}/assets/images/ch03_exercise_14_47_2.png)


Since the mismeasured observation is has a Cook's distance greater than 1, we'll call it a high leverage point.

{% endkatexmm %}
