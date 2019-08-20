---
layout: page
title: 3. Linear Regression
---

{% katexmm %}

# Exercise 15: Regression models for the `Boston` Dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#Preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-regression-models-for-each-predictor" data-toc-modified-id="a.-Regression-models-for-each-predictor-2">a. Regression models for each predictor</a></span></li><li><span><a href="#b-full-regression-model" data-toc-modified-id="b.-Full-regression-model-3">b. Full regression model</a></span></li></ul></div>

## Preparing the data


```python
import pandas as pd

boston = pd.read_csv('../../datasets/Boston.csv', index_col=0)
boston.head()
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
boston.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 506 entries, 1 to 506
    Data columns (total 14 columns):
    crim       506 non-null float64
    zn         506 non-null float64
    indus      506 non-null float64
    chas       506 non-null int64
    nox        506 non-null float64
    rm         506 non-null float64
    age        506 non-null float64
    dis        506 non-null float64
    rad        506 non-null int64
    tax        506 non-null int64
    ptratio    506 non-null float64
    black      506 non-null float64
    lstat      506 non-null float64
    medv       506 non-null float64
    dtypes: float64(11), int64(3)
    memory usage: 59.3 KB


## a. Regression models for each predictor

We want to predict the per capita crime rate `crim`.


```python
import numpy as np
import statsmodels.formula.api as smf

# get predictor names
predictors = np.delete(boston.columns.values, np.where(boston.columns.values==['crim']))

# dictionary for models
models = {}

# fit single predictor models
for predictor in predictors:
    models[predictor] = smf.ols('crim ~ ' + predictor, data=boston).fit()
```


```python
models_keys_iter = iter(models.values())

next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.040</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.038</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   21.10</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>5.51e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:41:34</td>     <th>  Log-Likelihood:    </th> <td> -1796.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3596.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3604.</td>
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
  <th>Intercept</th> <td>    4.4537</td> <td>    0.417</td> <td>   10.675</td> <td> 0.000</td> <td>    3.634</td> <td>    5.273</td>
</tr>
<tr>
  <th>zn</th>        <td>   -0.0739</td> <td>    0.016</td> <td>   -4.594</td> <td> 0.000</td> <td>   -0.106</td> <td>   -0.042</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>567.443</td> <th>  Durbin-Watson:     </th> <td>   0.857</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>32753.004</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.257</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>40.986</td>  <th>  Cond. No.          </th> <td>    28.8</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.165</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.164</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   99.82</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.45e-21</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:41:47</td>     <th>  Log-Likelihood:    </th> <td> -1760.6</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3525.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3534.</td>
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
  <th>Intercept</th> <td>   -2.0637</td> <td>    0.667</td> <td>   -3.093</td> <td> 0.002</td> <td>   -3.375</td> <td>   -0.753</td>
</tr>
<tr>
  <th>indus</th>     <td>    0.5098</td> <td>    0.051</td> <td>    9.991</td> <td> 0.000</td> <td>    0.410</td> <td>    0.610</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>585.118</td> <th>  Durbin-Watson:     </th> <td>   0.986</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>41418.938</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.449</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>45.962</td>  <th>  Cond. No.          </th> <td>    25.1</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.003</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.001</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.579</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th>  <td> 0.209</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>16:41:51</td>     <th>  Log-Likelihood:    </th> <td> -1805.6</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3615.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3624.</td>
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
  <th>Intercept</th> <td>    3.7444</td> <td>    0.396</td> <td>    9.453</td> <td> 0.000</td> <td>    2.966</td> <td>    4.523</td>
</tr>
<tr>
  <th>chas</th>      <td>   -1.8928</td> <td>    1.506</td> <td>   -1.257</td> <td> 0.209</td> <td>   -4.852</td> <td>    1.066</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>561.663</td> <th>  Durbin-Watson:     </th> <td>   0.817</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>30645.429</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.191</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>39.685</td>  <th>  Cond. No.          </th> <td>    3.96</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.177</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.176</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   108.6</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>3.75e-23</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:41:58</td>     <th>  Log-Likelihood:    </th> <td> -1757.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3518.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3526.</td>
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
  <th>Intercept</th> <td>  -13.7199</td> <td>    1.699</td> <td>   -8.073</td> <td> 0.000</td> <td>  -17.059</td> <td>  -10.381</td>
</tr>
<tr>
  <th>nox</th>       <td>   31.2485</td> <td>    2.999</td> <td>   10.419</td> <td> 0.000</td> <td>   25.356</td> <td>   37.141</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>591.712</td> <th>  Durbin-Watson:     </th> <td>   0.992</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>43138.106</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.546</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>46.852</td>  <th>  Cond. No.          </th> <td>    11.3</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.048</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.046</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   25.45</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>6.35e-07</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:03</td>     <th>  Log-Likelihood:    </th> <td> -1793.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3592.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3600.</td>
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
  <th>Intercept</th> <td>   20.4818</td> <td>    3.364</td> <td>    6.088</td> <td> 0.000</td> <td>   13.872</td> <td>   27.092</td>
</tr>
<tr>
  <th>rm</th>        <td>   -2.6841</td> <td>    0.532</td> <td>   -5.045</td> <td> 0.000</td> <td>   -3.729</td> <td>   -1.639</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>575.717</td> <th>  Durbin-Watson:     </th> <td>   0.879</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>36658.093</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.345</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>43.305</td>  <th>  Cond. No.          </th> <td>    58.4</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.124</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.123</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   71.62</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.85e-16</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:07</td>     <th>  Log-Likelihood:    </th> <td> -1772.7</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3549.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3558.</td>
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
  <th>Intercept</th> <td>   -3.7779</td> <td>    0.944</td> <td>   -4.002</td> <td> 0.000</td> <td>   -5.633</td> <td>   -1.923</td>
</tr>
<tr>
  <th>age</th>       <td>    0.1078</td> <td>    0.013</td> <td>    8.463</td> <td> 0.000</td> <td>    0.083</td> <td>    0.133</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>574.509</td> <th>  Durbin-Watson:     </th> <td>   0.956</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>36741.903</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.322</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>43.366</td>  <th>  Cond. No.          </th> <td>    195.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.144</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.142</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   84.89</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>8.52e-19</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:10</td>     <th>  Log-Likelihood:    </th> <td> -1767.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3538.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3546.</td>
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
  <th>Intercept</th> <td>    9.4993</td> <td>    0.730</td> <td>   13.006</td> <td> 0.000</td> <td>    8.064</td> <td>   10.934</td>
</tr>
<tr>
  <th>dis</th>       <td>   -1.5509</td> <td>    0.168</td> <td>   -9.213</td> <td> 0.000</td> <td>   -1.882</td> <td>   -1.220</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>576.519</td> <th>  Durbin-Watson:     </th> <td>   0.952</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>37426.729</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.348</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>43.753</td>  <th>  Cond. No.          </th> <td>    9.32</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.391</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.390</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   323.9</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.69e-56</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:13</td>     <th>  Log-Likelihood:    </th> <td> -1680.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3366.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3374.</td>
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
  <th>Intercept</th> <td>   -2.2872</td> <td>    0.443</td> <td>   -5.157</td> <td> 0.000</td> <td>   -3.158</td> <td>   -1.416</td>
</tr>
<tr>
  <th>rad</th>       <td>    0.6179</td> <td>    0.034</td> <td>   17.998</td> <td> 0.000</td> <td>    0.550</td> <td>    0.685</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>656.459</td> <th>  Durbin-Watson:     </th> <td>   1.337</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>75417.007</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 6.478</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>61.389</td>  <th>  Cond. No.          </th> <td>    19.2</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.340</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.338</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   259.2</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.36e-47</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:15</td>     <th>  Log-Likelihood:    </th> <td> -1701.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3407.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3415.</td>
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
  <th>Intercept</th> <td>   -8.5284</td> <td>    0.816</td> <td>  -10.454</td> <td> 0.000</td> <td>  -10.131</td> <td>   -6.926</td>
</tr>
<tr>
  <th>tax</th>       <td>    0.0297</td> <td>    0.002</td> <td>   16.099</td> <td> 0.000</td> <td>    0.026</td> <td>    0.033</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>635.377</td> <th>  Durbin-Watson:     </th> <td>   1.252</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>63763.835</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 6.156</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>56.599</td>  <th>  Cond. No.          </th> <td>1.16e+03</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.16e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.082</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   46.26</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.94e-11</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:17</td>     <th>  Log-Likelihood:    </th> <td> -1784.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3572.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3581.</td>
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
  <th>Intercept</th> <td>  -17.6469</td> <td>    3.147</td> <td>   -5.607</td> <td> 0.000</td> <td>  -23.830</td> <td>  -11.464</td>
</tr>
<tr>
  <th>ptratio</th>   <td>    1.1520</td> <td>    0.169</td> <td>    6.801</td> <td> 0.000</td> <td>    0.819</td> <td>    1.485</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>568.053</td> <th>  Durbin-Watson:     </th> <td>   0.905</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>34221.853</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.245</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>41.899</td>  <th>  Cond. No.          </th> <td>    160.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.148</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.147</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   87.74</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.49e-19</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:20</td>     <th>  Log-Likelihood:    </th> <td> -1765.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3536.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3544.</td>
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
  <th>Intercept</th> <td>   16.5535</td> <td>    1.426</td> <td>   11.609</td> <td> 0.000</td> <td>   13.752</td> <td>   19.355</td>
</tr>
<tr>
  <th>black</th>     <td>   -0.0363</td> <td>    0.004</td> <td>   -9.367</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.029</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>594.029</td> <th>  Durbin-Watson:     </th> <td>   0.994</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>44041.935</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.578</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>47.323</td>  <th>  Cond. No.          </th> <td>1.49e+03</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.49e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.208</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.206</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   132.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.65e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:22</td>     <th>  Log-Likelihood:    </th> <td> -1747.5</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3499.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3507.</td>
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
  <th>Intercept</th> <td>   -3.3305</td> <td>    0.694</td> <td>   -4.801</td> <td> 0.000</td> <td>   -4.694</td> <td>   -1.968</td>
</tr>
<tr>
  <th>lstat</th>     <td>    0.5488</td> <td>    0.048</td> <td>   11.491</td> <td> 0.000</td> <td>    0.455</td> <td>    0.643</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>601.306</td> <th>  Durbin-Watson:     </th> <td>   1.182</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>49918.826</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.645</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>50.331</td>  <th>  Cond. No.          </th> <td>    29.7</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
next(models_keys_iter).summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.151</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.149</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   89.49</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.17e-19</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:24</td>     <th>  Log-Likelihood:    </th> <td> -1765.0</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3534.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3542.</td>
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
  <th>Intercept</th> <td>   11.7965</td> <td>    0.934</td> <td>   12.628</td> <td> 0.000</td> <td>    9.961</td> <td>   13.632</td>
</tr>
<tr>
  <th>medv</th>      <td>   -0.3632</td> <td>    0.038</td> <td>   -9.460</td> <td> 0.000</td> <td>   -0.439</td> <td>   -0.288</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>558.880</td> <th>  Durbin-Watson:     </th> <td>   0.996</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>32740.044</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 5.108</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>41.059</td>  <th>  Cond. No.          </th> <td>    64.5</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Now we check which predictors are statistically significant, using the common rule-of-thumb $p < 0.05$


```python
for model in models:
    if models[model].pvalues[1] < 0.05:
        print("{} is statistically significant with p-value {}". format(model, models[model].pvalues[1]))
    else:
        print("{} is NOT statistically significant with p-value {}". format(model, models[model].pvalues[1]))
```

    zn is statistically significant with p-value 5.506472107679307e-06
    indus is statistically significant with p-value 1.4503489330272395e-21
    chas is NOT statistically significant with p-value 0.2094345015352004
    nox is statistically significant with p-value 3.751739260356923e-23
    rm is statistically significant with p-value 6.346702984687839e-07
    age is statistically significant with p-value 2.8548693502441573e-16
    dis is statistically significant with p-value 8.519948766926326e-19
    rad is statistically significant with p-value 2.6938443981864414e-56
    tax is statistically significant with p-value 2.357126835257048e-47
    ptratio is statistically significant with p-value 2.942922447359816e-11
    black is statistically significant with p-value 2.487273973773734e-19
    lstat is statistically significant with p-value 2.6542772314731968e-27
    medv is statistically significant with p-value 1.1739870821943694e-19
    full is statistically significant with p-value 0.01702489149848948


Looks like all predictors were significant

## b. Full regression model


```python
# formula for full model
formula = ''
for predictor in predictors:
    formula += predictor + ' + '
formula = formula[:-3]
formula

# add full model
models['full'] = smf.ols('crim ~ ' + formula, data=boston).fit()
```


```python
models['full'].summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>crim</td>       <th>  R-squared:         </th> <td>   0.454</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.440</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   31.47</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.57e-56</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:49:14</td>     <th>  Log-Likelihood:    </th> <td> -1653.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3335.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   492</td>      <th>  BIC:               </th> <td>   3394.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   17.0332</td> <td>    7.235</td> <td>    2.354</td> <td> 0.019</td> <td>    2.818</td> <td>   31.248</td>
</tr>
<tr>
  <th>zn</th>        <td>    0.0449</td> <td>    0.019</td> <td>    2.394</td> <td> 0.017</td> <td>    0.008</td> <td>    0.082</td>
</tr>
<tr>
  <th>indus</th>     <td>   -0.0639</td> <td>    0.083</td> <td>   -0.766</td> <td> 0.444</td> <td>   -0.228</td> <td>    0.100</td>
</tr>
<tr>
  <th>chas</th>      <td>   -0.7491</td> <td>    1.180</td> <td>   -0.635</td> <td> 0.526</td> <td>   -3.068</td> <td>    1.570</td>
</tr>
<tr>
  <th>nox</th>       <td>  -10.3135</td> <td>    5.276</td> <td>   -1.955</td> <td> 0.051</td> <td>  -20.679</td> <td>    0.052</td>
</tr>
<tr>
  <th>rm</th>        <td>    0.4301</td> <td>    0.613</td> <td>    0.702</td> <td> 0.483</td> <td>   -0.774</td> <td>    1.634</td>
</tr>
<tr>
  <th>age</th>       <td>    0.0015</td> <td>    0.018</td> <td>    0.081</td> <td> 0.935</td> <td>   -0.034</td> <td>    0.037</td>
</tr>
<tr>
  <th>dis</th>       <td>   -0.9872</td> <td>    0.282</td> <td>   -3.503</td> <td> 0.001</td> <td>   -1.541</td> <td>   -0.433</td>
</tr>
<tr>
  <th>rad</th>       <td>    0.5882</td> <td>    0.088</td> <td>    6.680</td> <td> 0.000</td> <td>    0.415</td> <td>    0.761</td>
</tr>
<tr>
  <th>tax</th>       <td>   -0.0038</td> <td>    0.005</td> <td>   -0.733</td> <td> 0.464</td> <td>   -0.014</td> <td>    0.006</td>
</tr>
<tr>
  <th>ptratio</th>   <td>   -0.2711</td> <td>    0.186</td> <td>   -1.454</td> <td> 0.147</td> <td>   -0.637</td> <td>    0.095</td>
</tr>
<tr>
  <th>black</th>     <td>   -0.0075</td> <td>    0.004</td> <td>   -2.052</td> <td> 0.041</td> <td>   -0.015</td> <td>   -0.000</td>
</tr>
<tr>
  <th>lstat</th>     <td>    0.1262</td> <td>    0.076</td> <td>    1.667</td> <td> 0.096</td> <td>   -0.023</td> <td>    0.275</td>
</tr>
<tr>
  <th>medv</th>      <td>   -0.1989</td> <td>    0.061</td> <td>   -3.287</td> <td> 0.001</td> <td>   -0.318</td> <td>   -0.080</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>666.613</td> <th>  Durbin-Watson:     </th> <td>   1.519</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>84887.625</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 6.617</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>      <td>65.058</td>  <th>  Cond. No.          </th> <td>1.58e+04</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.58e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Now we see which predictors were significant 


*TBC...*

{% endkatexmm %}
