---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Exercise 8: Cross-validation on simulated data

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-data" data-toc-modified-id="a.-Generate-data-1">a. Generate data</a></span></li><li><span><a href="#b-scatter-plot" data-toc-modified-id="b.-Scatter-plot-2">b. Scatter plot</a></span></li><li><span><a href="#c-loocv-errors-for-various-models" data-toc-modified-id="c.-LOOCV-errors-for-various-models-3">c. LOOCV errors for various models</a></span><ul class="toc-item"><li><span><a href="#dataframe" data-toc-modified-id="Dataframe-3.1">Dataframe</a></span></li><li><span><a href="#models" data-toc-modified-id="Models-3.2">Models</a></span></li><li><span><a href="#loocv-errors" data-toc-modified-id="LOOCV-errors-3.3">LOOCV errors</a></span></li></ul></li><li><span><a href="#d-repeat-c" data-toc-modified-id="d.-Repeat-c.-4">d. Repeat c.</a></span><ul class="toc-item"><li><span><a href="#e-which-model-had-the-smallest-error" data-toc-modified-id="e.-Which-model-had-the-smallest-error?-4.1">e. Which model had the smallest error?</a></span></li><li><span><a href="#f-hypothesis-testing-the-coefficients" data-toc-modified-id="f.-Hypothesis-testing-the-coefficients-4.2">f. Hypothesis testing the coefficients</a></span></li></ul></li></ul></div>


## a. Generate data


```python
import numpy as np

np.random.seed(0)

X = np.random.normal(size=100)
Y = -2*X**2 + X + np.random.normal(size=100)
```

The model here is 

$$Y = -2 X^2 + X + \epsilon$$

Where $X, \epsilon \sim \text{Normal}(0, 1)$. 

The sample size is $n=100$. Since we don't know polynomial regression yet, we have $p=2$, i.e. $X_1 = X, X_2 = X^2$. 

## b. Scatter plot


```python
import seaborn as sns
%matplotlib inline

sns.scatterplot(X, Y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1f70d4e0>




![png]({{site.baseurl}}/assets/images/ch05_exercise_8_5_1.png)


## c. LOOCV errors for various models

### Dataframe


```python
import pandas as pd

data = pd.DataFrame({'const': len(X)*[1], 'X': X, 'Y': Y})
for i in range(2,5):
    data['X_' + stri.] = X**i

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>const</th>
      <th>X</th>
      <th>Y</th>
      <th>X_2</th>
      <th>X_3</th>
      <th>X_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.764052</td>
      <td>-2.576558</td>
      <td>3.111881</td>
      <td>5.489520</td>
      <td>9.683801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.400157</td>
      <td>-1.267853</td>
      <td>0.160126</td>
      <td>0.064075</td>
      <td>0.025640</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.978738</td>
      <td>-2.207603</td>
      <td>0.957928</td>
      <td>0.937561</td>
      <td>0.917626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2.240893</td>
      <td>-6.832915</td>
      <td>5.021602</td>
      <td>11.252875</td>
      <td>25.216490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1.867558</td>
      <td>-6.281111</td>
      <td>3.487773</td>
      <td>6.513618</td>
      <td>12.164559</td>
    </tr>
  </tbody>
</table>
</div>



### Models


```python
from sklearn.linear_model import LinearRegression

models = {}
models['deg1'] = LinearRegression(data[['const', 'X']], data['Y'])
models['deg2'] = LinearRegression(data[['const', 'X', 'X_2']], data['Y'])
models['deg3'] = LinearRegression(data[['const', 'X', 'X_2', 'X_3']], data['Y'])
models['deg4'] = LinearRegression(data[['const', 'X', 'X_2', 'X_3', 'X_4']], data['Y'])
```

### LOOCV errors


```python
from sklearn.model_selection import LeaveOneOut

loocv = LeaveOneOut()

errors = []

### Degree 1 model
X = data[['const', 'X']].values
y = data['Y'].values
y_pred = np.array([])

for train_index, test_index in loocv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_pred = np.append(y_pred, LinearRegression().fit(X_train, y_train).predict(X_test))

errors += [abs(y-y_pred).mean()]

### Degree 2 model
X = data[['const', 'X', 'X_2']].values
y = data['Y'].values
y_pred = np.array([])

for train_index, test_index in loocv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_pred = np.append(y_pred, LinearRegression().fit(X_train, y_train).predict(X_test))

errors += [abs(y-y_pred).mean()]

### Degree 3 model
X = data[['const', 'X', 'X_2', 'X_3']].values
y = data['Y'].values
y_pred = np.array([])

for train_index, test_index in loocv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_pred = np.append(y_pred, LinearRegression().fit(X_train, y_train).predict(X_test))

errors += [abs(y-y_pred).mean()]

### Degree 4 model
X = data[['const', 'X', 'X_2', 'X_3', 'X_4']].values
y = data['Y'].values
y_pred = np.array([])

for train_index, test_index in loocv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_pred = np.append(y_pred, LinearRegression().fit(X_train, y_train).predict(X_test))

errors += [abs(y-y_pred).mean()]
```


```python
model_names = ['deg' + stri. for i in range(1,5)]
errors_df = pd.DataFrame({'model': model_names, 'est_LOOCV_err': errors})
errors_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>est_LOOCV_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>deg1</td>
      <td>2.239087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deg2</td>
      <td>0.904460</td>
    </tr>
    <tr>
      <th>2</th>
      <td>deg3</td>
      <td>0.917489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>deg4</td>
      <td>0.925485</td>
    </tr>
  </tbody>
</table>
</div>



## d. Repeat c.

If we repeat c. we don't get any difference, since LOOCV is deterministic.

### e. Which model had the smallest error?

The degree 2 model had the smallest error. This is to be expected. Since the original data was generated by a degree 2, we expect a degree 2 model to have lower test error, and the LOOCV is an estimate of the test error

### f. Hypothesis testing the coefficients


```python
import statsmodels.formula.api as smf

smf.ols('Y ~ X', data=data).fit().summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -1.9487</td> <td>    0.290</td> <td>   -6.726</td> <td> 0.000</td> <td>   -2.524</td> <td>   -1.374</td>
</tr>
<tr>
  <th>X</th>         <td>    0.8650</td> <td>    0.287</td> <td>    3.015</td> <td> 0.003</td> <td>    0.296</td> <td>    1.434</td>
</tr>
</table>




```python
smf.ols('Y ~ X + X_2', data=data).fit().summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.1427</td> <td>    0.132</td> <td>    1.079</td> <td> 0.283</td> <td>   -0.120</td> <td>    0.405</td>
</tr>
<tr>
  <th>X</th>         <td>    1.1230</td> <td>    0.104</td> <td>   10.829</td> <td> 0.000</td> <td>    0.917</td> <td>    1.329</td>
</tr>
<tr>
  <th>X_2</th>       <td>   -2.0668</td> <td>    0.080</td> <td>  -25.700</td> <td> 0.000</td> <td>   -2.226</td> <td>   -1.907</td>
</tr>
</table>




```python
smf.ols('Y ~ X + X_2 + X_3', data=data).fit().summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.1432</td> <td>    0.133</td> <td>    1.077</td> <td> 0.284</td> <td>   -0.121</td> <td>    0.407</td>
</tr>
<tr>
  <th>X</th>         <td>    1.1626</td> <td>    0.195</td> <td>    5.975</td> <td> 0.000</td> <td>    0.776</td> <td>    1.549</td>
</tr>
<tr>
  <th>X_2</th>       <td>   -2.0668</td> <td>    0.081</td> <td>  -25.575</td> <td> 0.000</td> <td>   -2.227</td> <td>   -1.906</td>
</tr>
<tr>
  <th>X_3</th>       <td>   -0.0148</td> <td>    0.061</td> <td>   -0.240</td> <td> 0.810</td> <td>   -0.137</td> <td>    0.107</td>
</tr>
</table>




```python
smf.ols('Y ~ X + X_2 + X_3 + X_4', data=data).fit().summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.2399</td> <td>    0.153</td> <td>    1.563</td> <td> 0.121</td> <td>   -0.065</td> <td>    0.545</td>
</tr>
<tr>
  <th>X</th>         <td>    1.1207</td> <td>    0.197</td> <td>    5.691</td> <td> 0.000</td> <td>    0.730</td> <td>    1.512</td>
</tr>
<tr>
  <th>X_2</th>       <td>   -2.3116</td> <td>    0.212</td> <td>  -10.903</td> <td> 0.000</td> <td>   -2.732</td> <td>   -1.891</td>
</tr>
<tr>
  <th>X_3</th>       <td>    0.0049</td> <td>    0.063</td> <td>    0.078</td> <td> 0.938</td> <td>   -0.121</td> <td>    0.130</td>
</tr>
<tr>
  <th>X_4</th>       <td>    0.0556</td> <td>    0.045</td> <td>    1.248</td> <td> 0.215</td> <td>   -0.033</td> <td>    0.144</td>
</tr>
</table>



Observations:

- The degree 1 model fit a constant coeffient with high significance while the higher degree models didn't.
- The higher degree models all fit $X$ and $X^2$ coefficients with high significance but constant and higher degree coefficients with very low significance.

These results are consistent with the LOOCV error results, which suggested a second degree model was best. If we decide which predictors to reject based on these hypothesis tests, we would end up with a model

$$Y = \beta_1X + \beta_2X^2 + \epsilon $$

which is the form of the true model

{% endkatexmm %}