---
layout: page
title: 6. Linear Model Selection and Regularization
---

{% katexmm %}

# Exercise 8: Feature Selection on Simulated Data 

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-predictor-and-noise" data-toc-modified-id="a.-Generate-predictor-and-noise-1">a. Generate predictor and noise</a></span></li><li><span><a href="#b-generate-response" data-toc-modified-id="b.-Generate-response-2">b. Generate response</a></span></li><li><span><a href="#c-best-subset-selection" data-toc-modified-id="c.-Best-Subset-Selection-3">c. Best Subset Selection</a></span></li><li><span><a href="#d-forward-and-backward-stepwise-selection" data-toc-modified-id="d.-Forward-and-Backward-Stepwise-Selection-4">d. Forward and Backward Stepwise Selection</a></span></li><li><span><a href="#e-lasso" data-toc-modified-id="e.-Lasso-5">e. Lasso</a></span></li><li><span><a href="#f-repeat-for-a-new-response" data-toc-modified-id="f.-Repeat-for-a-new-response-6">f. Repeat for a new response</a></span><ul class="toc-item"><li><span><a href="#new-response" data-toc-modified-id="New-response-6.1">New response</a></span></li><li><span><a href="#bss-model" data-toc-modified-id="BSS-model-6.2">BSS model</a></span></li><li><span><a href="#lasso-model" data-toc-modified-id="Lasso-model-6.3">Lasso model</a></span></li></ul></li></ul></div>


## a. Generate predictor and noise


```python
import numpy as np

X, e = np.random.normal(size=100), np.random.normal(size=100)
```


```python
X.shape
```




    (100,)



## b. Generate response


```python
(beta_0, beta_1, beta_2, beta_3) = 1, 1, 1, 1

y = beta_0*np.array(100*[1]) + beta_1*X + beta_2*X**2 + beta_2*X**3 + e
```


```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set()

sns.scatterplot(x=X, y=y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a186bed68>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_6_1.png)


## c. Best Subset Selection

First we generate the predictors $X^2, \dots, X^{10}$ and assemble all data in a dataframe


```python
import pandas as pd

data = pd.DataFrame({'X^' + stri.: X**i for i in range(11)})
data['y'] = y
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
      <th>X^0</th>
      <th>X^1</th>
      <th>X^2</th>
      <th>X^3</th>
      <th>X^4</th>
      <th>X^5</th>
      <th>X^6</th>
      <th>X^7</th>
      <th>X^8</th>
      <th>X^9</th>
      <th>X^10</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.115404</td>
      <td>0.013318</td>
      <td>0.001537</td>
      <td>0.000177</td>
      <td>0.000020</td>
      <td>0.000002</td>
      <td>2.726196e-07</td>
      <td>3.146150e-08</td>
      <td>3.630796e-09</td>
      <td>4.190099e-10</td>
      <td>1.914734</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.594583</td>
      <td>0.353530</td>
      <td>-0.210203</td>
      <td>0.124983</td>
      <td>-0.074313</td>
      <td>0.044185</td>
      <td>-2.627180e-02</td>
      <td>1.562078e-02</td>
      <td>-9.287856e-03</td>
      <td>5.522406e-03</td>
      <td>0.848536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.751695</td>
      <td>3.068436</td>
      <td>5.374965</td>
      <td>9.415301</td>
      <td>16.492737</td>
      <td>28.890250</td>
      <td>5.060691e+01</td>
      <td>8.864789e+01</td>
      <td>1.552841e+02</td>
      <td>2.720104e+02</td>
      <td>11.091760</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.662284</td>
      <td>0.438620</td>
      <td>-0.290491</td>
      <td>0.192387</td>
      <td>-0.127415</td>
      <td>0.084385</td>
      <td>-5.588676e-02</td>
      <td>3.701289e-02</td>
      <td>-2.451304e-02</td>
      <td>1.623459e-02</td>
      <td>1.893104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-1.921016</td>
      <td>3.690302</td>
      <td>-7.089130</td>
      <td>13.618332</td>
      <td>-26.161034</td>
      <td>50.255764</td>
      <td>-9.654213e+01</td>
      <td>1.854590e+02</td>
      <td>-3.562696e+02</td>
      <td>6.843997e+02</td>
      <td>-4.628707</td>
    </tr>
  </tbody>
</table>
</div>



The `mlxtend` library has [exhaustive feature selection](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/#api)


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# dict for results
bss = {}

for k in range(1, 12):
    reg = LinearRegression()
    efs = EFS(reg, 
               min_features=k,
               max_features=k,
               scoring='neg_mean_squared_error',
               print_progress=False,
               cv=None)
    efs = efs.fit(data.drop(columns=['y']), data['y'])
    bss[k] = efs.best_idx_

bss
```




    {1: (3,),
     2: (2, 3),
     3: (1, 2, 3),
     4: (1, 2, 3, 7),
     5: (1, 2, 3, 9, 10),
     6: (1, 2, 3, 6, 8, 9),
     7: (1, 2, 3, 6, 7, 9, 10),
     8: (1, 2, 3, 4, 6, 8, 9, 10),
     9: (1, 2, 4, 5, 6, 7, 8, 9, 10),
     10: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
     11: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}



Now we'll calculate $AIC, BIC$ and adjusted $R^2$ for each subset size and plot (We'll use $AIC$ instead of $C_p$ since they're proportional).

First some helper functions for $AIC, BIC$ and adjusted $R^2$


```python
from math import log

def AIC(n, rss, d, var_e):
    return (1 / (n * var_e)) * (rss + 2 * d * var_e)

def BIC(n, rss, d, var_e):
    return (1 / (n * var_e)) * (rss + log(n) * d * var_e)

def adj_r2(n, rss, tss, d):
    return 1 - ((rss / (n - d - 1)) / (tss / (n - 1)))
```

Then calculate


```python
def mse_estimates(X, y, bss):
    n, results = X.shape[1], {}
    for k in bss:
        model_fit = LinearRegression().fit(X[:, bss[k]], y)
        y_pred = model_fit.predict(X[:, bss[k]])
        
        errors = y - y_pred
        rss = np.dot(errors, errors)
        var_e = np.var(errors)
        tss = np.dot((y - np.mean(y)), (y - np.mean(y)))
        
        results[k] = dict(AIC=AIC(n, rss, k, var_e), BIC=BIC(n, rss, k, var_e),
                          adj_r2=adj_r2(n, rss, rss, k))
    return results
```


```python
X, y = data.drop(columns=['y']).values, data['y'].values
mses = mse_estimates(X, y, bss)
mses
```

    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: RuntimeWarning: divide by zero encountered in double_scalars
      # Remove the CWD from sys.path while we load stuff.





    {1: {'AIC': 9.272727272727272,
      'BIC': 9.308899570254397,
      'adj_r2': -0.11111111111111094},
     2: {'AIC': 9.454545454545455, 'BIC': 9.526890049599706, 'adj_r2': -0.25},
     3: {'AIC': 9.636363636363633,
      'BIC': 9.744880528945009,
      'adj_r2': -0.4285714285714284},
     4: {'AIC': 9.81818181818182,
      'BIC': 9.962871008290318,
      'adj_r2': -0.6666666666666665},
     5: {'AIC': 10.0, 'BIC': 10.180861487635624, 'adj_r2': -1.0},
     6: {'AIC': 10.181818181818185, 'BIC': 10.398851966980931, 'adj_r2': -1.5},
     7: {'AIC': 10.363636363636365,
      'BIC': 10.616842446326237,
      'adj_r2': -2.3333333333333335},
     8: {'AIC': 10.545454545454545, 'BIC': 10.834832925671542, 'adj_r2': -4.0},
     9: {'AIC': 10.727272727272727, 'BIC': 11.05282340501685, 'adj_r2': -9.0},
     10: {'AIC': 10.909090909090908, 'BIC': 11.270813884362154, 'adj_r2': -inf},
     11: {'AIC': 11.090909090909093, 'BIC': 11.488804363707464, 'adj_r2': 11.0}}




```python
AICs = np.array([mses[k]['AIC'] for k in mses])
BICs = np.array([mses[k]['BIC'] for k in mses])
adj_r2s = np.array([mses[k]['adj_r2'] for k in mses])
```


```python
x = np.arange(1, 12)
sns.lineplot(x, AICs)
sns.lineplot(x, BICs)
sns.lineplot(x, adj_r2s)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1980a668>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_18_1.png)


The best model has the highest AIC/BIC and lowest adjusted $R^2$, so on this basis, the model with $X^3$ the only feature is the best. 

The coefficient is:


```python
bss_model = LinearRegression().fit(X[:, 3].reshape(-1, 1), y)
bss_model.coef_
```




    array([1.06262895])



Now, lets generate a validataion data set and check this model's mse


```python
X_valid, e_valid = np.random.normal(size=100), np.random.normal(size=100)
(beta_0, beta_1, beta_2, beta_3) = 1, 1, 1, 1
y_valid = beta_0*np.array(100*[1]) + beta_1*X_valid + beta_2*X_valid**2 + beta_2*X_valid**3 + e
```


```python
y_pred = bss_model.coef_ * X_valid**3
errors = y_pred - y_valid
bss_mse_test = np.mean(np.dot(errors, errors))
bss_mse_test
```




    642.7014227682031



We'll save the results for later comparison


```python
model_selection_df = pd.DataFrame({'mse_test': [bss_mse_test]}, index=['bss'])
model_selection_df
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
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bss</th>
      <td>642.701423</td>
    </tr>
  </tbody>
</table>
</div>



## d. Forward and Backward Stepwise Selection

`mlxtend` also has forward and backward stepwise selection

First we look at forward stepwise selection.


```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

X, y = data.drop(columns=['y']).values, data['y'].values
# dict for results
fss = {}

for k in range(1, 12):
    sfs = SFS(LinearRegression(), 
          k_features=k, 
          forward=True, 
          floating=False, 
          scoring='neg_mean_squared_error')
    sfs = sfs.fit(X, y)
    fss[k] = sfs.k_feature_idx_

fss
```




    {1: (3,),
     2: (2, 3),
     3: (1, 2, 3),
     4: (1, 2, 3, 9),
     5: (0, 1, 2, 3, 9),
     6: (0, 1, 2, 3, 9, 10),
     7: (0, 1, 2, 3, 5, 9, 10),
     8: (0, 1, 2, 3, 4, 5, 9, 10),
     9: (0, 1, 2, 3, 4, 5, 7, 9, 10),
     10: (0, 1, 2, 3, 4, 5, 7, 8, 9, 10),
     11: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}




```python
mses = mse_estimates(X, y, fss)
mses
```

    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: RuntimeWarning: divide by zero encountered in double_scalars
      # Remove the CWD from sys.path while we load stuff.





    {1: {'AIC': 9.272727272727272,
      'BIC': 9.308899570254397,
      'adj_r2': -0.11111111111111094},
     2: {'AIC': 9.454545454545455, 'BIC': 9.526890049599706, 'adj_r2': -0.25},
     3: {'AIC': 9.636363636363633,
      'BIC': 9.744880528945009,
      'adj_r2': -0.4285714285714284},
     4: {'AIC': 9.818181818181818,
      'BIC': 9.962871008290316,
      'adj_r2': -0.6666666666666667},
     5: {'AIC': 10.0, 'BIC': 10.180861487635623, 'adj_r2': -1.0},
     6: {'AIC': 10.181818181818182, 'BIC': 10.39885196698093, 'adj_r2': -1.5},
     7: {'AIC': 10.363636363636365,
      'BIC': 10.616842446326237,
      'adj_r2': -2.3333333333333335},
     8: {'AIC': 10.545454545454547, 'BIC': 10.834832925671542, 'adj_r2': -4.0},
     9: {'AIC': 10.727272727272727, 'BIC': 11.052823405016847, 'adj_r2': -9.0},
     10: {'AIC': 10.90909090909091, 'BIC': 11.270813884362157, 'adj_r2': -inf},
     11: {'AIC': 11.090909090909093, 'BIC': 11.488804363707464, 'adj_r2': 11.0}}




```python
AICs = np.array([mses[k]['AIC'] for k in mses])
BICs = np.array([mses[k]['BIC'] for k in mses])
adj_r2s = np.array([mses[k]['adj_r2'] for k in mses])
```


```python
x = np.arange(1, 12)
sns.lineplot(x, AICs)
sns.lineplot(x, BICs)
sns.lineplot(x, adj_r2s)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a198f0d68>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_31_1.png)


FSS also selects the model with $X^3$ the only feature.

Now we consider backward stepwise selection.


```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# dict for results
bkss = {}

for k in range(1, 12):
    sfs = SFS(LinearRegression(), 
          k_features=k, 
          forward=False, 
          floating=False, 
          scoring='neg_mean_squared_error')
    sfs = sfs.fit(X, y)
    bkss[k] = sfs.k_feature_idx_

bkss
```




    {1: (3,),
     2: (2, 3),
     3: (1, 2, 3),
     4: (1, 2, 3, 9),
     5: (1, 2, 3, 8, 9),
     6: (1, 2, 3, 7, 8, 9),
     7: (1, 2, 3, 6, 7, 8, 9),
     8: (0, 1, 2, 3, 6, 7, 8, 9),
     9: (0, 1, 2, 3, 5, 6, 7, 8, 9),
     10: (0, 1, 2, 3, 5, 6, 7, 8, 9, 10),
     11: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}




```python
mses = mse_estimates(X, y, bkss)
mses
```

    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: RuntimeWarning: divide by zero encountered in double_scalars
      # Remove the CWD from sys.path while we load stuff.





    {1: {'AIC': 9.272727272727272,
      'BIC': 9.308899570254397,
      'adj_r2': -0.11111111111111094},
     2: {'AIC': 9.454545454545455, 'BIC': 9.526890049599706, 'adj_r2': -0.25},
     3: {'AIC': 9.636363636363633,
      'BIC': 9.744880528945009,
      'adj_r2': -0.4285714285714284},
     4: {'AIC': 9.818181818181818,
      'BIC': 9.962871008290316,
      'adj_r2': -0.6666666666666667},
     5: {'AIC': 10.0, 'BIC': 10.180861487635623, 'adj_r2': -1.0},
     6: {'AIC': 10.181818181818182, 'BIC': 10.398851966980928, 'adj_r2': -1.5},
     7: {'AIC': 10.363636363636363,
      'BIC': 10.616842446326235,
      'adj_r2': -2.333333333333333},
     8: {'AIC': 10.545454545454543, 'BIC': 10.83483292567154, 'adj_r2': -4.0},
     9: {'AIC': 10.727272727272725, 'BIC': 11.052823405016847, 'adj_r2': -9.0},
     10: {'AIC': 10.90909090909091, 'BIC': 11.270813884362155, 'adj_r2': -inf},
     11: {'AIC': 11.090909090909093, 'BIC': 11.488804363707464, 'adj_r2': 11.0}}




```python
AICs = np.array([mses[k]['AIC'] for k in mses])
BICs = np.array([mses[k]['BIC'] for k in mses])
adj_r2s = np.array([mses[k]['adj_r2'] for k in mses])
```


```python
x = np.arange(1, 12)
sns.lineplot(x, AICs)
sns.lineplot(x, BICs)
sns.lineplot(x, adj_r2s)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a199cccc0>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_36_1.png)


BKSS also selects the model with $X^3$ the only feature.

## e. Lasso


```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

alphas = np.array([10**i for i in np.linspace(-3, 0, num=20)])
lassos = {'alpha': alphas, '5_fold_cv_error': []}
```


```python
for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1e8, tol=1e-2)
        cv_error = np.mean(- cross_val_score(lasso, X, y, cv=5, 
                                             scoring='neg_mean_squared_error'))
        lassos['5_fold_cv_error'] += [cv_error]
```


```python
lassos_df = pd.DataFrame(lassos)
lassos_df
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
      <th>alpha</th>
      <th>5_fold_cv_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001000</td>
      <td>0.836428</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001438</td>
      <td>0.819378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002069</td>
      <td>0.895901</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002976</td>
      <td>0.893177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.004281</td>
      <td>0.888498</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.006158</td>
      <td>0.885597</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.008859</td>
      <td>0.884861</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.012743</td>
      <td>0.884489</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.018330</td>
      <td>0.888298</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.026367</td>
      <td>0.894860</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.037927</td>
      <td>0.906162</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.054556</td>
      <td>0.914259</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.078476</td>
      <td>0.930345</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.112884</td>
      <td>0.968640</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.162378</td>
      <td>1.049202</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.233572</td>
      <td>1.192103</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.335982</td>
      <td>1.506187</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.483293</td>
      <td>2.109461</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.695193</td>
      <td>2.854881</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.000000</td>
      <td>3.368783</td>
    </tr>
  </tbody>
</table>
</div>




```python
(alpha_hat, cv_error_min) = lassos_df.iloc[lassos_df['5_fold_cv_error'].idxmin(), ]
(alpha_hat, cv_error_min)
```




    (0.0014384498882876629, 0.8193781004864231)




```python
sns.lineplot(x='alpha', y='5_fold_cv_error', data=lassos_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a19a8b5c0>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_43_1.png)


Now we'll see what coefficient estimates this model produces


```python
np.set_printoptions(suppress=True)
lasso_model = Lasso(alpha=alpha_hat, fit_intercept=False, max_iter=1e8, tol=1e-2).fit(X, y)
lasso_model.coef_
```




    array([ 1.06544938,  0.80612409,  2.05979396,  0.7883535 , -1.66945357,
            0.24562823,  0.76952192, -0.0535788 , -0.13390294,  0.0027605 ,
            0.00767839])




```python
model_df = pd.DataFrame({'coef_': lasso_model.coef_}, index=data.columns[:-1])
model_df.sort_values(by='coef_', ascending=False)
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
      <th>coef_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X^2</th>
      <td>2.059794</td>
    </tr>
    <tr>
      <th>X^0</th>
      <td>1.065449</td>
    </tr>
    <tr>
      <th>X^1</th>
      <td>0.806124</td>
    </tr>
    <tr>
      <th>X^3</th>
      <td>0.788354</td>
    </tr>
    <tr>
      <th>X^6</th>
      <td>0.769522</td>
    </tr>
    <tr>
      <th>X^5</th>
      <td>0.245628</td>
    </tr>
    <tr>
      <th>X^10</th>
      <td>0.007678</td>
    </tr>
    <tr>
      <th>X^9</th>
      <td>0.002760</td>
    </tr>
    <tr>
      <th>X^7</th>
      <td>-0.053579</td>
    </tr>
    <tr>
      <th>X^8</th>
      <td>-0.133903</td>
    </tr>
    <tr>
      <th>X^4</th>
      <td>-1.669454</td>
    </tr>
  </tbody>
</table>
</div>



Let's check this model on a validation set:


```python
data_valid = pd.DataFrame({'X_valid^' + stri.: X_valid**i for i in range(11)})
data_valid['y_valid'] = y_valid
data_valid.head()
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
      <th>X_valid^0</th>
      <th>X_valid^1</th>
      <th>X_valid^2</th>
      <th>X_valid^3</th>
      <th>X_valid^4</th>
      <th>X_valid^5</th>
      <th>X_valid^6</th>
      <th>X_valid^7</th>
      <th>X_valid^8</th>
      <th>X_valid^9</th>
      <th>X_valid^10</th>
      <th>y_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.315735</td>
      <td>1.731159</td>
      <td>2.277747</td>
      <td>2.996911</td>
      <td>3.943142</td>
      <td>5.188130</td>
      <td>6.826205</td>
      <td>8.981478</td>
      <td>11.817247</td>
      <td>15.548367</td>
      <td>7.109116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.754402</td>
      <td>0.569123</td>
      <td>0.429348</td>
      <td>0.323901</td>
      <td>0.244352</td>
      <td>0.184340</td>
      <td>0.139066</td>
      <td>0.104912</td>
      <td>0.079146</td>
      <td>0.059708</td>
      <td>3.052666</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.910262</td>
      <td>0.828578</td>
      <td>0.754223</td>
      <td>0.686541</td>
      <td>0.624932</td>
      <td>0.568852</td>
      <td>0.517805</td>
      <td>0.471338</td>
      <td>0.429041</td>
      <td>0.390540</td>
      <td>3.389726</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.528947</td>
      <td>0.279785</td>
      <td>-0.147991</td>
      <td>0.078279</td>
      <td>-0.041406</td>
      <td>0.021901</td>
      <td>-0.011585</td>
      <td>0.006128</td>
      <td>-0.003241</td>
      <td>0.001714</td>
      <td>2.010106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-1.271336</td>
      <td>1.616296</td>
      <td>-2.054856</td>
      <td>2.612413</td>
      <td>-3.321256</td>
      <td>4.222433</td>
      <td>-5.368133</td>
      <td>6.824703</td>
      <td>-8.676493</td>
      <td>11.030740</td>
      <td>-1.018760</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred = lasso_model.predict(data_valid.drop(columns=['y_valid']))
errors = y_pred - y_valid
lasso_mse_test = np.mean(np.dot(errors, errors))
lasso_mse_test
```




    90.33155498089704




```python
model_selection_df = model_selection_df.append(pd.DataFrame({'mse_test': [lasso_mse_test]}, index=['lasso']))
model_selection_df
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
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bss</th>
      <td>642.701423</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>90.331555</td>
    </tr>
  </tbody>
</table>
</div>



A considerable improvement over the BSS model

## f. Repeat for a new response

We now repeat the above for a model

$$Y = \beta_0 + \beta_7 X^t + \epsilon$$

### New response


```python
# new train/test response
y = np.array(100*[1]) + data['X^7'] + e

# new validation response
y_valid = np.array(100*[1]) + data_valid['X_valid^7'] + e

# update dfs
data.loc[:, 'y'], data_valid.loc[:, 'y_valid'] = y, y_valid
```

### BSS model


```python
bss = {}

for k in range(1, 12):
    reg = LinearRegression()
    efs = EFS(reg, 
               min_features=k,
               max_features=k,
               scoring='neg_mean_squared_error',
               print_progress=False,
               cv=None)
    efs = efs.fit(data.drop(columns=['y']), data['y'])
    bss[k] = efs.best_idx_

bss
```




    {1: (7,),
     2: (7, 9),
     3: (1, 3, 7),
     4: (1, 5, 7, 10),
     5: (1, 7, 8, 9, 10),
     6: (1, 4, 6, 7, 8, 9),
     7: (2, 4, 6, 7, 8, 9, 10),
     8: (1, 2, 4, 5, 6, 7, 8, 10),
     9: (2, 3, 4, 5, 6, 7, 8, 9, 10),
     10: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
     11: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}




```python
X, y = data.drop(columns=['y']).values, data['y'].values
mses = mse_estimates(X, y, bss)
mses
```

    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: RuntimeWarning: divide by zero encountered in double_scalars
      # Remove the CWD from sys.path while we load stuff.





    {1: {'AIC': 9.272727272727272,
      'BIC': 9.308899570254397,
      'adj_r2': -0.11111111111111116},
     2: {'AIC': 9.454545454545455, 'BIC': 9.526890049599706, 'adj_r2': -0.25},
     3: {'AIC': 9.636363636363637,
      'BIC': 9.74488052894501,
      'adj_r2': -0.4285714285714286},
     4: {'AIC': 9.818181818181817,
      'BIC': 9.962871008290316,
      'adj_r2': -0.6666666666666667},
     5: {'AIC': 10.000000000000002, 'BIC': 10.180861487635623, 'adj_r2': -1.0},
     6: {'AIC': 10.18181818181818, 'BIC': 10.398851966980928, 'adj_r2': -1.5},
     7: {'AIC': 10.363636363636365,
      'BIC': 10.616842446326238,
      'adj_r2': -2.3333333333333335},
     8: {'AIC': 10.545454545454547, 'BIC': 10.834832925671542, 'adj_r2': -4.0},
     9: {'AIC': 10.727272727272723, 'BIC': 11.052823405016845, 'adj_r2': -9.0},
     10: {'AIC': 10.909090909090908, 'BIC': 11.270813884362154, 'adj_r2': -inf},
     11: {'AIC': 11.090909090909088, 'BIC': 11.488804363707459, 'adj_r2': 11.0}}




```python
AICs = np.array([mses[k]['AIC'] for k in mses])
BICs = np.array([mses[k]['BIC'] for k in mses])
adj_r2s = np.array([mses[k]['adj_r2'] for k in mses])
```


```python
x = np.arange(1, 12)
sns.lineplot(x, AICs)
sns.lineplot(x, BICs)
sns.lineplot(x, adj_r2s)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1a241c18>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_60_1.png)


Again, BSS likes a single predictor (this time $X^{10}$)


```python
bss_model = LinearRegression().fit(data['X^7'].values.reshape(-1,1), data['y'].values)
bss_model.coef_
```




    array([0.99897181])



Now, lets generate a validataion data set and check this model's mse


```python
y_pred = bss_model.coef_ * X_valid**3
errors = y_pred - y_valid
bss_mse_test = np.mean(np.dot(errors, errors))
bss_mse_test
```




    91636.16819263707




```python
model_selection_df = pd.DataFrame({'mse_test': [bss_mse_test]}, index=['bss'])
model_selection_df
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
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bss</th>
      <td>91636.168193</td>
    </tr>
  </tbody>
</table>
</div>



### Lasso model


```python
alphas = np.array([10**i for i in np.linspace(-4, 1, num=50)])
lassos = {'alpha': alphas, '5_fold_cv_error': []}

for alpha in alphas:
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1e6, tol=1e-3)
        cv_error = np.mean(- cross_val_score(lasso, X, y, cv=5, 
                                             scoring='neg_mean_squared_error'))
        lassos['5_fold_cv_error'] += [cv_error]
```


```python
(alpha, cv_error_min) = lassos_df.iloc[lassos_df['5_fold_cv_error'].idxmin(), ]
(alpha, cv_error_min)
```




    (0.028117686979742307, 2.750654968384496)




```python
lassos_df = pd.DataFrame(lassos)
sns.lineplot(x='alpha', y='5_fold_cv_error', data=lassos_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1a120da0>




![png]({{site.baseurl}}/assets/images/ch06_exercise_8_69_1.png)


Now we'll see what coefficient estimates this model produces


```python
lasso_model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1e8, tol=1e-2).fit(X, y)
lasso_model.coef_
```




    array([-0.        ,  1.43769581,  4.23293404, -4.72479707, -1.3268075 ,
            2.69250184, -0.02715976,  0.40441206,  0.01230247,  0.04725243,
            0.00255472])




```python
model_df = pd.DataFrame({'coef_': lasso_model.coef_}, index=data.columns[:-1])
model_df.sort_values(by='coef_', ascending=False)
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
      <th>coef_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X^2</th>
      <td>4.232934</td>
    </tr>
    <tr>
      <th>X^5</th>
      <td>2.692502</td>
    </tr>
    <tr>
      <th>X^1</th>
      <td>1.437696</td>
    </tr>
    <tr>
      <th>X^7</th>
      <td>0.404412</td>
    </tr>
    <tr>
      <th>X^9</th>
      <td>0.047252</td>
    </tr>
    <tr>
      <th>X^8</th>
      <td>0.012302</td>
    </tr>
    <tr>
      <th>X^10</th>
      <td>0.002555</td>
    </tr>
    <tr>
      <th>X^0</th>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>X^6</th>
      <td>-0.027160</td>
    </tr>
    <tr>
      <th>X^4</th>
      <td>-1.326808</td>
    </tr>
    <tr>
      <th>X^3</th>
      <td>-4.724797</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_pred = lasso_model.predict(data_valid.drop(columns=['y_valid']))
errors = y_pred - y_valid
lasso_mse_test = np.mean(np.dot(errors, errors))
lasso_mse_test
```




    292.261615448788




```python
model_selection_df = model_selection_df.append(pd.DataFrame({'mse_test': [lasso_mse_test]}, index=['lasso']))
model_selection_df
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
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bss</th>
      <td>91636.168193</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>292.261615</td>
    </tr>
  </tbody>
</table>
</div>



Once again, the lasso dramatically outperforms.

{% endkatexmm %}
