---
layout: page
title: 7. Moving Beyond Linearity
---


{% katexmm %}

# Exercise 8: Investigating non-linear relationships in `Auto` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span><ul class="toc-item"><li><span><a href="#import" data-toc-modified-id="Import-1.1">Import</a></span></li><li><span><a href="#encode-categorical-variables" data-toc-modified-id="Encode-categorical-variables-1.2">Encode categorical variables</a></span></li></ul></li><li><span><a href="#inspecting-the-data" data-toc-modified-id="Inspecting-the-Data-2">Inspecting the Data</a></span></li><li><span><a href="#modeling-some-non-linear-relationships-with-mpg" data-toc-modified-id="Modeling-some-non-linear-relationships-with-mpg-3">Modeling some non-linear relationships with <code>mpg</code></a></span><ul class="toc-item"><li><span><a href="#local-regression" data-toc-modified-id="Local-Regression-3.1">Local Regression</a></span></li><li><span><a href="#Polynomial-Regression" data-toc-modified-id="Polynomial-Regression-3.2">Polynomial Regression</a></span></li><li><span><a href="#cubic-p-spline-regression" data-toc-modified-id="Cubic-P-Spline-Regression-3.3">Cubic P-Spline Regression</a></span></li><li><span><a href="#model-comparison" data-toc-modified-id="Model-Comparison-3.4">Model Comparison</a></span></li><li><span><a href="#optimal-models-and-their-test-errors" data-toc-modified-id="Optimal-models-and-their-test-errors-3.5">Optimal models and their test errors</a></span></li><li><span><a href="#analysis-of-optimal-models" data-toc-modified-id="Analysis-of-optimal-models-3.6">Analysis of optimal models</a></span></li><li><span><a href="#mpg-vs-accleration" data-toc-modified-id="mpg-vs-accleration-3.7"><code>mpg</code> vs <code>accleration</code></a></span></li><li><span><a href="#mpg-vs-weight" data-toc-modified-id="mpg-vs.-weight-3.8"><code>mpg</code> vs. <code>weight</code></a></span></li><li><span><a href="#mpg-vs-horsepower" data-toc-modified-id="mpg-vs.-horsepower-3.9"><code>mpg</code> vs. <code>horsepower</code></a></span></li><li><span><a href="#mpg-vs-displacement" data-toc-modified-id="mpg-vs.-displacement-3.10"><code>mpg</code> vs. <code>displacement</code></a></span></li></ul></li><li><span><a href="#gam-for-predicting-mpg" data-toc-modified-id="GAM-for-predicting-mpg-4">GAM for predicting <code>mpg</code></a></span><ul class="toc-item"><li><span><a href="#find-variables-with-linear-relationships-to-mpg" data-toc-modified-id="Find-variables-with-linear-relationships-to-mpg-4.1">Find variables with linear relationships to <code>mpg</code></a></span></li><li><span><a href="#train-gam" data-toc-modified-id="Train-GAM-4.2">Train GAM</a></span></li><li><span><a href="#compare-to-alternative-regression-models" data-toc-modified-id="Compare-to-alternative-regression-models-4.3">Compare to alternative regression models</a></span></li></ul></li></ul></div>

## Preparing the data

### Import


```python
# standard imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

auto = pd.read_csv('../../datasets/Auto.csv', index_col=0)
auto.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
auto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 392 entries, 1 to 397
    Data columns (total 9 columns):
    mpg             392 non-null float64
    cylinders       392 non-null int64
    displacement    392 non-null float64
    horsepower      392 non-null int64
    weight          392 non-null int64
    acceleration    392 non-null float64
    year            392 non-null int64
    origin          392 non-null int64
    name            392 non-null object
    dtypes: float64(3), int64(5), object(1)
    memory usage: 30.6+ KB


### Encode categorical variables

The only categorical (non-ordinal) variable is `origin`


```python
# numerical df with one hot encoding for origin variable
auto_num = pd.get_dummies(auto, columns=['origin'])
auto_num.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>name</th>
      <th>origin_1</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>chevrolet chevelle malibu</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>buick skylark 320</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>plymouth satellite</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>amc rebel sst</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>ford torino</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Inspecting the Data

Before training models we'll do some inspection to see if non-linear relationships are suggested by the data.

A pairplot produces all possible scatterplots of the variables


```python
# pair plot to inspect distributions and scatterplots
sns.pairplot(data=auto, diag_kind='kde', plot_kws={'alpha':0.5},
             diag_kws={'alpha':0.5})
```




    <seaborn.axisgrid.PairGrid at 0x1a20dc77b8>




![png]({{site.baseurl}}/assets/images/ch07_exercise_08_10_1.png)


Observations:

- The plots strongly suggest that variables `weight`, `horsepower` and `displacement` have non-linear relationships to `mpg`
- The plots weakly suggest that `displacement` and `horsepower` may have a non-linear relationships to `acceleration`
- There is strong suggestion of some linear relationships as well.
- It's difficult to identify a non-linear relationship between two variables when one of them is discrete (e.g. `cylinders`, `year`)

## Modeling some non-linear relationships with `mpg`

Based on the pairplot we'll investigate non-linear relationships between  `acceleration`, `weight`, `horsepower`, and `displacement` with `mpg`. We'll try a few different types of models, using cross-validation to optimize, and then compare


```python
cols = ['acceleration', 'weight', 'horsepower', 'displacement', 'mpg']
df = auto[cols]

# normalize
df = (df - df.mean())/df.std()

# series
acc, wt, hp, dp, mpg = df.acceleration.values, df.weight.values, df.horsepower.values, df.displacement.values, df.mpg.values
```

### Local Regression


```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

lr_param_grid = {'n_neighbors': np.arange(1,7), 'weights': ['uniform', 'distance'], 
                     'p': np.arange(1, 7)}
lr_searches = [GridSearchCV(KNeighborsRegressor(), lr_param_grid, cv=5, 
                         scoring='neg_mean_squared_error') for i in range(4)]
```


```python
%%capture
var_pairs = {'acc_mpg', 'wt_mpg', 'hp_mpg', 'dp_mpg'}
models = {name:None for name in ['local', 'poly', 'p-spline']}

models['local'] = {pair:None for pair in var_pairs}
models['local']['acc_mpg'] = lr_searches[0].fit(acc.reshape(-1, 1), mpg)
models['local']['wt_mpg'] = lr_searches[1].fit(wt.reshape(-1, 1), mpg)
models['local']['hp_mpg'] = lr_searches[2].fit(hp.reshape(-1, 1), mpg)
models['local']['dp_mpg'] = lr_searches[3].fit(dp.reshape(-1, 1), mpg)
```

### Polynomial Regression


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
reg_tree_search.best_params_

# 6-fold cv estimate of test rmse
np.sqrt(-reg_tree_search.best_score_)

from sklearn.metrics import mean_squared_error

# test set mse
final_reg_tree = reg_tree_search.best_estimator_
reg_tree_test_mse = mean_squared_error(final_reg_tree.predict(X_test), y_test)
np.sqrt(reg_tree_test_mse)
pr_pipe = Pipeline(steps=[('poly', PolynomialFeatures()), ('ridge', Ridge())])
pr_pipe_param_grid = {'poly__degree': np.arange(1, 5), 'ridge__alpha': np.logspace(-4, 4, 5)}
pr_searches = [GridSearchCV(estimator=pr_pipe, param_grid=pr_pipe_param_grid, cv=5, 
                              scoring='neg_mean_squared_error') for i in range(4)]
```


```python
%%capture
models['poly'] = {pair:None for pair in var_pairs}
models['poly']['acc_mpg'] = pr_searches[0].fit(acc.reshape(-1, 1), mpg)
models['poly']['wt_mpg'] = pr_searches[1].fit(wt.reshape(-1, 1), mpg)
models['poly']['hp_mpg'] = pr_searches[2].fit(hp.reshape(-1, 1), mpg)
models['poly']['dp_mpg'] = pr_searches[3].fit(dp.reshape(-1, 1), mpg)
```

### Cubic P-Spline Regression

Thankfully `pygam`'s `GAM` plays nice with `sklearn`'s `GridSearchCV`.


```python
from pygam import GAM, s

gam = GAM(s(0))

ps_param_grid = {'n_splines': np.arange(10, 16), 'spline_order': np.arange(2, 4), 
                 'lam': np.exp(np.random.rand(100, 1) * 6 - 3).flatten()}
ps_searches = [GridSearchCV(estimator=GAM(), param_grid=ps_param_grid, cv=5, scoring='neg_mean_squared_error')
               for i in range(4)]
```

Note that `pygam.GAM.gridsearch` uses generalized cross-validation (GCV).


```python
%%capture
models['p-spline'] = {pair:None for pair in var_pairs}
models['p-spline']['acc_mpg'] = ps_searches[0].fit(acc.reshape(-1, 1), mpg)
models['p-spline']['wt_mpg'] = ps_searches[1].fit(wt.reshape(-1, 1), mpg)
models['p-spline']['hp_mpg'] = ps_searches[2].fit(hp.reshape(-1, 1), mpg)
models['p-spline']['dp_mpg'] = ps_searches[3].fit(dp.reshape(-1, 1), mpg)
```

### Model Comparison

### Optimal models and their test errors


```python
cols = pd.MultiIndex.from_product([['acc_mpg', 'wt_mpg', 'hp_mpg', 'dp_mpg'], ['params', 'cv_mse']], 
                                  names=['var_pair', 'opt_results'])
rows = pd.Index(['local', 'poly', 'p-spline'], name='model_type')
    
models_df = pd.DataFrame(index=rows, columns=cols)
models_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>var_pair</th>
      <th colspan="2" halign="left">acc_mpg</th>
      <th colspan="2" halign="left">wt_mpg</th>
      <th colspan="2" halign="left">hp_mpg</th>
      <th colspan="2" halign="left">dp_mpg</th>
    </tr>
    <tr>
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
for var_pair in models_df.columns.levels[0]:
    for name in models_df.index:
        models_df.loc[name, var_pair] = models[name][var_pair].best_params_, -models[name][var_pair].best_score_
        
models_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>var_pair</th>
      <th colspan="2" halign="left">acc_mpg</th>
      <th colspan="2" halign="left">wt_mpg</th>
      <th colspan="2" halign="left">hp_mpg</th>
      <th colspan="2" halign="left">dp_mpg</th>
    </tr>
    <tr>
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>1.02543</td>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>0.421562</td>
      <td>{'n_neighbors': 4, 'p': 1, 'weights': 'distance'}</td>
      <td>0.371018</td>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>0.406882</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>{'poly__degree': 4, 'ridge__alpha': 0.0001}</td>
      <td>0.969238</td>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.384849</td>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.399221</td>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.403257</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.970488</td>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.391603</td>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.376321</td>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.392898</td>
    </tr>
  </tbody>
</table>
</div>



### Analysis of optimal models


```python
# helper for plotting results

def plot_results(var_name, var):
    fig, axes = plt.subplots(nrows=3, figsize=(10,10))

    for i, model_type in enumerate(models):
        model = models[model_type][var_name + '_mpg']
        plt.subplot(3, 1, i + 1)
        sns.lineplot(x=var, y=model.predict(var.reshape(-1, 1)), color='red', 
                     label=model_type + " regression prediction")
        sns.scatterplot(x=var, y=mpg)
        plt.xlabel('std ' + var_name)
        plt.ylabel('std mpg')
        plt.legend()
        plt.tight_layout()
```

### `mpg` vs `accleration` 

The models seem to have much harder time predicting `mpg` from `acceleration`.


```python
# mse estimates for acceleration models
models_df['acc_mpg']
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
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>1.02543</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>{'poly__degree': 4, 'ridge__alpha': 0.0001}</td>
      <td>0.969238</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.970488</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_results('acc', acc)
```


![png]({{site.baseurl}}/assets/images/ch07_exercise_08_35_0.png)


Observations:

- The local regression model seems to be fitting noise 
- The polynomial and p-spline models are smoother, and less likely to overfit, which is consistent with their lower mse estimates
- The relationship between `acceleration` and `mpg` appears weak

Conclusion:

There is little evidence of a relationship (linear or otherwise) between `acceleration` and `mpg` so we'll omit `acceleration` from the final model

### `mpg` vs. `weight`


```python
# # mse estimates for acceleration models
models_df['wt_mpg']
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
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>0.421562</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.384849</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.391603</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_results('wt', wt)
```


![png]({{site.baseurl}}/assets/images/ch07_exercise_08_39_0.png)


Observations:

- The local regression model again seems to be fitting noise 
- The polynomial and p-spline models are smoother, and less likely to overfit, which is consistent with their lower mse estimates
- The relationship between `weight` and `mpg` appears strong. 
- The p-spline and polynomial regression mses are very similar


```python
models_df[('wt_mpg', 'params')]['poly']
```




    {'poly__degree': 2, 'ridge__alpha': 0.0001}




```python
models_df[('wt_mpg', 'params')]['p-spline']
```




    {'lam': 3.1804238375997853, 'n_splines': 10, 'spline_order': 2}



Conclusion:

Optimal polynomial and p-spline models are both degree 2. Given its flexibility at the lower end of the range of `weight`, we'll select the p-spline for the final model.

### `mpg` vs. `horsepower`


```python
# # mse estimates for acceleration models
models_df['hp_mpg']
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
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>{'n_neighbors': 4, 'p': 1, 'weights': 'distance'}</td>
      <td>0.371018</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.399221</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.376321</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_results('hp', hp)
```


![png]({{site.baseurl}}/assets/images/ch07_exercise_08_46_0.png)


Observations:

- The local regression model again seems to be fitting noise 
- The polynomial and p-spline models are smoother, and less likely to overfit, which is consistent with their lower mse estimates
- The relationship between `horsepower` and `mpg` appears strong. 
- The p-spline and polynomial regression mses are very similar


```python
models_df[('hp_mpg', 'params')]['poly']
```




    {'poly__degree': 2, 'ridge__alpha': 0.0001}




```python
models_df[('hp_mpg', 'params')]['p-spline']
```




    {'lam': 3.1804238375997853, 'n_splines': 10, 'spline_order': 2}



Conclusion:

Optimal polynomial and p-spline models are both degree 2. Given its flexibility at the lower end of the range of `weight`, we'll select the p-spline for the final model.

### `mpg` vs. `displacement`


```python
# # mse estimates for acceleration models
models_df['dp_mpg']
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
      <th>opt_results</th>
      <th>params</th>
      <th>cv_mse</th>
    </tr>
    <tr>
      <th>model_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>local</th>
      <td>{'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}</td>
      <td>0.406882</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>{'poly__degree': 2, 'ridge__alpha': 0.0001}</td>
      <td>0.403257</td>
    </tr>
    <tr>
      <th>p-spline</th>
      <td>{'lam': 3.1804238375997853, 'n_splines': 10, '...</td>
      <td>0.392898</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_results('dp', dp)
```


![png]({{site.baseurl}}/assets/images/ch07_exercise_08_53_0.png)


Observations:

- The local regression model again seems to be fitting noise 
- The polynomial and p-spline models are smoother, and less likely to overfit, which is consistent with their lower mse estimates
- The relationship between `displacement` and `mpg` appears strong. 
- The p-spline and polynomial regression mses are very similar


```python
models_df[('dp_mpg', 'params')]['poly']
```




    {'poly__degree': 2, 'ridge__alpha': 0.0001}




```python
models_df[('dp_mpg', 'params')]['p-spline']
```




    {'lam': 3.1804238375997853, 'n_splines': 10, 'spline_order': 2}



Conclusion:

Optimal polynomial and p-spline models are both degree 2. Given its flexibility at the lower end of the range of `weight`, we'll select the p-spline for the final model.

## GAM for predicting `mpg`

We identified some variables with non-linear relationships to `mpg` above, now we search for linear relationships. We'll then fit a GAM which is a kind of hybrid model - linear on the linear variables, non-linear on the non-linear variables.

$$Y = \beta_0 + \sum_{\text{linear}} \beta_j X_j + \sum_{\text{nonlinear}} \beta_k f_k(X_k)$$

where the nonlinear functions $f_k$ are those found above.

### Find variables with linear relationships to `mpg`


```python
auto.corr()[auto.corr() > 0.5]
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.580541</td>
      <td>0.565209</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.950823</td>
      <td>0.842983</td>
      <td>0.897527</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>NaN</td>
      <td>0.950823</td>
      <td>1.000000</td>
      <td>0.897257</td>
      <td>0.932994</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>NaN</td>
      <td>0.842983</td>
      <td>0.897257</td>
      <td>1.000000</td>
      <td>0.864538</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>NaN</td>
      <td>0.897527</td>
      <td>0.932994</td>
      <td>0.864538</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.580541</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.565209</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Train GAM


```python
gam_df = auto_num.copy()

gam_df = gam_df.drop(columns=['acceleration', 'name'])
num_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'year']
gam_df.loc[: , num_cols] = (gam_df[num_cols] - gam_df[num_cols].mean()) / gam_df[num_cols].std()

gam_df.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>year</th>
      <th>origin_1</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.697747</td>
      <td>1.482053</td>
      <td>1.075915</td>
      <td>0.663285</td>
      <td>0.619748</td>
      <td>-1.623241</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.082115</td>
      <td>1.482053</td>
      <td>1.486832</td>
      <td>1.572585</td>
      <td>0.842258</td>
      <td>-1.623241</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.697747</td>
      <td>1.482053</td>
      <td>1.181033</td>
      <td>1.182885</td>
      <td>0.539692</td>
      <td>-1.623241</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.953992</td>
      <td>1.482053</td>
      <td>1.047246</td>
      <td>1.182885</td>
      <td>0.536160</td>
      <td>-1.623241</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.825870</td>
      <td>1.482053</td>
      <td>1.028134</td>
      <td>0.923085</td>
      <td>0.554997</td>
      <td>-1.623241</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pygam import f

final_gam = GAM(s(1) + s(2) + s(3) + s(4) + s(5) + f(6) + f(7) + f(8))

ps_param_grid = {'n_splines': np.arange(15, 20), 'spline_order': np.arange(2, 3), 
                 'lam': np.exp(np.random.rand(100, 1) * 6 - 3).flatten()}
ps_search = GridSearchCV(estimator=GAM(), param_grid=ps_param_grid, cv=10, scoring='neg_mean_squared_error')
ps_search.fit(gam_df.drop(columns=['mpg']), gam_df['mpg'])
```

    /Users/home/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv=10, error_score='raise-deprecating',
           estimator=GAM(callbacks=['deviance', 'diffs'], distribution='normal',
       fit_intercept=True, link='identity', max_iter=100, terms='auto',
       tol=0.0001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'n_splines': array([15, 16, 17, 18, 19]), 'spline_order': array([2]), 'lam': array([1.22679, 1.71212, ..., 0.19191, 0.68306])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)



### Compare to alternative regression models

We'll compare our GAM to to alternative null, ordinary least squares and polynomial ridge regression models.


```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

X, y = gam_df.drop(columns=['mpg']), gam_df['mpg']

# dummy model that always predicts mean response
dummy = DummyRegressor().fit(X, y)
dummy_cv_mse = np.mean(-cross_val_score(dummy, X, y, scoring='neg_mean_squared_error', cv=10))

# ordinary least squares
ols = LinearRegression().fit(X, y)
ols_cv_mse = np.mean(-cross_val_score(ols, X, y, scoring='neg_mean_squared_error', cv=10))

# optimized polynomial ridge
pr_pipe = Pipeline(steps=[('poly', PolynomialFeatures()), ('ridge', Ridge())])
pr_pipe_param_grid = {'poly__degree': np.arange(1, 5), 'ridge__alpha': np.logspace(-4, 4, 5)}
pr_search = GridSearchCV(estimator=pr_pipe, param_grid=pr_pipe_param_grid, cv=10, 
                              scoring='neg_mean_squared_error')
ridge = pr_search.fit(X, y)
ridge_cv_mse = -ridge.best_score_

gam_cv_mse = -ps_search.best_score_
```

    /Users/home/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
comparison_df = pd.DataFrame({'cv_mse': [dummy_cv_mse, ols_cv_mse, ridge_cv_mse, gam_cv_mse]}, 
                             index=['dummy', 'ols', 'poly ridge', 'gam'])
comparison_df
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
      <th>cv_mse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dummy</th>
      <td>1.092509</td>
    </tr>
    <tr>
      <th>ols</th>
      <td>0.203995</td>
    </tr>
    <tr>
      <th>poly ridge</th>
      <td>0.127831</td>
    </tr>
    <tr>
      <th>gam</th>
      <td>0.201342</td>
    </tr>
  </tbody>
</table>
</div>



The polynomial ridge model has outperformed


```python
ridge.best_params_
```




    {'poly__degree': 3, 'ridge__alpha': 1.0}


{% endkatexmm %}
