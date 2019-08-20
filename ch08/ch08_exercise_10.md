---
layout: page
title: 8. Tree-based Methods
---

{% katexmm %}

# Exercise 10: Boosting to predict `Salary` in `Hitters` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-remove-observations-with-missing-Salary-and-log-transform-salary" data-toc-modified-id="a.-Remove-observations-with-missing-Salary-and-log-transform-Salary-2">a. Remove observations with missing <code>Salary</code> and log-transform <code>Salary</code></a></span></li><li><span><a href="#b-train-test-split" data-toc-modified-id="b.-Train-test-split-3">b. Train test split</a></span></li><li><span><a href="#c-boosting-on-training-set" data-toc-modified-id="c.-Boosting-on-training-set-4">c. Boosting on training set</a></span></li><li><span><a href="#d-plotting-train-error-cv-test-error-estimate-and-test-error" data-toc-modified-id="d.-Plotting-train-error,-cv-test-error-estimate,-and-test-error-5">d. Plotting train error, cv test error estimate, and test error</a></span></li><li><span><a href="#e-comparing-errors-with-ols-lasso,-and-Ridge-Regression-models" data-toc-modified-id="e.--Comparing-errors-with-OLS,-Lasso,-and-Ridge-Regression-models-6">e.  Comparing errors with OLS, Lasso, and Ridge Regression models</a></span></li><li><span><a href="#f.-Which-variables-are-the-most-important-in-the-boosted-tree-model?" data-toc-modified-id="f.-Which-variables-are-the-most-important-in-the-boosted-tree-model?-7">f. Which variables are the most important in the boosted tree model?</a></span></li><li><span><a href="#Bagged-tree" data-toc-modified-id="Bagged-tree-8">Bagged tree</a></span></li></ul></div>

## Preparing the data


```python
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
hitters = pd.read_csv('../../datasets/Hitters.csv')
hitters = hitters.rename(columns={'Unnamed: 0': 'Name'})
hitters.head()
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

<div style="overflow-x:auto;">
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>Name</th>
        <th>AtBat</th>
        <th>Hits</th>
        <th>HmRun</th>
        <th>Runs</th>
        <th>RBI</th>
        <th>Walks</th>
        <th>Years</th>
        <th>CAtBat</th>
        <th>CHits</th>
        <th>...</th>
        <th>CRuns</th>
        <th>CRBI</th>
        <th>CWalks</th>
        <th>League</th>
        <th>Division</th>
        <th>PutOuts</th>
        <th>Assists</th>
        <th>Errors</th>
        <th>Salary</th>
        <th>NewLeague</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>-Andy Allanson</td>
        <td>293</td>
        <td>66</td>
        <td>1</td>
        <td>30</td>
        <td>29</td>
        <td>14</td>
        <td>1</td>
        <td>293</td>
        <td>66</td>
        <td>...</td>
        <td>30</td>
        <td>29</td>
        <td>14</td>
        <td>A</td>
        <td>E</td>
        <td>446</td>
        <td>33</td>
        <td>20</td>
        <td>NaN</td>
        <td>A</td>
      </tr>
      <tr>
        <th>1</th>
        <td>-Alan Ashby</td>
        <td>315</td>
        <td>81</td>
        <td>7</td>
        <td>24</td>
        <td>38</td>
        <td>39</td>
        <td>14</td>
        <td>3449</td>
        <td>835</td>
        <td>...</td>
        <td>321</td>
        <td>414</td>
        <td>375</td>
        <td>N</td>
        <td>W</td>
        <td>632</td>
        <td>43</td>
        <td>10</td>
        <td>475.0</td>
        <td>N</td>
      </tr>
      <tr>
        <th>2</th>
        <td>-Alvin Davis</td>
        <td>479</td>
        <td>130</td>
        <td>18</td>
        <td>66</td>
        <td>72</td>
        <td>76</td>
        <td>3</td>
        <td>1624</td>
        <td>457</td>
        <td>...</td>
        <td>224</td>
        <td>266</td>
        <td>263</td>
        <td>A</td>
        <td>W</td>
        <td>880</td>
        <td>82</td>
        <td>14</td>
        <td>480.0</td>
        <td>A</td>
      </tr>
      <tr>
        <th>3</th>
        <td>-Andre Dawson</td>
        <td>496</td>
        <td>141</td>
        <td>20</td>
        <td>65</td>
        <td>78</td>
        <td>37</td>
        <td>11</td>
        <td>5628</td>
        <td>1575</td>
        <td>...</td>
        <td>828</td>
        <td>838</td>
        <td>354</td>
        <td>N</td>
        <td>E</td>
        <td>200</td>
        <td>11</td>
        <td>3</td>
        <td>500.0</td>
        <td>N</td>
      </tr>
      <tr>
        <th>4</th>
        <td>-Andres Galarraga</td>
        <td>321</td>
        <td>87</td>
        <td>10</td>
        <td>39</td>
        <td>42</td>
        <td>30</td>
        <td>2</td>
        <td>396</td>
        <td>101</td>
        <td>...</td>
        <td>48</td>
        <td>46</td>
        <td>33</td>
        <td>N</td>
        <td>E</td>
        <td>805</td>
        <td>40</td>
        <td>4</td>
        <td>91.5</td>
        <td>N</td>
      </tr>
    </tbody>
  </table>
</div>

<p>5 rows × 21 columns</p>
</div>




```python
hitters.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 322 entries, 0 to 321
    Data columns (total 21 columns):
    Name         322 non-null object
    AtBat        322 non-null int64
    Hits         322 non-null int64
    HmRun        322 non-null int64
    Runs         322 non-null int64
    RBI          322 non-null int64
    Walks        322 non-null int64
    Years        322 non-null int64
    CAtBat       322 non-null int64
    CHits        322 non-null int64
    CHmRun       322 non-null int64
    CRuns        322 non-null int64
    CRBI         322 non-null int64
    CWalks       322 non-null int64
    League       322 non-null object
    Division     322 non-null object
    PutOuts      322 non-null int64
    Assists      322 non-null int64
    Errors       322 non-null int64
    Salary       263 non-null float64
    NewLeague    322 non-null object
    dtypes: float64(1), int64(16), object(4)
    memory usage: 52.9+ KB



```python
hitters = pd.concat([hitters['Name'],
                     pd.get_dummies(hitters.drop(columns=['Name']), drop_first=True)], axis=1)
hitters.head()
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
      <th>Name</th>
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>...</th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>Salary</th>
      <th>League_N</th>
      <th>Division_W</th>
      <th>NewLeague_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-Andy Allanson</td>
      <td>293</td>
      <td>66</td>
      <td>1</td>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>1</td>
      <td>293</td>
      <td>66</td>
      <td>...</td>
      <td>30</td>
      <td>29</td>
      <td>14</td>
      <td>446</td>
      <td>33</td>
      <td>20</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-Alan Ashby</td>
      <td>315</td>
      <td>81</td>
      <td>7</td>
      <td>24</td>
      <td>38</td>
      <td>39</td>
      <td>14</td>
      <td>3449</td>
      <td>835</td>
      <td>...</td>
      <td>321</td>
      <td>414</td>
      <td>375</td>
      <td>632</td>
      <td>43</td>
      <td>10</td>
      <td>475.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-Alvin Davis</td>
      <td>479</td>
      <td>130</td>
      <td>18</td>
      <td>66</td>
      <td>72</td>
      <td>76</td>
      <td>3</td>
      <td>1624</td>
      <td>457</td>
      <td>...</td>
      <td>224</td>
      <td>266</td>
      <td>263</td>
      <td>880</td>
      <td>82</td>
      <td>14</td>
      <td>480.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-Andre Dawson</td>
      <td>496</td>
      <td>141</td>
      <td>20</td>
      <td>65</td>
      <td>78</td>
      <td>37</td>
      <td>11</td>
      <td>5628</td>
      <td>1575</td>
      <td>...</td>
      <td>828</td>
      <td>838</td>
      <td>354</td>
      <td>200</td>
      <td>11</td>
      <td>3</td>
      <td>500.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-Andres Galarraga</td>
      <td>321</td>
      <td>87</td>
      <td>10</td>
      <td>39</td>
      <td>42</td>
      <td>30</td>
      <td>2</td>
      <td>396</td>
      <td>101</td>
      <td>...</td>
      <td>48</td>
      <td>46</td>
      <td>33</td>
      <td>805</td>
      <td>40</td>
      <td>4</td>
      <td>91.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## a. Remove observations with missing `Salary` and log-transform `Salary`


```python
hitters = hitters[hitters['Salary'].notna()]
hitters.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 263 entries, 1 to 321
    Data columns (total 21 columns):
    Name           263 non-null object
    AtBat          263 non-null int64
    Hits           263 non-null int64
    HmRun          263 non-null int64
    Runs           263 non-null int64
    RBI            263 non-null int64
    Walks          263 non-null int64
    Years          263 non-null int64
    CAtBat         263 non-null int64
    CHits          263 non-null int64
    CHmRun         263 non-null int64
    CRuns          263 non-null int64
    CRBI           263 non-null int64
    CWalks         263 non-null int64
    PutOuts        263 non-null int64
    Assists        263 non-null int64
    Errors         263 non-null int64
    Salary         263 non-null float64
    League_N       263 non-null uint8
    Division_W     263 non-null uint8
    NewLeague_N    263 non-null uint8
    dtypes: float64(1), int64(16), object(1), uint8(3)
    memory usage: 39.8+ KB



```python
hitters.loc[:, 'Salary'] = np.log(hitters['Salary'])
```

## b. Train test split


```python
from sklearn.model_selection import train_test_split

X, y = hitters.drop(columns=['Name', 'Salary']), hitters['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=200)
```

## c. Boosting on training set

We used this [article](https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/) to suggest customary values of the boosting parameter $\lambda$ (the "learning rate")


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

params = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
boost_tree = GradientBoostingRegressor(n_estimators=1000)
boost_tree_search = GridSearchCV(estimator=boost_tree,
                                 param_grid=params,
                                 cv=10,
                                 scoring='neg_mean_squared_error')
%timeit -n1 -r1 boost_tree_search.fit(X_train, y_train)
```

    19.3 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
boost_tree_search.best_params_
```




    {'learning_rate': 0.01}




```python
-boost_tree_search.best_score_
```




    0.0066770413811597165



## d. Plotting train error, cv test error estimate, and test error


```python
from sklearn.metrics import mean_squared_error

boost_tree_search_df = pd.DataFrame(boost_tree_search.cv_results_)

plt.figure(figsize=(10, 5))
x = boost_tree_search_df['param_learning_rate']
y = -boost_tree_search_df['mean_train_score']
plt.plot(x, y, '-b')

y = -boost_tree_search_df['mean_test_score']
plt.plot(x, y, '--r', label='mean_cv_test_estimate')

y = []
for rate in params['learning_rate']:
    boost_tree = GradientBoostingRegressor(n_estimators=1000, learning_rate=rate).fit(X_train, y_train)
    y += [mean_squared_error(y_test, boost_tree.predict(X_test))]
plt.plot(x, y, ':g', label='mean_test_error')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a180ac550>




![png]({{site.baseurl}}/assets/images/ch08_exercise_10_17_1.png)


## e.  Comparing errors with OLS, Lasso, and Ridge Regression models


```python
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score

# df for comparison results
comp_df = pd.DataFrame(index=['OLS', 'Ridge', 'Boosted Tree'], columns=['mse_train', 'cv_mse_test', 'mse_test'])

# OLS linear regression errors
linreg = LinearRegression()
linreg.fit(X_train, y_train)
comp_df.at['OLS', 'mse_train'] = mean_squared_error(linreg.predict(X_train), y_train)
comp_df.at['OLS', 'cv_mse_test'] = np.mean(-cross_val_score(linreg, X_train, y_train, cv=10, 
                                                            scoring='neg_mean_squared_error'))
comp_df.at['OLS', 'mse_test'] = mean_squared_error(linreg.predict(X_test), y_test)

# Lasso Regression errors
params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
lasso = Lasso(max_iter=10000)
lasso_search = GridSearchCV(lasso, param_grid=params, cv=10, scoring='neg_mean_squared_error').fit(X_train, y_train)
lasso_best = lasso_search.best_estimator_
comp_df.at['Lasso', 'mse_train'] = mean_squared_error(lasso_best.predict(X_train), y_train)
comp_df.at['Lasso', 'cv_mse_test'] = np.mean(-cross_val_score(lasso_best, X_train, y_train, cv=10, 
                                                            scoring='neg_mean_squared_error'))
comp_df.at['Lasso', 'mse_test'] = mean_squared_error(lasso_best.predict(X_test), y_test)

# Ridge Regression errors
ridge = Ridge(max_iter=10000)
ridge_search = GridSearchCV(ridge, param_grid=params, cv=10, scoring='neg_mean_squared_error').fit(X_train, y_train)
ridge_best = ridge_search.best_estimator_
comp_df.at['Ridge', 'mse_train'] = mean_squared_error(ridge_best.predict(X_train), y_train)
comp_df.at['Ridge', 'cv_mse_test'] = np.mean(-cross_val_score(ridge_best, X_train, y_train, cv=10, 
                                                            scoring='neg_mean_squared_error'))
comp_df.at['Ridge', 'mse_test'] = mean_squared_error(ridge_best.predict(X_test), y_test)

# Boosted Tree errors
boost_tree_best = boost_tree_search.best_estimator_
comp_df.at['Boosted Tree', 'mse_train'] = mean_squared_error(boost_tree_best.predict(X_train), y_train)
comp_df.at['Boosted Tree', 'cv_mse_test'] = np.mean(-cross_val_score(boost_tree_best, X_train, y_train, cv=10, 
                                                            scoring='neg_mean_squared_error'))
comp_df.at['Boosted Tree', 'mse_test'] = mean_squared_error(boost_tree_best.predict(X_test), y_test)
```

Here's the model comparison in order of increasing training error


```python
comp_df.sort_values(by='mse_train')
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
      <th>mse_train</th>
      <th>cv_mse_test</th>
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Boosted Tree</th>
      <td>0.00037481</td>
      <td>0.00662225</td>
      <td>0.00473355</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>0.0113734</td>
      <td>0.0148324</td>
      <td>0.0100858</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.0114361</td>
      <td>0.014843</td>
      <td>0.00972244</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.0114704</td>
      <td>0.0146984</td>
      <td>0.00965864</td>
    </tr>
  </tbody>
</table>
</div>



And cv test error estimate


```python
comp_df.sort_values(by='cv_mse_test')
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
      <th>mse_train</th>
      <th>cv_mse_test</th>
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Boosted Tree</th>
      <td>0.00037481</td>
      <td>0.00662225</td>
      <td>0.00473355</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.0114704</td>
      <td>0.0146984</td>
      <td>0.00965864</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>0.0113734</td>
      <td>0.0148324</td>
      <td>0.0100858</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.0114361</td>
      <td>0.014843</td>
      <td>0.00972244</td>
    </tr>
  </tbody>
</table>
</div>



And finally test error


```python
comp_df.sort_values(by='mse_test')
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
      <th>mse_train</th>
      <th>cv_mse_test</th>
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Boosted Tree</th>
      <td>0.00037481</td>
      <td>0.00662225</td>
      <td>0.00473355</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.0114704</td>
      <td>0.0146984</td>
      <td>0.00965864</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.0114361</td>
      <td>0.014843</td>
      <td>0.00972244</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>0.0113734</td>
      <td>0.0148324</td>
      <td>0.0100858</td>
    </tr>
  </tbody>
</table>
</div>



The boosted tree is a clear winner in all 3 cases

## f. Which variables are the most important in the boosted tree model?


```python
feat_imp_df = pd.DataFrame({'Feature Importance': boost_tree_best.feature_importances_}, 
                        index=X.columns)
feat_imp_df.sort_values(by='Feature Importance', ascending=False)
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
      <th>Feature Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAtBat</th>
      <td>0.441317</td>
    </tr>
    <tr>
      <th>CHits</th>
      <td>0.152543</td>
    </tr>
    <tr>
      <th>CRuns</th>
      <td>0.069480</td>
    </tr>
    <tr>
      <th>Hits</th>
      <td>0.049296</td>
    </tr>
    <tr>
      <th>CWalks</th>
      <td>0.048265</td>
    </tr>
    <tr>
      <th>CRBI</th>
      <td>0.035715</td>
    </tr>
    <tr>
      <th>CHmRun</th>
      <td>0.025762</td>
    </tr>
    <tr>
      <th>AtBat</th>
      <td>0.025063</td>
    </tr>
    <tr>
      <th>Walks</th>
      <td>0.024959</td>
    </tr>
    <tr>
      <th>Years</th>
      <td>0.024882</td>
    </tr>
    <tr>
      <th>RBI</th>
      <td>0.024426</td>
    </tr>
    <tr>
      <th>Runs</th>
      <td>0.022669</td>
    </tr>
    <tr>
      <th>Errors</th>
      <td>0.018002</td>
    </tr>
    <tr>
      <th>HmRun</th>
      <td>0.015309</td>
    </tr>
    <tr>
      <th>PutOuts</th>
      <td>0.014301</td>
    </tr>
    <tr>
      <th>Assists</th>
      <td>0.005260</td>
    </tr>
    <tr>
      <th>League_N</th>
      <td>0.001742</td>
    </tr>
    <tr>
      <th>NewLeague_N</th>
      <td>0.000780</td>
    </tr>
    <tr>
      <th>Division_W</th>
      <td>0.000229</td>
    </tr>
  </tbody>
</table>
</div>



## Bagged tree


```python
from sklearn.ensemble import BaggingRegressor

# Bagged Tree randomized CV search for rough hyperparameter tuning
params = {'n_estimators': np.arange(1, 100, 10)}
bag_tree = BaggingRegressor()
bag_tree_search = GridSearchCV(estimator=bag_tree, param_grid=params, cv=10,
                               scoring='neg_mean_squared_error')
bag_tree_best = bag_tree_search.fit(X_train, y_train).best_estimator_

%timeit -n1 -r1 bag_tree_search.fit(X_train, y_train)
```

    7.55 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
bag_tree_search.best_params_
```




    {'n_estimators': 21}




```python
bag_tree_best = bag_tree_search.best_estimator_

# Bagged Tree errors
comp_df.at['Bagged Tree', 'mse_train'] = mean_squared_error(bag_tree_best.predict(X_train), y_train)
comp_df.at['Bagged Tree', 'cv_mse_test'] = np.mean(-cross_val_score(bag_tree_best, X_train, y_train, cv=10, 
                                                            scoring='neg_mean_squared_error'))
comp_df.at['Bagged Tree', 'mse_test'] = mean_squared_error(bag_tree_best.predict(X_test), y_test)

comp_df.sort_values(by='mse_test')
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
      <th>mse_train</th>
      <th>cv_mse_test</th>
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bagged Tree</th>
      <td>0.00100862</td>
      <td>0.0064751</td>
      <td>0.0045024</td>
    </tr>
    <tr>
      <th>Boosted Tree</th>
      <td>0.00037481</td>
      <td>0.00662225</td>
      <td>0.00473355</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>0.0114704</td>
      <td>0.0146984</td>
      <td>0.00965864</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>0.0114361</td>
      <td>0.014843</td>
      <td>0.00972244</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>0.0113734</td>
      <td>0.0148324</td>
      <td>0.0100858</td>
    </tr>
  </tbody>
</table>
</div>

{% endkatexmm %}
