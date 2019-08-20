
# Nonlinear models for predicting `nox` using `dis` in `Boston` dataset

## Preparing the data

Information on the dataset can be [found here](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style(style='whitegrid')
```


```python
boston = pd.read_csv('../../datasets/Boston.csv', index_col=0)
boston = boston.reset_index(drop=True)
boston.head()
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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
    RangeIndex: 506 entries, 0 to 505
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
    memory usage: 55.4 KB


## a. Cubic regression


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=3)
linreg = LinearRegression()

X, y = boston['dis'].values, boston['nox'].values
X, y = (X - X.mean())/X.std(), (y - y.mean())/y.std()
linreg.fit(poly.fit_transform(X.reshape(-1, 1)), y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
fig = plt.figure(figsize=(10, 7))
sns.lineplot(x=X, y=linreg.predict(poly.fit_transform(X.reshape(-1,1))), color='red')
sns.scatterplot(x=X, y=y, color='grey', alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2329f320>




![png](ch07_exercise_09_files/ch07_exercise_09_9_1.png)


## b. Polynomial regression for degree $d = 1,\dots, 10$


```python
regs = {d:None for d in range(1, 11)}

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12,20))

for (i, d) in enumerate(regs):
    poly = PolynomialFeatures(degree=d)
    linreg = LinearRegression().fit(poly.fit_transform(X.reshape(-1, 1)), y)
    plt.subplot(5, 2, i + 1)
    sns.lineplot(x=X, y=linreg.predict(poly.fit_transform(X.reshape(-1,1))), color='red')
    sns.scatterplot(x=X, y=y, color='grey', alpha=0.5)
    plt.xlabel('degree ' + strd.)
    fig.tight_layout()
```


![png](ch07_exercise_09_files/ch07_exercise_09_11_0.png)


## c. Optimizing the degree of the polynomial regression model

 We'll estimate the root mean squared error using 10-fold cross validation


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

poly_reg_pipe = Pipeline([('poly', PolynomialFeatures()), ('linreg', LinearRegression())])
poly_reg_params = {'poly__degree': np.arange(1, 10)}
poly_reg_search = GridSearchCV(estimator=poly_reg_pipe, param_grid=poly_reg_params, cv=10,
                               scoring='neg_mean_squared_error')
poly_reg_search.fit(X.reshape(-1, 1), y)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv=10, error_score='raise-deprecating',
           estimator=Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('linreg', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False))]),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'poly__degree': array([1, 2, 3, 4, 5, 6, 7, 8, 9])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
poly_reg_search.best_params_
```




    {'poly__degree': 3}




```python
np.sqrt(-poly_reg_search.best_score_)
```




    0.5609950540561572



## d.  Cubic spline regression with $4$ degrees of freedom

For this we'll use [`patsy`](https://patsy.readthedocs.io/en/latest/) Python module. This [blog post](https://www.analyticsvidhya.com/blog/2018/03/introduction-regression-splines-python-codes/) was helpful. 

We're using [`patsy.bs`](https://patsy.readthedocs.io/en/latest/API-reference.html#spline-regression) default choice for the knots (equally spaced quantiles), and default degree (3).


```python
from patsy import dmatrix

X_tr = dmatrix("bs(x, df=4)", {'x': X})
spline_reg = LinearRegression()
spline_reg.fit(X_tr, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
fig = plt.figure(figsize=(10, 7))
sns.lineplot(x=X, y=spline_reg.predict(X_tr), color='red')
sns.scatterplot(x=X, y=y, color='grey', alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2338a240>




![png](ch07_exercise_09_files/ch07_exercise_09_20_1.png)



```python
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(spline_reg.predict(X_tr), y))
```




    0.5324989652537614



## e. Cubic spline regression with degrees of freedom $d = 4, \dots 11$


```python
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))

for (i, d) in enumerate(range(4, 12)):
    
    X_tr = dmatrix("bs(x, df=" + strd. + ")", {'x': X})
    spline_reg = LinearRegression()
    spline_reg.fit(X_tr, y)
    rmse = round(np.sqrt(mean_squared_error(spline_reg.predict(X_tr), y)), 4)
    
    plt.subplot(2, 4, i + 1)
    sns.lineplot(x=X, y=spline_reg.predict(X_tr), color='red')
    plt.plot([], [], label='rmse = ' + str(rmse))
    sns.scatterplot(x=X, y=y, color='grey', alpha=0.5)
    plt.xlabel(strd. + ' degree of freedom')
    fig.tight_layout()
```


![png](ch07_exercise_09_files/ch07_exercise_09_23_0.png)


## f. Optimizing the degrees of freedom of the cubic spline model


```python
from sklearn.model_selection import cross_val_score

spline_cv_rmses = {d:None for d in range(4, 12)}

for d in spline_cv_rmses:
    X_tr = dmatrix("bs(x, df=" + strd. + ")", {'x': X})
    linreg = LinearRegression()
    cv_rmse = np.sqrt(-np.mean(cross_val_score(linreg, 
                                               X_tr, y, cv=10, 
                                               scoring='neg_mean_squared_error')))
    spline_cv_rmses[d] = cv_rmse

spline_cvs = pd.DataFrame({'dofs': list(spline_cv_rmses.keys()), 
                           'cv_rmses': list(spline_cv_rmses.values())})
spline_cvs
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
      <th>dofs</th>
      <th>cv_rmses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.632125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.589646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0.592387</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>0.608314</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>0.629096</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>0.621942</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10</td>
      <td>0.640462</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>0.661658</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(10, 7))
sns.lineplot(x=spline_cvs['dofs'], y = spline_cvs['cv_rmses'], color='red')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a23c056a0>




![png](ch07_exercise_09_files/ch07_exercise_09_26_1.png)


By cross-validation, (for this choice of knots), 5 is the best number of degrees of freedom
