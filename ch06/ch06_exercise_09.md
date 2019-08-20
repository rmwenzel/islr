---
layout: page
title: 6. Linear Model Selection and Regularization
---

{% katexmm %}

# Exercise 9: Predicting `Apps` in `College` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-train-test-split" data-toc-modified-id="a.-Train---test-split-2">a. Train - test split</a></span></li><li><span><a href="#b-linear-regression-model" data-toc-modified-id="b.-Linear-regression-model-3">b. Linear regression model</a></span></li><li><span><a href="#c-ridge-regression-model" data-toc-modified-id="c.-Ridge-regression-model-4">c. Ridge regression model</a></span></li><li><span><a href="#d-lasso-regression-model" data-toc-modified-id="d.-Lasso-regression-model-5">d. Lasso regression model</a></span></li><li><span><a href="#e-pcr-model" data-toc-modified-id="e.-PCR-model-6">e. PCR model</a></span></li><li><span><a href="#f-pls-model" data-toc-modified-id="f.-PLS-model-7">f. PLS model</a></span></li><li><span><a href="#g-comments" data-toc-modified-id="g.-Comments-8">g. Comments</a></span><ul class="toc-item"><li><span><a href="#how-accurately-can-we-predict-applications" data-toc-modified-id="How-accurately-can-we-predict-applications?-8.1">How accurately can we predict <code>applications</code>?</a></span></li><li><span><a href="#is-there-much-difference-among-the-test-errors" data-toc-modified-id="Is-there-much-difference-among-the-test-errors-8.2">Is there much difference among the test errors</a></span></li></ul></li></ul></div>

## Preparing the data


```python
import pandas as pd

college = pd.read_csv('../../datasets/College.csv')
college = college.rename({'Unnamed: 0': 'Name'}, axis='columns')
college.head()
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
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene Christian University</td>
      <td>Yes</td>
      <td>1660</td>
      <td>1232</td>
      <td>721</td>
      <td>23</td>
      <td>52</td>
      <td>2885</td>
      <td>537</td>
      <td>7440</td>
      <td>3300</td>
      <td>450</td>
      <td>2200</td>
      <td>70</td>
      <td>78</td>
      <td>18.1</td>
      <td>12</td>
      <td>7041</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelphi University</td>
      <td>Yes</td>
      <td>2186</td>
      <td>1924</td>
      <td>512</td>
      <td>16</td>
      <td>29</td>
      <td>2683</td>
      <td>1227</td>
      <td>12280</td>
      <td>6450</td>
      <td>750</td>
      <td>1500</td>
      <td>29</td>
      <td>30</td>
      <td>12.2</td>
      <td>16</td>
      <td>10527</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adrian College</td>
      <td>Yes</td>
      <td>1428</td>
      <td>1097</td>
      <td>336</td>
      <td>22</td>
      <td>50</td>
      <td>1036</td>
      <td>99</td>
      <td>11250</td>
      <td>3750</td>
      <td>400</td>
      <td>1165</td>
      <td>53</td>
      <td>66</td>
      <td>12.9</td>
      <td>30</td>
      <td>8735</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agnes Scott College</td>
      <td>Yes</td>
      <td>417</td>
      <td>349</td>
      <td>137</td>
      <td>60</td>
      <td>89</td>
      <td>510</td>
      <td>63</td>
      <td>12960</td>
      <td>5450</td>
      <td>450</td>
      <td>875</td>
      <td>92</td>
      <td>97</td>
      <td>7.7</td>
      <td>37</td>
      <td>19016</td>
      <td>59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alaska Pacific University</td>
      <td>Yes</td>
      <td>193</td>
      <td>146</td>
      <td>55</td>
      <td>16</td>
      <td>44</td>
      <td>249</td>
      <td>869</td>
      <td>7560</td>
      <td>4120</td>
      <td>800</td>
      <td>1500</td>
      <td>76</td>
      <td>72</td>
      <td>11.9</td>
      <td>2</td>
      <td>10922</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
college.loc[:, 'Private'] = [0 if entry == 'No' else 1 for entry in college['Private']]
college.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 777 entries, 0 to 776
    Data columns (total 19 columns):
    Name           777 non-null object
    Private        777 non-null int64
    Apps           777 non-null int64
    Accept         777 non-null int64
    Enroll         777 non-null int64
    Top10perc      777 non-null int64
    Top25perc      777 non-null int64
    F.Undergrad    777 non-null int64
    P.Undergrad    777 non-null int64
    Outstate       777 non-null int64
    Room.Board     777 non-null int64
    Books          777 non-null int64
    Personal       777 non-null int64
    PhD            777 non-null int64
    Terminal       777 non-null int64
    S.F.Ratio      777 non-null float64
    perc.alumni    777 non-null int64
    Expend         777 non-null int64
    Grad.Rate      777 non-null int64
    dtypes: float64(1), int64(17), object(1)
    memory usage: 115.4+ KB


## a. Train - test split


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(college.drop(columns=['Apps', 'Name']), 
                                                    college['Apps'])
```

## b. Linear regression model


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linreg = LinearRegression().fit(X_train, y_train)
```


```python
linreg_mse_test = mean_squared_error(y_test, linreg.predict(X_test))
mses_df = pd.DataFrame({'mse_test': linreg_mse_test},
                       index=['linreg'])
mses_df
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
      <th>linreg</th>
      <td>1.869641e+06</td>
    </tr>
  </tbody>
</table>
</div>



## c. Ridge regression model 


```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters = {'alpha': [10**i for i in range(-3, 4)]}
ridge = GridSearchCV(Ridge(), parameters, cv=10, 
                     scoring='neg_mean_squared_error')
```


```python
%%capture

ridge.fit(X_train, y_train)
```


```python
%%capture

ridge_cv_df = pd.DataFrame(ridge.cv_results_)
ridge_cv_df
```


```python
ridge_mse_test = mean_squared_error(y_test, ridge.best_estimator_.predict(X_test))
mses_df = mses_df.append(pd.DataFrame({'mse_test': ridge_mse_test}, index=['ridge']))
mses_df
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
      <th>linreg</th>
      <td>1.869641e+06</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>1.875181e+06</td>
    </tr>
  </tbody>
</table>
</div>



## d. Lasso regression model


```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

parameters = {'alpha': [10**i for i in range(-3, 4)]}
lasso = GridSearchCV(Lasso(), parameters, cv=10, scoring='neg_mean_squared_error')
```


```python
%%capture

lasso.fit(X_train, y_train)
```


```python
%%capture

lasso_cv_df = pd.DataFrame(lasso.cv_results_)
lasso_cv_df
```


```python
lasso_mse_test = mean_squared_error(y_test, lasso.best_estimator_.predict(X_test))
mses_df = mses_df.append(pd.DataFrame({'mse_test': lasso_mse_test}, index=['lasso']))
mses_df
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
      <th>linreg</th>
      <td>1.869641e+06</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>1.875181e+06</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>1.870846e+06</td>
    </tr>
  </tbody>
</table>
</div>



## e. PCR model

`scikit-learn` doesn't have combined PCA and regression so we'll use the top answer to [this CrossValidated question](https://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python)


```python
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
```


```python
n = len(X_train_reduced)
linreg = LinearRegression()

pcr_mses = [-cross_val_score(linreg, np.ones((n,1)), y_train, cv=10, 
                         scoring='neg_mean_squared_error').mean()]   
for i in range(1, college.shape[1] - b1):
    pcr_mses += [-cross_val_score(linreg, X_train_reduced[:, :i], y_train, cv=10, 
                              scoring='neg_mean_squared_error').mean()]
```


```python
np.argmin(pcr_mses)
```




    17



10 fold Cross-validation selects $M = 17$ (full PCR model with no intercept).


```python
pcr = LinearRegression().fit(X_train.iloc[:, :np.argmin(pcr_mses)], y_train)
```

The test error of this model is


```python
pcr_mse_test = mean_squared_error(y_test, pcr.predict(X_test))
mses_df = mses_df.append(pd.DataFrame({'mse_test': pcr_mse_test}, index=['pcr']))
mses_df
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
      <th>linreg</th>
      <td>1.869641e+06</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>1.875181e+06</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>1.870846e+06</td>
    </tr>
    <tr>
      <th>pcr</th>
      <td>1.869641e+06</td>
    </tr>
  </tbody>
</table>
</div>



## f. PLS model


```python
from sklearn.cross_decomposition import PLSRegression

# mse for only constant predictor same as for pcr
pls_mses = pcr_mses[:1]

for i in range(1, college.shape[1] - 1):
    pls_mses += [-cross_val_score(estimator=PLSRegression(n_components = i), 
                                  X=X_train, y=y_train, cv=10, 
                                  scoring='neg_mean_squared_error').mean()]
```


```python
np.argmin(pls_mses)
```




    13



10 fold CV selects $M = 13$


```python
pls = PLSRegression(n_components=13).fit(X_train, y_train)
```


```python
pls_mse_test = mean_squared_error(y_test, pls.predict(X_test))
mses_df = mses_df.append(pd.DataFrame({'mse_test': pls_mse_test}, index=['pls']))
mses_df
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
      <th>linreg</th>
      <td>1.869641e+06</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>1.875181e+06</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>1.870846e+06</td>
    </tr>
    <tr>
      <th>pcr</th>
      <td>1.869641e+06</td>
    </tr>
    <tr>
      <th>pls</th>
      <td>1.862860e+06</td>
    </tr>
  </tbody>
</table>
</div>



## g. Comments

### How accurately can we predict `applications`?

The test mses for each model were $\approx 1.86 \times 10^6$. This corresponds to an (absolute) error of $\sqrt{1.86 \times 10^6} \approx 1387 $. Given the distribution of `applications`


```python
college['Apps'].describe()
```




    count      777.000000
    mean      3001.638353
    std       3870.201484
    min         81.000000
    25%        776.000000
    50%       1558.000000
    75%       3624.000000
    max      48094.000000
    Name: Apps, dtype: float64




```python
import seaborn as sns

% matplotlib inline

sns.distplot(college['Apps'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a212479b0>




![png]({{site.baseurl}}/assets/images/ch06_exercise_9_38_1.png)


The prediction doesn't seem that accurate. Given that distribution is highly concentrated about the mean $\approx 3000$ and the upper quartile is $\approx 3625$, we can say that for most values, the prediction is off by $\geqslant\ \approx 30\%$ of the true value.

### Is there much difference among the test errors

No

{% endkatexmm %}
