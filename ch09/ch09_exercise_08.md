---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}

# Exercise 8: Using SVMs to classify `Purchase` in `OJ` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-train-test-split" data-toc-modified-id="a.-Train-test-split-2">a. Train test split</a></span></li><li><span><a href="#b-linear-svc-with-c-001" data-toc-modified-id="b.-Linear-SVC-with-C=001-3">b. Linear SVC with C = 0.01 </a></span></li><li><span><a href="#c-training-and-test-error-rates-for-linear-svc" data-toc-modified-id="c.-Training-and-test-error-rates-for-linear-SVC-4">c. Training and test error rates for linear SVC</a></span></li><li><span><a href="#d-tuning-cost-parameter-for-linear-svc" data-toc-modified-id="d.-Tuning-cost-parameter-for-linear-SVC-5">d. Tuning cost parameter for linear SVC</a></span></li><li><span><a href="#e-training-and-test-error-rates-for-radial-svc-with-optimized-cost" data-toc-modified-id="e.-Training-and-test-error-rates-for-radial-SVC-with-optimized-cost-6">e. Training and test error rates for radial SVC with optimized cost</a></span></li><li><span><a href="#f-repeat-b-e-for-radial-svc" data-toc-modified-id="f.-Repeat-b.---e.-for-radial-SVC-7">f. Repeat b. - e. for radial SVC</a></span><ul class="toc-item"><li><span><a href="#radial-svc-with-with-c-001" data-toc-modified-id="Radial-SVC-with-with-C-001-7.1">Radial SVC with C = 0.01 </a></span></li><li><span><a href="#training-and-test-error-rates-for-radial-svc" data-toc-modified-id="Training-and-test-error-rates-for-radial-SVC-7.2">Training and test error rates for radial SVC</a></span></li><li><span><a href="#tuning-cost-parameter-for-radial-svc" data-toc-modified-id="Tuning-cost-parameter-for-radial-SVC-7.3">Tuning cost parameter for radial SVC</a></span></li><li><span><a href="#training-and-test-error-rates-for-svc-with-optimized-cost" data-toc-modified-id="Training-and-test-error-rates-for--SVC-with-optimized-cost-7.4">Training and test error rates for SVC with optimized cost</a></span></li></ul></li><li><span><a href="#g-repeat-b-e-for-quadratic-svc" data-toc-modified-id="g.-Repeat-b.---e.-for-quadratic-SVC-8">g. Repeat b. - e. for quadratic SVC</a></span><ul class="toc-item"><li><span><a href="#quadratic-svc-with-c-001" data-toc-modified-id="Quadratic-SVC-with-C-001-8.1">Quadratic SVC with C = 0.01 </a></span></li><li><span><a href="#training-and-test-error-rates-for-radial-svc" data-toc-modified-id="Training-and-test-error-rates-for-radial-SVC-8.2">Training and test error rates for radial SVC</a></span></li><li><span><a href="#tuning-cost-parameter-for-radial-svc" data-toc-modified-id="Tuning-cost-parameter-for-radial-SVC-8.3">Tuning cost parameter for radial SVC</a></span></li><li><span><a href="#training-and-test-error-rates-for-svc-with-optimized-cost" data-toc-modified-id="Training-and-test-error-rates-for--SVC-with-optimized-cost-8.4">Training and test error rates for  SVC with optimized cost</a></span></li></ul></li></ul></div>

```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

## Preparing the data

Information on the dataset can be [found here](https://rdrr.io/cran/ISLR/man/OJ.html)


```python
oj = pd.read_csv('../../datasets/OJ.csv', index_col=0)
oj.reset_index(inplace=True, drop=True)
oj.head()
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
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH</td>
      <td>237</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CH</td>
      <td>239</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CH</td>
      <td>245</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MM</td>
      <td>227</td>
      <td>1</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CH</td>
      <td>228</td>
      <td>7</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
oj.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1070 entries, 0 to 1069
    Data columns (total 18 columns):
    Purchase          1070 non-null object
    WeekofPurchase    1070 non-null int64
    StoreID           1070 non-null int64
    PriceCH           1070 non-null float64
    PriceMM           1070 non-null float64
    DiscCH            1070 non-null float64
    DiscMM            1070 non-null float64
    SpecialCH         1070 non-null int64
    SpecialMM         1070 non-null int64
    LoyalCH           1070 non-null float64
    SalePriceMM       1070 non-null float64
    SalePriceCH       1070 non-null float64
    PriceDiff         1070 non-null float64
    Store7            1070 non-null object
    PctDiscMM         1070 non-null float64
    PctDiscCH         1070 non-null float64
    ListPriceDiff     1070 non-null float64
    STORE             1070 non-null int64
    dtypes: float64(11), int64(5), object(2)
    memory usage: 150.5+ KB



```python
# drop superfluous variables
oj = oj.drop(columns=['STORE', 'Store7'])
oj.columns
```




    Index(['Purchase', 'WeekofPurchase', 'StoreID', 'PriceCH', 'PriceMM', 'DiscCH',
           'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH', 'SalePriceMM',
           'SalePriceCH', 'PriceDiff', 'PctDiscMM', 'PctDiscCH', 'ListPriceDiff'],
          dtype='object')




```python
from sklearn.preprocessing import LabelEncoder

# label encode string variable
purchase_le = LabelEncoder()
purchase_le.fit(oj['Purchase'].values)
oj.loc[ : , 'Purchase'] = purchase_le.transform(oj['Purchase'])
purchase_le.classes_
```




    array(['CH', 'MM'], dtype=object)




```python
from sklearn.preprocessing import MinMaxScaler

# scale all columns to interval [0, 1]
oj_std = (oj - oj.min()) / (oj.max() - oj.min())
oj_std.describe()
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
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
      <td>1070.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.389720</td>
      <td>0.536888</td>
      <td>0.493302</td>
      <td>0.443551</td>
      <td>0.659019</td>
      <td>0.103720</td>
      <td>0.154206</td>
      <td>0.147664</td>
      <td>0.161682</td>
      <td>0.565808</td>
      <td>0.701861</td>
      <td>0.607944</td>
      <td>0.623272</td>
      <td>0.147505</td>
      <td>0.108093</td>
      <td>0.495433</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.487915</td>
      <td>0.305064</td>
      <td>0.384831</td>
      <td>0.254924</td>
      <td>0.223976</td>
      <td>0.234948</td>
      <td>0.267292</td>
      <td>0.354932</td>
      <td>0.368331</td>
      <td>0.307862</td>
      <td>0.229725</td>
      <td>0.204834</td>
      <td>0.207300</td>
      <td>0.253128</td>
      <td>0.246282</td>
      <td>0.244399</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.254902</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.325267</td>
      <td>0.454545</td>
      <td>0.514286</td>
      <td>0.511450</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.318182</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.588235</td>
      <td>0.333333</td>
      <td>0.425000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.600027</td>
      <td>0.818182</td>
      <td>0.671429</td>
      <td>0.687023</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.545455</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.803922</td>
      <td>1.000000</td>
      <td>0.750000</td>
      <td>0.816667</td>
      <td>0.000000</td>
      <td>0.287500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.850916</td>
      <td>0.854545</td>
      <td>0.714286</td>
      <td>0.755725</td>
      <td>0.280282</td>
      <td>0.000000</td>
      <td>0.681818</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## a. Train test split


```python
from sklearn.model_selection import train_test_split

X, Y = oj.drop(columns=['Purchase']), oj['Purchase']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=800)
```

## b. Linear SVC with C = 0.01


```python
from sklearn.svm import SVC

linear_svc = SVC(kernel='linear', C=0.01)
linear_svc.fit(X_train, Y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



## c. Training and test error rates for linear SVC


```python
from sklearn.metrics import accuracy_score

linear_svc_train_error = accuracy_score(linear_svc.predict(X_train), Y_train)
linear_svc_test_error = accuracy_score(linear_svc.predict(X_test), Y_test)

f'The linear SVC train error is {linear_svc_train_error}'
```




    'The linear SVC train error is 0.75125'




```python
f'The linear SVC test error is {linear_svc_test_error}'
```




    'The linear SVC test error is 0.7333333333333333'



## d. Tuning cost parameter for linear SVC


```python
from sklearn.model_selection import GridSearchCV

param = {'C': [0.01, 0.1, 1, 10]}
linear_svc = SVC(kernel='linear')
linear_svc_search = GridSearchCV(estimator=linear_svc,
                                 param_grid=param,
                                 cv=8,
                                 scoring='accuracy')
%timeit -n1 -r1 linear_svc_search.fit(X_train, Y_train)
```

    3.33 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


    /anaconda3/envs/islr/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
linear_svc_search.best_params_
```




    {'C': 1}




```python
linear_svc_search.best_score_
```




    0.82625



## e. Training and test error rates for radial SVC with optimized cost


```python
linear_svc = SVC(kernel='linear', C=linear_svc_search.best_params_['C'])
linear_svc.fit(X_train, Y_train)
```




    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
from sklearn.metrics import accuracy_score

linear_svc_train_error = accuracy_score(linear_svc.predict(X_train), Y_train)
linear_svc_test_error = accuracy_score(linear_svc.predict(X_test), Y_test)

f'The linear SVC train error is {linear_svc_train_error}'
```




    'The linear SVC train error is 0.83125'




```python
f'The linear SVC test error is {linear_svc_test_error}'
```




    'The linear SVC test error is 0.8555555555555555'



## f. Repeat b. - e. for radial SVC

### Radial SVC with C = 0.01


```python
from sklearn.svm import SVC

radial_svc = SVC(kernel='rbf', C=0.01)
radial_svc.fit(X_train, Y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



### Training and test error rates for radial SVC


```python
from sklearn.metrics import accuracy_score

radial_svc_train_error = accuracy_score(radial_svc.predict(X_train), Y_train)
radial_svc_test_error = accuracy_score(radial_svc.predict(X_test), Y_test)

f'The radial SVC train error is {radial_svc_train_error}'
```




    'The radial SVC train error is 0.6175'




```python
f'The radial SVC test error is {radial_svc_test_error}'
```




    'The radial SVC test error is 0.5888888888888889'



### Tuning cost parameter for radial SVC


```python
from sklearn.model_selection import GridSearchCV

param = {'C': [0.01, 0.1, 1, 10]}
radial_svc = SVC(kernel='rbf')
radial_svc_search = GridSearchCV(estimator=radial_svc,
                                 param_grid=param,
                                 cv=8,
                                 scoring='accuracy')
%timeit -n1 -r1 radial_svc_search.fit(X_train, Y_train)
```

    1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


    /anaconda3/envs/islr/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
radial_svc_search.best_params_
```




    {'C': 10}




```python
radial_svc_search.best_score_
```




    0.795



### Training and test error rates for  SVC with optimized cost


```python
radial_svc = SVC(kernel='rbf', C=radial_svc_search.best_params_['C'])
radial_svc.fit(X_train, Y_train)
```




    SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
radial_svc_train_error = accuracy_score(radial_svc.predict(X_train), Y_train)
radial_svc_test_error = accuracy_score(radial_svc.predict(X_test), Y_test)

f'The radial SVC train error is {radial_svc_train_error}'
```




    'The radial SVC train error is 0.85375'




```python
f'The radial SVC test error is {radial_svc_test_error}'
```




    'The radial SVC test error is 0.8074074074074075'



## g. Repeat b. - e. for quadratic SVC

### Quadratic SVC with C = 0.01


```python
from sklearn.svm import SVC

quad_svc = SVC(kernel='poly', degree=2, C=0.01)
quad_svc.fit(X_train, Y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=2, gamma='auto_deprecated',
      kernel='poly', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



### Training and test error rates for radial SVC


```python
from sklearn.metrics import accuracy_score

quad_svc_train_error = accuracy_score(quad_svc.predict(X_train), Y_train)
quad_svc_test_error = accuracy_score(quad_svc.predict(X_test), Y_test)

f'The quadratic SVC train error is {quad_svc_train_error}'
```




    'The quadratic SVC train error is 0.83625'




```python
f'The quadratic SVC test error is {quad_svc_test_error}'
```




    'The quadratic SVC test error is 0.8555555555555555'



### Tuning cost parameter for radial SVC


```python
from sklearn.model_selection import GridSearchCV

param = {'C': [0.01, 0.1, 1, 10]}
quad_svc = SVC(kernel='poly', degree=2)
quad_svc_search = GridSearchCV(estimator=quad_svc,
                                 param_grid=param,
                                 cv=8,
                                 scoring='accuracy')
%timeit -n1 -r1 quad_svc_search.fit(X_train, Y_train)
```


```python
quad_svc_search.best_params_
```


```python
quad_svc_search.best_score_
```

### Training and test error rates for  SVC with optimized cost


```python
quad_svc = SVC(kernel='', C=quad_svc_search.best_params_['C'])
quad_svc.fit(X_train, Y_train)
```


```python
quad_svc_train_error = accuracy_score(quad_svc.predict(X_train), Y_train)
quad_svc_test_error = accuracy_score(quad_svc.predict(X_test), Y_test)

f'The quadratic SVC train error is {quad_svc_train_error}'
```


```python
f'The quadratic SVC test error is {quad_svc_test_error}'
```

{% endkatexmm %}