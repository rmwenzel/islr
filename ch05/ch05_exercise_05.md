---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Exercise 5: Estimate the test error of a logistic regression model

<div class="toc"><ul class="toc-item"><li><span><a href="#prepare-the-data" data-toc-modified-id="Prepare-the-data-1">Prepare the data</a></span><ul class="toc-item"><li><span><a href="#add-constant" data-toc-modified-id="Add-constant-1.1">Add constant</a></span></li><li><span><a href="#convert-default-student-to-numerical-values" data-toc-modified-id="Convert-default,-student-to-numerical-values-1.2">Convert <code>default</code>, <code>student</code> to numerical values</a></span></li></ul></li><li><span><a href="#a" data-toc-modified-id="a.-2">a.</a></span></li><li><span><a href="#b" data-toc-modified-id="b.-3">b.</a></span><ul class="toc-item"><li><span><a href="#i" data-toc-modified-id="i.-3.1">i.</a></span></li><li><span><a href="#ii" data-toc-modified-id="ii.-3.2">ii.</a></span></li><li><span><a href="#iii" data-toc-modified-id="iii.-3.3">iii.</a></span></li><li><span><a href="#iv" data-toc-modified-id="iv.-3.4">iv.</a></span></li><li><span><a href="#c" data-toc-modified-id="c.-3.5">c.</a></span></li><li><span><a href="#d" data-toc-modified-id="d.-3.6">d.</a></span></li></ul></li></ul></div>

## Prepare the data


```python
import pandas as pd

default = pd.read_csv("../../datasets/Default.csv", index_col=0)
```


```python
default.head()
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
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>




```python
default.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10000 entries, 1 to 10000
    Data columns (total 4 columns):
    default    10000 non-null object
    student    10000 non-null object
    balance    10000 non-null float64
    income     10000 non-null float64
    dtypes: float64(2), object(2)
    memory usage: 390.6+ KB


### Add constant


```python
default['const'] = 1
columns = list(default.columns)
columns.remove('const')
default = default[['const'] + columns]
```


```python
default.head()
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
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>



### Convert `default`, `student` to numerical values

Let Yes=1, No=0


```python
default['default'] = [int(value=='Yes') for value in default['default']]
default['student'] = [int(value=='Yes') for value in default['student']]
```


```python
default.head()
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
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>




```python
default.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10000 entries, 1 to 10000
    Data columns (total 5 columns):
    const      10000 non-null int64
    default    10000 non-null int64
    student    10000 non-null int64
    balance    10000 non-null float64
    income     10000 non-null float64
    dtypes: float64(2), int64(3)
    memory usage: 468.8 KB


## a.


```python
from sklearn.linear_model import LogisticRegression

X, y = default[['balance', 'income', 'const']], default['default']
logit = LogisticRegression(solver='lbfgs').fit(X, y)
```

## b.

### i.


```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

### ii.


```python
logit_train_0 = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
```

### iii.


```python
y_pred_0 = logit_train_0.predict(X_test)
```

### iv.


```python
from sklearn.metrics import accuracy_score

errs_0 = {}
errs_0['err0'] = 1 - accuracy_score(y_test, y_pred_0)
errs_0['err0']
```




    0.038799999999999946



### c.


```python
# train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logit_train_1 = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
y_pred_1 = logit_train_1.predict(X_test)

# get error
errs_0['err1'] = 1 - accuracy_score(y_test, y_pred_1)
errs_0['err1']
```




    0.03159999999999996




```python
# train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
logit_train_2 = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
y_pred_2 = logit_train_2.predict(X_test)

# get error
errs_0['err2'] = 1 - accuracy_score(y_test, y_pred_2)
errs_0['err2']
```




    0.027200000000000002



These results are close to each other? Their average is


```python
sum(errs_0.values())/len(errs_0)
```




    0.0325333333333333



### d.


```python
X, y = default[['balance', 'income', 'student', 'const']], default['default']
```


```python
errs_1 = {}

for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
    errs_1['err' + stri.] = 1 - accuracy_score(y_test, 
                                         LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test))
```


```python
errs_1
```




    {'err0': 0.038799999999999946,
     'err1': 0.03159999999999996,
     'err2': 0.027200000000000002}




```python
sum(errs_1.values())/len(errs_1)
```




    0.0325333333333333



The test error hasn't changed

{% endkatexmm %}
