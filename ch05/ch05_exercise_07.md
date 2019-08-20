---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Exercise 7: Estimate the LOOCV error

<div class="toc"><ul class="toc-item"><li><span><a href="#prepare-the-data" data-toc-modified-id="Prepare-the-data-1">Prepare the data</a></span></li><li><span><a href="#a" data-toc-modified-id="a.-2">a.</a></span></li><li><span><a href="#b" data-toc-modified-id="b.-3">b.</a></span></li><li><span><a href="#c" data-toc-modified-id="c.-4">c.</a></span></li><li><span><a href="#d-e" data-toc-modified-id="d.,-e.-5">d., e.</a></span></li></ul></div>


## Prepare the data


```python
import pandas as pd

weekly = pd.read_csv("../../datasets/weekly.csv", index_col=0)
```


```python
weekly.head()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1990</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>-3.484</td>
      <td>0.154976</td>
      <td>-0.270</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
</div>




```python
weekly.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1089 entries, 1 to 1089
    Data columns (total 9 columns):
    Year         1089 non-null int64
    Lag1         1089 non-null float64
    Lag2         1089 non-null float64
    Lag3         1089 non-null float64
    Lag4         1089 non-null float64
    Lag5         1089 non-null float64
    Volume       1089 non-null float64
    Today        1089 non-null float64
    Direction    1089 non-null object
    dtypes: float64(7), int64(1), object(1)
    memory usage: 85.1+ KB



```python
weekly['Direction'] = [int(value=="Up") for value in weekly['Direction']]
weekly.head()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1990</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>-3.484</td>
      <td>0.154976</td>
      <td>-0.270</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## a.


```python
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(solver='lbfgs').fit(weekly[['Lag1', 'Lag2']], weekly['Direction'])
```

## b.


```python
df = weekly.drop(labels=1, axis=0)
df.head()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1990</td>
      <td>1.178</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.154444</td>
      <td>-1.372</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
loocv_model = LogisticRegression(solver='lbfgs').fit(df[['Lag1', 'Lag2']], df['Direction'])
```

## c.


```python
first_obs = weekly.iloc[1, ]
loocv_model.predict_proba(first_obs[['Lag1', 'Lag2']].values.reshape(1, -1))
```




    array([[0.42966146, 0.57033854]])



Since P(`Direction="Up"`|`Lag1, Lag2`) = 0.57 > 0.5, we'll predict $\hat{y} = 1$ for this observation. The true value is


```python
first_obs['Direction']
```




    0.0



which is incorrect. 

Note that, since by default the classes of `LogisticRegression()` are equally weighted, we could have got the same prediction directly via


```python
loocv_model.predict(first_obs[['Lag1', 'Lag2']].values.reshape(1, -1))
```




    array([1])



## d., e.


```python
from sklearn.model_selection import LeaveOneOut
import numpy as np

# store loocv predictions
y_pred = np.array([])

# data
X, y = weekly[['Lag1', 'Lag2']].values, weekly['Direction'].values

# LOOCV splits
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_pred = np.append(y_pred, LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test))
```


```python
abs(y_pred - y).mean()
```




    0.44995408631772266



This is a point estimate of the LOOCV error - to do better we'd need to repeat this many times

{% endkatexmm %}
