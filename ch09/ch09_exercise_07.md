---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}

# Exercise 7: Using SVMs to classify mileage in `Auto` dataset


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


```python
auto = pd.read_csv('../../datasets/Auto.csv', index_col=0)
auto.reset_index(inplace=True, drop=True)
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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
    RangeIndex: 392 entries, 0 to 391
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
    memory usage: 27.6+ KB


## a. Create high/low mileage variable and preprocess


```python
# add mileage binary variable
auto['high_mpg'] = (auto['mpg'] > auto['mpg'].median()).astype(int)

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
      <th>high_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We found [this article](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) helpful. We'll scale the data to the interval [0, 1].


```python
df = auto.drop(columns=['name'])
df = (df - df.min())/(df.max() - df.min())
```

## b. Linear SVC


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# rough tuning param
param = {'C': np.logspace(0, 9, 10)}
linear_svc = SVC(kernel='linear')
linear_svc_search = GridSearchCV(estimator=linear_svc,
                                 param_grid=param,
                                 cv=7,
                                 scoring='accuracy')
X, Y = df.drop(columns=['high_mpg']), auto['high_mpg']
%timeit -n1 -r1 linear_svc_search.fit(X, Y)
```

    804 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
linear_svc_search_df = pd.DataFrame(linear_svc_search.cv_results_)
linear_svc_search_df[['param_C', 'mean_test_score']]
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
      <th>param_C</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.908163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.969388</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>0.984694</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100000</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1e+06</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1e+07</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1e+08</td>
      <td>0.997449</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1e+09</td>
      <td>0.997449</td>
    </tr>
  </tbody>
</table>
</div>




```python
linear_svc_search.best_params_
```




    {'C': 1000.0}




```python
linear_svc_search.best_score_
```




    0.9974489795918368




```python
# fine tuning param
param = {'C': np.linspace(500, 1500, 1000)}
linear_svc = SVC(kernel='linear')
linear_svc_search = GridSearchCV(estimator=linear_svc,
                                 param_grid=param,
                                 cv=7,
                                 scoring='accuracy')
%timeit -n1 -r1 linear_svc_search.fit(X, Y)
```

    54.5 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
linear_svc_search.best_params_
```




    {'C': 696.1961961961962}




```python
linear_svc_search.best_score_
```




    0.9974489795918368



## c. Nonlinear SVCs

### Polynomial SVC


```python
# rough param tuning
params = {'C': np.logspace(-4, 4, 9),
          'gamma': np.logspace(-4, 4, 9),
          'degree': [2, 3]}
poly_svc = SVC(kernel='poly')
poly_svc_search = GridSearchCV(estimator=poly_svc,
                                 param_grid=params,
                                 cv=7,
                                 scoring='accuracy')
%timeit -n1 -r1 poly_svc_search.fit(X, Y)
```

    8.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
poly_svc_search.best_params_
```




    {'C': 0.001, 'degree': 2, 'gamma': 100.0}




```python
poly_svc_search.best_score_
```




    0.9617346938775511




```python
params = {'C': np.linspace(0.0001, 0.01, 20),
          'gamma': np.linspace(50, 150, 20)}
poly_svc = SVC(kernel='poly', degree=2)
poly_svc_search = GridSearchCV(estimator=poly_svc,
                                 param_grid=params,
                                 cv=7,
                                 scoring='accuracy')
%timeit -n1 -r1 poly_svc_search.fit(X, Y)
```

    19.3 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
poly_svc_search.best_params_
```




    {'C': 0.0068736842105263166, 'gamma': 50.0}




```python
poly_svc_search.best_score_
```




    0.9668367346938775



### Radial SVC


```python
# rough param tuning
params = {'C': np.logspace(-4, 4, 9),
          'gamma': np.logspace(-4, 4, 9)}
radial_svc = SVC(kernel='rbf')
radial_svc_search = GridSearchCV(estimator=radial_svc,
                                 param_grid=params,
                                 cv=7,
                                 scoring='accuracy')
%timeit -n1 -r1 radial_svc_search.fit(X, Y)
```

    5.34 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
radial_svc_search.best_params_
```




    {'C': 1000.0, 'gamma': 0.01}




```python
radial_svc_search.best_score_
```




    0.9795918367346939




```python
params = {'C': np.logspace(4, 9, 5),
          'gamma': np.linspace(0.001, 0.1, 100)}
radial_svc = SVC(kernel='rbf')
radial_svc_search = GridSearchCV(estimator=radial_svc,
                                 param_grid=params,
                                 cv=7,
                                 scoring='accuracy')
%timeit -n1 -r1 radial_svc_search.fit(X, Y)
```

    20.9 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
radial_svc_search.best_params_
```




    {'C': 177827.94100389228, 'gamma': 0.002}




```python
radial_svc_search.best_score_
```




    0.9974489795918368



## d. CV error plots

### Linear SVC


```python
linear_svc_df = pd.DataFrame(linear_svc_search.cv_results_)
sns.lineplot(x=linear_svc_df['param_C'], y=linear_svc_df['mean_test_score'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1f5e42b0>




![png]({{site.baseurl}}/assets/images/ch09_exercise_07_34_1.png)


### Quadratic SVC


```python
from mpl_toolkits import mplot3d

poly_svc_df = pd.DataFrame(poly_svc_search.cv_results_)

cost, gamma = poly_svc_df['param_C'].unique(), poly_svc_df['param_gamma'].unique()

X, Y = np.meshgrid(cost, gamma)
Z = scores = poly_svc_df['mean_test_score'].values.reshape(len(cost), len(gamma))

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='Greys', edgecolor='none')
ax.set_xlabel('cost')
ax.set_ylabel('gamma')
ax.set_zlabel('cv_accuracy');
ax.view_init(20, 135)
```


![png]({{site.baseurl}}/assets/images/ch09_exercise_07_36_0.png)



```python
radial_svc_df = pd.DataFrame(radial_svc_search.cv_results_)

cost, gamma = radial_svc_df['param_C'].unique(), radial_svc_df['param_gamma'].unique()

X, Y = np.meshgrid(cost, gamma)
Z = scores = radial_svc_df['mean_test_score'].values.reshape(len(gamma), len(cost))

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='Greys', edgecolor='none')
ax.set_xlabel('cost')
ax.set_ylabel('gamma')
ax.set_zlabel('cv_accuracy');
ax.view_init(20, 15)
```


![png]({{site.baseurl}}/assets/images/ch09_exercise_07_37_0.png)

{% endkatexmm %}