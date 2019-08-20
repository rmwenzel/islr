---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}

# Exercise 6: SVCs for barely linearly separable data

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-data-and-scatterplot" data-toc-modified-id="a.-Generate-data-and-scatterplot-1">a. Generate data and scatterplot</a></span></li><li><span><a href="#b-vross-validation-error-for-svc-as-a-function-of-cost-parameter" data-toc-modified-id="b.-Cross-validation-error-for-SVC-as-a-function-of-cost-parameter-2">b. Cross-validation error for SVC as a function of <code>cost</code> parameter</a></span></li><li><span><a href="#c-test-error-for-svc-as-a-function-of-cost-parameter" data-toc-modified-id="c.-Test-error-for-SVC-as-a-function-of-cost-parameter-3">c. Test error for SVC as a function of <code>cost</code> parameter</a></span></li></ul></div>

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

## a. Generate data and scatterplot


```python
data = pd.DataFrame({'X_1': np.random.uniform(size=1000), 'X_2': np.random.uniform(size=1000),
                     'Y': np.zeros(1000)})

np.random.seed(27)

for i in data.index:
    X_1, X_2 = data.loc[i, 'X_1'], data.loc[i, 'X_2']
    if X_1 + X_2 > 1.05:
        data.loc[i, 'Y'] = 1
    elif X_1 + X_2 < 0.95:
        data.loc[i, 'Y'] = -1
    else:
        data.loc[i, 'Y'] = np.random.choice([-1, 1])
        
data.loc[:, 'Y'] = pd.to_numeric(data['Y'], downcast='integer')
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
      <th>X_1</th>
      <th>X_2</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.349514</td>
      <td>0.564869</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.336660</td>
      <td>0.669171</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.452217</td>
      <td>0.179149</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.724898</td>
      <td>0.141218</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.509156</td>
      <td>0.747757</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='Y')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x105e2d208>




![png]({{site.baseurl}}/assets/images/ch09_exercise_06_4_1.png)


## b. Cross-validation error for SVC as a function of `cost` parameter


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# cost parameter
params = {'C': np.linspace(1, 50, 1000)}

# fit model
svc = SVC(kernel='linear')
svc_search = GridSearchCV(svc, 
                          param_grid=params,
                          cv=10,
                          scoring='accuracy')
svc_search.fit(data[['X_1', 'X_2']], data['Y'])
```




    GridSearchCV(cv=10, error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'C': array([ 1.     ,  1.04905, ..., 49.95095, 50.     ])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)




```python
svc_search_df = pd.DataFrame(svc_search.cv_results_)
svc_search_df.head()
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_C</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>...</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>split5_train_score</th>
      <th>split6_train_score</th>
      <th>split7_train_score</th>
      <th>split8_train_score</th>
      <th>split9_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.005417</td>
      <td>0.001175</td>
      <td>0.001424</td>
      <td>0.000477</td>
      <td>1</td>
      <td>{'C': 1.0}</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.940594</td>
      <td>0.93</td>
      <td>...</td>
      <td>0.952169</td>
      <td>0.956667</td>
      <td>0.951111</td>
      <td>0.952222</td>
      <td>0.953333</td>
      <td>0.957825</td>
      <td>0.952275</td>
      <td>0.948946</td>
      <td>0.953334</td>
      <td>0.002893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.004414</td>
      <td>0.000236</td>
      <td>0.001259</td>
      <td>0.000308</td>
      <td>1.04905</td>
      <td>{'C': 1.049049049049049}</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.93</td>
      <td>...</td>
      <td>0.952169</td>
      <td>0.957778</td>
      <td>0.951111</td>
      <td>0.953333</td>
      <td>0.953333</td>
      <td>0.957825</td>
      <td>0.952275</td>
      <td>0.950055</td>
      <td>0.953667</td>
      <td>0.002855</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.004460</td>
      <td>0.000329</td>
      <td>0.001215</td>
      <td>0.000190</td>
      <td>1.0981</td>
      <td>{'C': 1.098098098098098}</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.93</td>
      <td>...</td>
      <td>0.951057</td>
      <td>0.957778</td>
      <td>0.951111</td>
      <td>0.953333</td>
      <td>0.953333</td>
      <td>0.956715</td>
      <td>0.953385</td>
      <td>0.952275</td>
      <td>0.953555</td>
      <td>0.002108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.004940</td>
      <td>0.000989</td>
      <td>0.001507</td>
      <td>0.000670</td>
      <td>1.14715</td>
      <td>{'C': 1.147147147147147}</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.93</td>
      <td>...</td>
      <td>0.952169</td>
      <td>0.957778</td>
      <td>0.951111</td>
      <td>0.954444</td>
      <td>0.953333</td>
      <td>0.956715</td>
      <td>0.952275</td>
      <td>0.951165</td>
      <td>0.954000</td>
      <td>0.002438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005254</td>
      <td>0.000438</td>
      <td>0.001622</td>
      <td>0.000398</td>
      <td>1.1962</td>
      <td>{'C': 1.1961961961961962}</td>
      <td>0.950495</td>
      <td>0.950495</td>
      <td>0.960396</td>
      <td>0.93</td>
      <td>...</td>
      <td>0.953281</td>
      <td>0.957778</td>
      <td>0.951111</td>
      <td>0.954444</td>
      <td>0.952222</td>
      <td>0.956715</td>
      <td>0.952275</td>
      <td>0.952275</td>
      <td>0.954111</td>
      <td>0.002326</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
sns.lineplot(x=np.linspace(1, 50, 1000), y=svc_search_df['mean_test_score'])
plt.xlabel('logc.')
```




    Text(0.5, 0, 'logc.')




![png]({{site.baseurl}}/assets/images/ch09_exercise_06_8_1.png)



```python
svc_search.best_params_
```




    {'C': 3.305305305305305}




```python
svc_search.best_score_
```




    0.955



## c. Test error for SVC as a function of `cost` parameter


```python
# generate test data
np.random.seed(27)
test_data = pd.DataFrame({'X_1': np.random.uniform(size=1000), 'X_2': np.random.uniform(size=1000),
                     'Y': np.zeros(1000)})

for i in test_data.index:
    X_1, X_2 = test_data.loc[i, 'X_1'], test_data.loc[i, 'X_2']
    if X_1 + X_2 > 1.05:
        test_data.loc[i, 'Y'] = 1
    elif X_1 + X_2 < 0.95:
        test_data.loc[i, 'Y'] = -1
    else:
        test_data.loc[i, 'Y'] = np.random.choice([-1, 1])
        
test_data.loc[:, 'Y'] = pd.to_numeric(test_data['Y'], downcast='integer')
```


```python
from sklearn.metrics import accuracy_score

# train and test data
X_train, Y_train = data[['X_1', 'X_2']], data['Y']
X_test, Y_test = test_data[['X_1', 'X_2']], test_data['Y']

# trained models
svcs = {C: SVC(kernel='linear', C=C).fit(X_train, Y_train) for C in np.linspace(1, 50, 1000)}

# errors df
svcs_train_errors = np.array([accuracy_score(svcs[C].predict(X_train), Y_train) for C in svcs])
svcs_test_errors = np.array([accuracy_score(svcs[C].predict(X_test), Y_test) for C in svcs])
svcs_errors_df = pd.DataFrame({'C': np.linspace(1, 50, 1000), 
                               'train_error': svcs_train_errors,
                               'cv_error': svc_search_df['mean_test_score'],
                               'test_error': svcs_test_errors
                                })
svcs_errors_df.head()
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
      <th>C</th>
      <th>train_error</th>
      <th>cv_error</th>
      <th>test_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.953</td>
      <td>0.952</td>
      <td>0.954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.049049</td>
      <td>0.954</td>
      <td>0.954</td>
      <td>0.954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.098098</td>
      <td>0.954</td>
      <td>0.953</td>
      <td>0.954</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.147147</td>
      <td>0.954</td>
      <td>0.953</td>
      <td>0.953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.196196</td>
      <td>0.954</td>
      <td>0.954</td>
      <td>0.954</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 8))
sns.lineplot(x='C', y='train_error', data=svcs_errors_df, label='train_error')
sns.lineplot(x='C', y='cv_error', data=svcs_errors_df, label='cv_error')
sns.lineplot(x='C', y='test_error', data=svcs_errors_df, label='test_error')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a1a1dc1d0>




![png]({{site.baseurl}}/assets/images/ch09_exercise_06_14_1.png)

{% endkatexmm %}