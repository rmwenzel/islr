---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}

# Exercise 4: Comparing polynomial, radial, and linear kernel SVMs on a simulated dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#generating-the-data" data-toc-modified-id="Generating-the-data-1">Generating the data</a></span></li><li><span><a href="#Train-test-split" data-toc-modified-id="Train-test-split-2">Train test split</a></span></li><li><span><a href="#svms-with-linear,-polynomial,-and-radial-kernels" data-toc-modified-id="SVMs-with-linear,-polynomial,-and-radial-kernels-3">SVMs with linear, polynomial, and radial kernels</a></span><ul class="toc-item"><li><span><a href="#fit-models" data-toc-modified-id="Fit-models-3.1">Fit models</a></span></li><li><span><a href="#compare-models-on-the-test-data" data-toc-modified-id="Compare-models-on-the-test-data-3.2">Compare models on the test data</a></span><ul class="toc-item"><li><span><a href="#plot-model-predictions" data-toc-modified-id="Plot-model-predictions-3.2.1">Plot model predictions</a></span></li><li><span><a href="#train-and-test-errors" data-toc-modified-id="Train-and-test-errors-3.2.2">Train and test errors</a></span></li></ul></li></ul></li></ul></div>


## Generating the data


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
X = np.random.uniform(0, 3, size=(1000, 2))
Y = np.array([1 if X[i, 1] > 2 * np.sin(X[i, 0]) else -1 for i in range(X.shape[0])])
data = pd.DataFrame({'X_1': X[:, 0], 'X_2': X[:, 1], 'Y': Y})
```


```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='Y')
x = np.linspace(0, 3, 50)
plt.plot(x, 2* np.sin(x), 'r')
```




    [<matplotlib.lines.Line2D at 0x1a1dc46eb8>]




![png]({{site.baseurl}}/assets/images/ch09_exercise_04_4_1.png)


## Train test split


```python
from sklearn.model_selection import train_test_split

X, Y = data[['X_1', 'X_2']], data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=100)
```

## SVMs with linear, polynomial, and radial kernels

Note that what the authors call the support vector classifier is the support vector machine with linear kernel.

### Fit models


```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)

svc_poly = SVC(kernel='poly', degree=6)
svc_poly.fit(X_train, y_train)

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
data_train = pd.concat([X_train, y_train], axis=1)
data_train['linear_pred'] = svc_linear.predict(X_train)
data_train['poly_pred'] = svc_poly.predict(X_train)
data_train['rbf_pred'] = svc_rbf.predict(X_train)
data_train.head()
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
      <th>linear_pred</th>
      <th>poly_pred</th>
      <th>rbf_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>0.358341</td>
      <td>2.459384</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>446</th>
      <td>2.103954</td>
      <td>0.025083</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>232</th>
      <td>2.035364</td>
      <td>2.543201</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>559</th>
      <td>1.507827</td>
      <td>0.750842</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>418</th>
      <td>2.756047</td>
      <td>2.396298</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_test = pd.concat([X_test, y_test], axis=1)
data_test['linear_pred'] = svc_linear.predict(X_test)
data_test['poly_pred'] = svc_poly.predict(X_test)
data_test['rbf_pred'] = svc_rbf.predict(X_test)
data_test.head()
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
      <th>linear_pred</th>
      <th>poly_pred</th>
      <th>rbf_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>127</th>
      <td>0.490400</td>
      <td>0.268376</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>837</th>
      <td>1.329158</td>
      <td>1.556672</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>518</th>
      <td>0.769138</td>
      <td>2.898040</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>743</th>
      <td>0.678482</td>
      <td>1.677742</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.375443</td>
      <td>1.528661</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Compare models on the test data

#### Plot model predictions


```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X_1', y='X_2', data=data_train, hue='linear_pred')
x = np.linspace(0, 3, 50)
sns.lineplot(x, 2*np.sin(x))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b73a9e8>




![png]({{site.baseurl}}/assets/images/ch09_exercise_04_15_1.png)



```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X_1', y='X_2', data=data_train, hue='poly_pred')
x = np.linspace(0, 3, 50)
sns.lineplot(x, 2*np.sin(x))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e8b9630>




![png]({{site.baseurl}}/assets/images/ch09_exercise_04_16_1.png)



```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X_1', y='X_2', data=data_train, hue='rbf_pred')
x = np.linspace(0, 3, 50)
sns.lineplot(x, 2*np.sin(x))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e9289b0>




![png]({{site.baseurl}}/assets/images/ch09_exercise_04_17_1.png)


#### Train and test errors


```python
from sklearn.metrics import accuracy_score

errors = pd.DataFrame(columns=['train', 'test'], index=['linear', 'poly', 'rbf'])

for model in ['linear', 'poly', 'rbf']:
    errors.at[model, 'train'] = accuracy_score(data_train['Y'], data_train[model+'_pred'])
    errors.at[model, 'test'] = accuracy_score(data_test['Y'], data_test[model+'_pred'])
```


```python
errors.sort_values('train', ascending=False)
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
      <th>train</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rbf</th>
      <td>0.97</td>
      <td>0.965556</td>
    </tr>
    <tr>
      <th>poly</th>
      <td>0.92</td>
      <td>0.908889</td>
    </tr>
    <tr>
      <th>linear</th>
      <td>0.84</td>
      <td>0.838889</td>
    </tr>
  </tbody>
</table>
</div>



On both training and testing data, the rankings were the same (1) rbf, (2) degree 2 polynomial, (3) linear.

{% endkatexmm %}
