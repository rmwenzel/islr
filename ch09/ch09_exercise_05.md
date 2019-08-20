---
layout: page
title: 9. Support Vector Machines
---

{% katexmm %}

# Exercise 5: Nonlinear decision boundary with logistic regression and nonlinear feature transformation

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-data" data-toc-modified-id="a.-Generate-data-1">a. Generate data</a></span></li><li><span><a href="#b-scatterplot" data-toc-modified-id="b.-Scatterplot-2">b. Scatterplot</a></span></li><li><span><a href="#c-train-linear-logistic-tegression-model" data-toc-modified-id="c.-Train-linear-Logistic-Regression-model-3">c. Train linear Logistic Regression model</a></span></li><li><span><a href="#d-linear-logistic-regression-model-prediction-for-training-data" data-toc-modified-id="d.-Linear-logistic-regression-model-prediction-for-training-data-4">d. Linear logistic regression model prediction for training data</a></span></li><li><span><a href="#e-train-nonlinear-logisitic-regression-models" data-toc-modified-id="e.-Train-nonlinear-logisitic-regression-models-5">e. Train nonlinear logisitic regression models</a></span></li><li><span><a href="#f-nonlinear-logistic-regression-model-prediction-for-training-data" data-toc-modified-id="f.-Nonlinear-logistic-regression-model-prediction-for-training-data-6">f. Nonlinear logistic regression model prediction for training data</a></span></li><li><span><a href="#g-train-linear-svc-and-get-prediction-for-training-data" data-toc-modified-id="g.-Train-linear-SVC-and-get-prediction-for-training-data-7">g. Train linear SVC and get prediction for training data</a></span></li><li><span><a href="#h-train-nonlinear-svm-and-get-prediction-for-training-data" data-toc-modified-id="h.-Train-nonlinear-SVM-and-get-prediction-for-training-data-8">h. Train nonlinear SVM and get prediction for training data</a></span></li></ul></div>

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

## a. Generate data


```python
X_1, X_2 = np.random.uniform(size=500) - 0.5, np.random.uniform(size=500) - 0.5
Y = np.sign(X_1**2 - X_2**2)

data = pd.DataFrame({'X_1': X_1, 'X_2': X_2, 'Y': Y})
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
      <td>0.314655</td>
      <td>0.347183</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.042885</td>
      <td>-0.171818</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.365191</td>
      <td>-0.486412</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.284827</td>
      <td>-0.241447</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.061084</td>
      <td>-0.190989</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>



## b. Scatterplot


```python
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='Y')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b95db70>




![png]({{site.baseurl}}/assets/images/ch09_exercise_05_5_1.png)


## c. Train linear Logistic Regression model

Here we fit a logistic regression model
$$ P(Y = k) = \exp(\beta_0 + \beta_1 X_1 + \beta_2 X_2)$$


```python
from sklearn.linear_model import LogisticRegression

linear_logit = LogisticRegression()
linear_logit.fit(data[['X_1', 'X_2']], data['Y'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)



## d. Linear logistic regression model prediction for training data


```python
data['linear_logit'] = linear_logit.predict(data[['X_1', 'X_2']])


plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='linear_logit')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1c960e48>




![png]({{site.baseurl}}/assets/images/ch09_exercise_05_10_1.png)


## e. Train nonlinear logisitic regression models

We'll train a model

$$P(Y = k) =  \exp\left(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1^2 + \beta_4 X_2^2 + \beta_5 X_1X_2
                        + \beta_6 \log(|X_1|) + \beta_7 \log(|X_2|) + \beta_8 \log(|X_1 X_2|) \right)$$





```python
# transform data
trans_data = data[['X_1', 'X_2']].copy()
trans_data['X_1^2'] = data['X_1']**2
trans_data['X_2^2'] = data['X_2']**2
trans_data['X_1X_2'] = data['X_1']*data['X_2']
trans_data['log(X_1)'] = np.log(np.absolute(trans_data[['X_1']]))
trans_data['log(X_2)'] = np.log(np.absolute(trans_data[['X_2']]))
trans_data['log(X_1X_2)'] = np.log(np.absolute(trans_data[['X_1X_2']]))
trans_data['Y'] = data['Y']
trans_data.head()
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
      <th>X_1^2</th>
      <th>X_2^2</th>
      <th>X_1X_2</th>
      <th>log(X_1)</th>
      <th>log(X_2)</th>
      <th>log(X_1X_2)</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.314655</td>
      <td>0.347183</td>
      <td>0.099008</td>
      <td>0.120536</td>
      <td>0.109243</td>
      <td>-1.156278</td>
      <td>-1.057902</td>
      <td>-2.214180</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.042885</td>
      <td>-0.171818</td>
      <td>0.001839</td>
      <td>0.029521</td>
      <td>-0.007368</td>
      <td>-3.149226</td>
      <td>-1.761322</td>
      <td>-4.910548</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.365191</td>
      <td>-0.486412</td>
      <td>0.133364</td>
      <td>0.236596</td>
      <td>0.177633</td>
      <td>-1.007335</td>
      <td>-0.720700</td>
      <td>-1.728035</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.284827</td>
      <td>-0.241447</td>
      <td>0.081126</td>
      <td>0.058297</td>
      <td>0.068771</td>
      <td>-1.255874</td>
      <td>-1.421104</td>
      <td>-2.676979</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.061084</td>
      <td>-0.190989</td>
      <td>0.003731</td>
      <td>0.036477</td>
      <td>0.011666</td>
      <td>-2.795504</td>
      <td>-1.655540</td>
      <td>-4.451044</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fit model
nonlinear_logit = LogisticRegression()
nonlinear_logit.fit(trans_data.drop(columns=['Y']), trans_data['Y'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)



## f. Nonlinear logistic regression model prediction for training data


```python
trans_data['nonlinear_logit'] = nonlinear_logit.predict(trans_data.drop(columns=['Y']))

plt.figure(figsize=(10, 8))
sns.scatterplot(x=trans_data['X_1'], y=trans_data['X_2'], data=trans_data, hue='nonlinear_logit')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1caaa780>




![png]({{site.baseurl}}/assets/images/ch09_exercise_05_16_1.png)


## g. Train linear SVC and get prediction for training data


```python
from sklearn.svm import SVC

linear_svc = SVC(kernel='linear')
linear_svc.fit(data.drop(columns=['Y']), data['Y'])
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
data['linear_svc'] = linear_svc.predict(data.drop(columns=['Y']))

plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='linear_svc')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1cc38eb8>




![png]({{site.baseurl}}/assets/images/ch09_exercise_05_19_1.png)


## h. Train nonlinear SVM and get prediction for training data


```python
rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(data.drop(columns=['Y']), data['Y'])
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
data['rbf_svc'] = rbf_svc.predict(data.drop(columns=['Y']))

plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['X_1'], y=data['X_2'], data=data, hue='rbf_svc')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1cdd05f8>


![png]({{site.baseurl}}/assets/images/ch09_exercise_05_22_1.png)

{% endkatexmm %}