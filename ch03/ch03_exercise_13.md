---
layout: page
title: 3. Linear Regression
---

{% katexmm %}

# Exercise 13: Regression models on simulated data

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-feature-data" data-toc-modified-id="a.-Generate-feature-data-1">a. Generate feature data</a></span></li><li><span><a href="#b-generate-noise" data-toc-modified-id="b.-Generate-noise-2">b. Generate noise</a></span></li><li><span><a href="#c-generate-response-data" data-toc-modified-id="c.-Generate-response-data-3">c. Generate response data</a></span></li><li><span><a href="#d.-Scatterplot" data-toc-modified-id="d.-Scatterplot-4">d. Scatterplot</a></span></li><li><span><a href="#e-fitting-a-least-squares-regression-model" data-toc-modified-id="e.-Fitting-a-least-squares-regression-model-5">e. Fitting a least squares regression model</a></span></li><li><span><a href="#f-plotting-the-least-squares-and-population-lines" data-toc-modified-id="f.-Plotting-the-least-squares-and-population-lines-6">f. Plotting the least squares and population lines</a></span></li><li><span><a href="#g-fitting-a-least-squares-quadratic-regression-model" data-toc-modified-id="g.-Fitting-a-least-squares-quadratic-regression-model-7">g. Fitting a least squares quadratic regression model</a></span></li><li><span><a href="#h-repeating-a---f-with-less-noise" data-toc-modified-id="h.-Repeating-a.---f.-with-less-noise-8">h. Repeating a. - f. with less noise</a></span></li><li><span><a href="#i-repeating-a---f-with-more-noise" data-toc-modified-id="i.-Repeating-a.---f.-with-more-noise-9">i. Repeating a. - f. with more noise</a></span></li><li><span><a href="#j-confidence-intervals-for-the-coefficients" data-toc-modified-id="j.-Confidence-intervals-for-the-coefficients-10">j. Confidence intervals for the coefficients</a></span></li></ul></div>

## a. Generate feature data


```python
import numpy as np

np.random.seed(0)
x = np.random.normal(loc=0, scale=1, size=100)
```

## b. Generate noise


```python
eps = np.random.normal(loc=0, scale=0.25, size=100)
```

## c. Generate response data


```python
y = -1 + 0.5*x + eps
```


```python
y.shape
```




    (100,)



In this case, $\beta_0 = -1, \beta_1 = 0.5$.

## d. Scatterplot


```python
%matplotlib inline
import seaborn as sns

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a201c7a58>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_11_1.png)


## e. Fitting a least squares regression model


```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression().fit(x.reshape(-1, 1), y)
```

We find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(linear_model.intercept_, 4), round(linear_model.coef_[0], 4))
```




    (-0.9812, 0.5287)



Not too far from the true value of (-1, 0.5)

## f. Plotting the least squares and population lines


```python
import matplotlib.pyplot as plt

%matplotlib inline

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
sns.lineplot(x, linear_model.predict(x.reshape(-1, 1)), label="least squares line")
sns.lineplot(x, -1 + 0.5*x, label="population line")

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a2019b550>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_18_1.png)


## g. Fitting a least squares quadratic regression model


```python
import pandas as pd

data_1 = pd.DataFrame({'x': x, 'x_sq': x**2, 'y': y})

quadratic_model = LinearRegression().fit(data_1[['x', 'x_sq']].values, data_1['y'].values)
```

In this case we find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(quadratic_model.intercept_, 4), round(quadratic_model.coef_[0], 4))
```




    (-0.9643, 0.5308)



The difference between the two models is


```python
(round(abs(linear_model.intercept_ - quadratic_model.intercept_), 4), 
 round(abs(linear_model.coef_[0] - quadratic_model.coef_[0]), 4))
```




    (0.0169, 0.0021)



Still very close to the true value of (-1, 0.5), but not as close as the linear model. This at least is is not evidence that the quadratic term improves the fit

## h. Repeating a. - f. with less noise

###### Generate feature data


```python
import numpy as np

np.random.seed(0)
x = np.random.normal(loc=0, scale=1, size=100)
```

###### Generate noise with order of magnitude less variance


```python
eps = np.random.normal(loc=0, scale=0.025, size=100)
```

###### Generate response data


```python
y = -1 + 0.5*x + eps
```


```python
y.shape
```




    (100,)



###### Scatterplot


```python
import seaborn as sns

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a20fadeb8>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_35_1.png)


###### Fitting a least squares regression model


```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression().fit(x.reshape(-1, 1), y)
```

We find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(linear_model.intercept_, 4), round(linear_model.coef_[0], 4))
```




    (-0.9981, 0.5029)



This is very close to the true value of (-1, 0.5)

###### Plotting the least squares and population lines


```python
import matplotlib.pyplot as plt

%matplotlib inline

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
sns.lineplot(x, linear_model.predict(x.reshape(-1, 1)), label="least squares line")
sns.lineplot(x, -1 + 0.5*x, label="population line")

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a20f06240>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_42_1.png)


###### Fitting a least squares quadratic regression model


```python
import pandas as pd

data_2 = pd.DataFrame({'x': x, 'x_sq': x**2, 'y': y})

quadratic_model = LinearRegression().fit(data_2[['x', 'x_sq']].values, data_2['y'].values)
```

In this case we find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(quadratic_model.intercept_, 4), round(quadratic_model.coef_[0], 4))
```




    (-0.9964, 0.5031)



Still very close to the true value of (-1, 0.5), but not as close as the linear model.The difference between the two models is


```python
(round(abs(linear_model.intercept_ - quadratic_model.intercept_), 4), 
 round(abs(linear_model.coef_[0] - quadratic_model.coef_[0]), 4))
```




    (0.0017, 0.0002)



 However the difference between the two models is smaller now 

## i. Repeating a. - f. with more noise

###### Generate feature data


```python
import numpy as np

np.random.seed(0)
x = np.random.normal(loc=0, scale=1, size=100)
```

###### Generate noise with order of magnitude more variance


```python
eps = np.random.normal(loc=0, scale=2.5, size=100)
```

###### Generate response data


```python
y = -1 + 0.5*x + eps
```


```python
y.shape
```




    (100,)



###### Scatterplot


```python
import seaborn as sns

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2114fbe0>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_59_1.png)


###### Fitting a least squares regression model


```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression().fit(x.reshape(-1, 1), y)
```

We find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(linear_model.intercept_, 4), round(linear_model.coef_[0], 4))
```




    (-0.8121, 0.7867)



Much farther from the true value of (-1, 0.5) in previous cases

###### Plotting the least squares and population lines


```python
import matplotlib.pyplot as plt

%matplotlib inline

sns.scatterplot(x, y, facecolors='grey', edgecolors='grey', alpha=0.5)
sns.lineplot(x, linear_model.predict(x.reshape(-1, 1)), label="least squares line")
sns.lineplot(x, -1 + 0.5*x, label="population line")

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a21073748>




![png]({{site.baseurl}}/assets/images/ch03_exercise_13_66_1.png)


###### Fitting a least squares quadratic regression model


```python
import pandas as pd

data_3 = pd.DataFrame({'x': x, 'x_sq': x**2, 'y': y})

quadratic_model = LinearRegression().fit(data_3[['x', 'x_sq']].values, data_3['y'].values)
```

In this case we find $\hat{\beta}=(\hat{\beta}_0, \hat{\beta}_1)$ is


```python
(round(quadratic_model.intercept_, 4), round(quadratic_model.coef_[0], 4))
```




    (-0.6432, 0.8076)



Even further from the true value than the linear model


```python
(round(abs(linear_model.intercept_ - quadratic_model.intercept_), 4), 
 round(abs(linear_model.coef_[0] - quadratic_model.coef_[0]), 4))
```




    (0.1689, 0.0208)



The difference between the two models is greater than the previous cases

## j. Confidence intervals for the coefficients 


```python
import statsmodels.formula.api as smf

linear_model_1 = smf.ols('y ~ x', data=data_1).fit()
linear_model_2 = smf.ols('y ~ x', data=data_2).fit()
linear_model_3 = smf.ols('y ~ x', data=data_3).fit()
```


```python
linear_model_1.conf_int()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>-1.03283</td>
      <td>-0.929593</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.47755</td>
      <td>0.579800</td>
    </tr>
  </tbody>
</table>
</div>




```python
linear_model_2.conf_int()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>-1.003283</td>
      <td>-0.992959</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.497755</td>
      <td>0.507980</td>
    </tr>
  </tbody>
</table>
</div>




```python
linear_model_3.conf_int()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead th {
        text-aligned: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-aligned: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>-1.328304</td>
      <td>-0.295930</td>
    </tr>
    <tr>
      <th>x</th>
      <td>0.275495</td>
      <td>1.297997</td>
    </tr>
  </tbody>
</table>
</div>

{% endkatexmm %}

