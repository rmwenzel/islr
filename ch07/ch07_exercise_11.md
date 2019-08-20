
# Exploring backfitting for multiple linear regression

## a. Generating some data


```python
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

np.random.seed(27)

X_1, X_2, e = np.random.normal(loc=4, scale=2, size=100, ), np.random.exponential(size=100), np.random.normal(size=100)
Y = np.full(100, 3) + X_1 + 2*X_2**2 + e
```


```python
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(X_1, X_2, Y, c=Y)
```




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x1a280b4c88>




![png](ch07_exercise_11_files/ch07_exercise_11_3_1.png)


## b. Setting initial value for $\hat{\beta}_1$.


```python
beta_1_hat = 0.2
```

## c. Fitting $Y - \hat{\beta}_1X_1$


```python
from sklearn.linear_model import LinearRegression

A = Y - beta_1_hat*X_1
linreg = LinearRegression()
linreg.fit(X_2.reshape(-1, 1), A)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
beta_0_hat = linreg.intercept_
f'beta_0_hat = {beta_0_hat}'
```




    'beta_0_hat = 2.8350247328203544'




```python
beta_2_hat = linreg.coef_[0]
f'beta_2_hat = {beta_2_hat}'
```




    'beta_2_hat = 7.512178662654875'



## d. Fitting $Y - \hat{\beta}_2 X_2$


```python
B = Y - beta_2_hat*X_2
linreg = LinearRegression()
linreg.fit(X_1.reshape(-1, 1), B)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
beta_0_hat = linreg.intercept_
f'beta_0_hat = {beta_0_hat}'
```




    'beta_0_hat = -0.8300795357307318'




```python
beta_1_hat = linreg.coef_[0]
f'beta_1_hat = {beta_1_hat}'
```




    'beta_1_hat = 1.0727863090555483'



## e., f. Backfitting to estimate OLS multiple regression coefficients


```python
def backfit(beta_0_hat, beta_1_hat, beta_2_hat, n_iters=100):
    coefs = {'beta_0_hat':[beta_0_hat], 'beta_1_hat':[beta_1_hat], 'beta_2_hat':[beta_2_hat]}
    for i in range(n_iters - 1):
        
        # new beta_2_hat
        A = Y - beta_1_hat*X_1
        linreg = LinearRegression()
        linreg.fit(X_2.reshape(-1, 1), A)
        beta_2_hat = linreg.coef_[0]
        
        # new beta_0_hat, beta_1_hat
        B = Y - beta_2_hat*X_2
        linreg = LinearRegression()
        linreg.fit(X_1.reshape(-1, 1), B)
        beta_0_hat, beta_1_hat = linreg.intercept_, linreg.coef_[0]
        
        # update dict
        coefs['beta_0_hat'] += [beta_0_hat]
        coefs['beta_1_hat'] += [beta_1_hat]
        coefs['beta_2_hat'] += [beta_2_hat]
        
    return coefs
```


```python
n_iters = 20
backfit_coefs = pd.DataFrame(backfit(beta_0_hat, beta_1_hat, beta_2_hat, n_iters=n_iters))
backfit_coefs.head()
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
      <th>beta_0_hat</th>
      <th>beta_1_hat</th>
      <th>beta_2_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.830080</td>
      <td>1.072786</td>
      <td>7.512179</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.132632</td>
      <td>1.090675</td>
      <td>7.717398</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.138833</td>
      <td>1.091042</td>
      <td>7.721604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.138960</td>
      <td>1.091049</td>
      <td>7.721690</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.138963</td>
      <td>1.091050</td>
      <td>7.721692</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get OLS multiple regression coeffs
X = np.stack((X_1, X_2), axis=1)
mreg = LinearRegression()
mreg.fit(X, Y)
mreg_coefs = pd.DataFrame({'beta_0_hat': n_iters*[linreg.intercept_], 
              'beta_1_hat': n_iters*[linreg.coef_[0]], 
              'beta_2_hat': n_iters*[linreg.coef_[1]]})

x = np.arange(n_iters)

plt.figure(figsize=(15, 10))
plt.plot(x, backfit_coefs['beta_0_hat'], color='blue')
plt.plot(x, backfit_coefs['beta_1_hat'], color='red')
plt.plot(x, backfit_coefs['beta_2_hat'], color='green')

plt.plot(x, mreg_coefs['beta_0_hat'], color='blue', linestyle='dashed')
plt.plot(x, mreg_coefs['beta_1_hat'], color='red', linestyle='dashed')
plt.plot(x, mreg_coefs['beta_2_hat'], color='green', linestyle='dashed')

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a292b8860>




![png](ch07_exercise_11_files/ch07_exercise_11_17_1.png)


## g. How may iterations needed for a good approximation?


```python
# differences of backfit and mreg coefs
backfit_coefs - mreg_coefs
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
      <th>beta_0_hat</th>
      <th>beta_1_hat</th>
      <th>beta_2_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.088834e-01</td>
      <td>-1.826322e-02</td>
      <td>-2.095132e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.330967e-03</td>
      <td>-3.743286e-04</td>
      <td>-4.294247e-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.297614e-04</td>
      <td>-7.672352e-06</td>
      <td>-8.801619e-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.659630e-06</td>
      <td>-1.572549e-07</td>
      <td>-1.804007e-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.451259e-08</td>
      <td>-3.223144e-09</td>
      <td>-3.697547e-08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.117309e-09</td>
      <td>-6.606249e-11</td>
      <td>-7.578631e-10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.290435e-11</td>
      <td>-1.354250e-12</td>
      <td>-1.553691e-11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.729550e-13</td>
      <td>-2.797762e-14</td>
      <td>-3.206324e-13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.287859e-14</td>
      <td>-4.440892e-16</td>
      <td>-1.065814e-14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.220446e-15</td>
      <td>0.000000e+00</td>
      <td>-2.664535e-15</td>
    </tr>
  </tbody>
</table>
</div>



The differences are all exceedingly small after only a few iterations
