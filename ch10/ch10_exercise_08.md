---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

# Exercise 8: Calculating PVE for `USArrests` dataset

<h1><span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-calculating-pve-using-a-builtin-method" data-toc-modified-id="a.-Calculating-PVE-using-a-builtin-method-2">a. Calculating PVE using a builtin method</a></span></li><li><span><a href="#b-calculating-pve-by-hand" data-toc-modified-id="b.-Calculating-PVE-by-hand-3">b. Calculating PVE by hand</a></span></li></ul></div>

## Preparing the data


```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('whitegrid')
```


```python
arrests = pd.read_csv('../../datasets/USAressts.csv', index_col=0)
arrests.head()
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
      <th>Murder</th>
      <th>Assault</th>
      <th>UrbanPop</th>
      <th>Rape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama</th>
      <td>13.2</td>
      <td>236</td>
      <td>58</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>10.0</td>
      <td>263</td>
      <td>48</td>
      <td>44.5</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>8.1</td>
      <td>294</td>
      <td>80</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>8.8</td>
      <td>190</td>
      <td>50</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>California</th>
      <td>9.0</td>
      <td>276</td>
      <td>91</td>
      <td>40.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
arrests.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 50 entries, Alabama to Wyoming
    Data columns (total 4 columns):
    Murder      50 non-null float64
    Assault     50 non-null int64
    UrbanPop    50 non-null int64
    Rape        50 non-null float64
    dtypes: float64(2), int64(2)
    memory usage: 2.0+ KB



```python
# standardize the data
arrests_std = (arrests - arrests.mean())/arrests.std()
```

## a. Calculating PVE using a builtin method


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=arrests_std.shape[1])
pca.fit(arrests_std)
```




    PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
pca.explained_variance_ratio_
```




    array([0.62006039, 0.24744129, 0.0891408 , 0.04335752])



## b. Calculating PVE by hand

Confusingly the "components" in sklearn are what the book calls principal component loading vectors, in this case $\phi_1, \dots, \phi_4$ (see table 10.1). They are given as the loading matrix $\phi = [\phi_1, \dots, \phi_4]^\top$


```python
pca.components_.transpose()
```




    array([[ 0.53589947,  0.41818087, -0.34123273,  0.6492278 ],
           [ 0.58318363,  0.1879856 , -0.26814843, -0.74340748],
           [ 0.27819087, -0.87280619, -0.37801579,  0.13387773],
           [ 0.54343209, -0.16731864,  0.81777791,  0.08902432]])



The principal components (or really, the "scores") are given by the transform


```python
Z = pca.transform(arrests_std)
Z = pd.DataFrame(Z)
Z.var()/var_total
```




    0    0.620060
    1    0.247441
    2    0.089141
    3    0.043358
    dtype: float64



Tying this all together, if 

$$X = \begin{pmatrix} -\ \  x_1\ \  - \\ \vdots \\ -\ \  x_n \ \ - \end{pmatrix}$$

is the data matrix and

$$\phi = \begin{pmatrix} | &  & |\\
                         \phi_1 & \cdots & \phi_m\\
                         | &  & |\\ \end{pmatrix}$$
                         
is the loading matrix then

$$
\begin{aligned}
X\phi &= \begin{pmatrix} x_1\cdot\phi_1 & \cdots & x_1\cdot\phi_m \\
                         \vdots & & \vdots\\
                         x_n\cdot\phi_1 & \cdots & x_n\cdot \phi_m
         \end{pmatrix}\\
      &= \begin{pmatrix} | &  & |\\
                         z_1 & \cdots & z_m\\
                         | &  & |\\
         \end{pmatrix}\\
      &= Z
\end{aligned}
$$

is the score matrix (i.e. $z_i = (z_{1i}, \dots, z_{ni})$ is the $i$-th score vectors). 

Indeed


```python
np.matmul(arrests_std, pca.components_.transpose()).head()
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
      <th>Murder</th>
      <th>Assault</th>
      <th>UrbanPop</th>
      <th>Rape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama</th>
      <td>0.975660</td>
      <td>1.122001</td>
      <td>-0.439804</td>
      <td>0.154697</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>1.930538</td>
      <td>1.062427</td>
      <td>2.019500</td>
      <td>-0.434175</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>1.745443</td>
      <td>-0.738460</td>
      <td>0.054230</td>
      <td>-0.826264</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>-0.139999</td>
      <td>1.108542</td>
      <td>0.113422</td>
      <td>-0.180974</td>
    </tr>
    <tr>
      <th>California</th>
      <td>2.498613</td>
      <td>-1.527427</td>
      <td>0.592541</td>
      <td>-0.338559</td>
    </tr>
  </tbody>
</table>
</div>



while


```python
Z.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.975660</td>
      <td>1.122001</td>
      <td>-0.439804</td>
      <td>0.154697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.930538</td>
      <td>1.062427</td>
      <td>2.019500</td>
      <td>-0.434175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.745443</td>
      <td>-0.738460</td>
      <td>0.054230</td>
      <td>-0.826264</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.139999</td>
      <td>1.108542</td>
      <td>0.113422</td>
      <td>-0.180974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.498613</td>
      <td>-1.527427</td>
      <td>0.592541</td>
      <td>-0.338559</td>
    </tr>
  </tbody>
</table>
</div>

{% endkatexmm %}
