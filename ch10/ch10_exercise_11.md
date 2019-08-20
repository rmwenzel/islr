---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

# Exercise 11: Hierarchical clustering on gene expression dataset

<h1><span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#a-preparing-the-data" data-toc-modified-id="a.-Preparing-the-data-1">a. Preparing the data</a></span></li><li><span><a href="#b-hierarchical-clustering" data-toc-modified-id="b.-Hierarchical-clustering-2">b. Hierarchical clustering</a></span><ul class="toc-item"><li><span><a href="#clustering-with-precomputed-correlation-distance" data-toc-modified-id="Clustering-with-precomputed-correlation-distance-2.1">Clustering with precomputed correlation distance</a></span><ul class="toc-item"><li><span><a href="#single-linkage" data-toc-modified-id="Single-linkage-2.1.1">Single linkage</a></span></li><li><span><a href="#complete-linkage" data-toc-modified-id="Complete-linkage-2.1.2">Complete linkage</a></span></li><li><span><a href="#average-linkage" data-toc-modified-id="Average-linkage-2.1.3">Average linkage</a></span></li><li><span><a href="#centroid-linkage" data-toc-modified-id="Centroid-linkage-2.1.4">Centroid linkage</a></span></li></ul></li></ul></li><li><span><a href="#c-which-genes-differ-the-most-across-the-two-groups" data-toc-modified-id="c.-Which-genes-differ-the-most-across-the-two-groups?-3">c. Which genes differ the most across the two groups?</a></span></li></ul></div>

## a. Preparing the data


```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('whitegrid')
```


```python
genes = pd.read_csv('../../datasets/Ch10Ex11.csv', header=None)
genes.head()
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
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.961933</td>
      <td>0.441803</td>
      <td>-0.975005</td>
      <td>1.417504</td>
      <td>0.818815</td>
      <td>0.316294</td>
      <td>-0.024967</td>
      <td>-0.063966</td>
      <td>0.031497</td>
      <td>-0.350311</td>
      <td>...</td>
      <td>-0.509591</td>
      <td>-0.216726</td>
      <td>-0.055506</td>
      <td>-0.484449</td>
      <td>-0.521581</td>
      <td>1.949135</td>
      <td>1.324335</td>
      <td>0.468147</td>
      <td>1.061100</td>
      <td>1.655970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.292526</td>
      <td>-1.139267</td>
      <td>0.195837</td>
      <td>-1.281121</td>
      <td>-0.251439</td>
      <td>2.511997</td>
      <td>-0.922206</td>
      <td>0.059543</td>
      <td>-1.409645</td>
      <td>-0.656712</td>
      <td>...</td>
      <td>1.700708</td>
      <td>0.007290</td>
      <td>0.099062</td>
      <td>0.563853</td>
      <td>-0.257275</td>
      <td>-0.581781</td>
      <td>-0.169887</td>
      <td>-0.542304</td>
      <td>0.312939</td>
      <td>-1.284377</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.258788</td>
      <td>-0.972845</td>
      <td>0.588486</td>
      <td>-0.800258</td>
      <td>-1.820398</td>
      <td>-2.058924</td>
      <td>-0.064764</td>
      <td>1.592124</td>
      <td>-0.173117</td>
      <td>-0.121087</td>
      <td>...</td>
      <td>-0.615472</td>
      <td>0.009999</td>
      <td>0.945810</td>
      <td>-0.318521</td>
      <td>-0.117889</td>
      <td>0.621366</td>
      <td>-0.070764</td>
      <td>0.401682</td>
      <td>-0.016227</td>
      <td>-0.526553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.152132</td>
      <td>-2.213168</td>
      <td>-0.861525</td>
      <td>0.630925</td>
      <td>0.951772</td>
      <td>-1.165724</td>
      <td>-0.391559</td>
      <td>1.063619</td>
      <td>-0.350009</td>
      <td>-1.489058</td>
      <td>...</td>
      <td>-0.284277</td>
      <td>0.198946</td>
      <td>-0.091833</td>
      <td>0.349628</td>
      <td>-0.298910</td>
      <td>1.513696</td>
      <td>0.671185</td>
      <td>0.010855</td>
      <td>-1.043689</td>
      <td>1.625275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.195783</td>
      <td>0.593306</td>
      <td>0.282992</td>
      <td>0.247147</td>
      <td>1.978668</td>
      <td>-0.871018</td>
      <td>-0.989715</td>
      <td>-1.032253</td>
      <td>-1.109654</td>
      <td>-0.385142</td>
      <td>...</td>
      <td>-0.692998</td>
      <td>-0.845707</td>
      <td>-0.177497</td>
      <td>-0.166491</td>
      <td>1.483155</td>
      <td>-1.687946</td>
      <td>-0.141430</td>
      <td>0.200778</td>
      <td>-0.675942</td>
      <td>2.220611</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
genes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 40 columns):
    0     1000 non-null float64
    1     1000 non-null float64
    2     1000 non-null float64
    3     1000 non-null float64
    4     1000 non-null float64
    5     1000 non-null float64
    6     1000 non-null float64
    7     1000 non-null float64
    8     1000 non-null float64
    9     1000 non-null float64
    10    1000 non-null float64
    11    1000 non-null float64
    12    1000 non-null float64
    13    1000 non-null float64
    14    1000 non-null float64
    15    1000 non-null float64
    16    1000 non-null float64
    17    1000 non-null float64
    18    1000 non-null float64
    19    1000 non-null float64
    20    1000 non-null float64
    21    1000 non-null float64
    22    1000 non-null float64
    23    1000 non-null float64
    24    1000 non-null float64
    25    1000 non-null float64
    26    1000 non-null float64
    27    1000 non-null float64
    28    1000 non-null float64
    29    1000 non-null float64
    30    1000 non-null float64
    31    1000 non-null float64
    32    1000 non-null float64
    33    1000 non-null float64
    34    1000 non-null float64
    35    1000 non-null float64
    36    1000 non-null float64
    37    1000 non-null float64
    38    1000 non-null float64
    39    1000 non-null float64
    dtypes: float64(40)
    memory usage: 312.6 KB


## b. Hierarchical clustering

Since we are interested in seeing whether the genes separate the tissue samples into health and diseased classes, the observations are the tissue samples, while the genes are the variables (i.e. we have 40 points in $\mathbb{R}^{1000}$ not 1000 points in $\mathbb{R}^{40}$). Thus we want to work with the transpose of the `gene` matrix.

Note we do not scale the gene variables by their standard deviation, since they are all measured in the same units


```python
tissues = genes.transpose()
tissues.shape
```




    (40, 1000)



Standard python library clustering methods (e.g. `sklearn` and `scipy`) don't have correlation-based distance built-in. We might recall (see [exercise 7](CH10_Exercise_07.ipynb)) that for data with zero mean and standard deviation 1, $d_{Eucl}(x_i, x_j) \propto d_{corr}(x_i, x_j)$. But in this case we are not standardizing the data (see above) so we don't want to use the Euclidean distance.

A better alternative is to precompute the [correlation based distance](https://www.datanovia.com/en/lessons/clustering-distance-measures/)
$d_{corr}(x_i, x_j) = 1 - \text{corr}(x_i, x_j)$ since `scipy`'s linkage will accept a dissimilarity (distance) matrix as input.

### Clustering with precomputed correlation distance


```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# pandas.DataFrame.corr() gives correlation of columns but we want correlation of rows (genes)
corr_dist_matrix = squareform(1 - tissues.transpose().corr())
```


```python
corr_dist_matrix.shape
```




    (780,)



#### Single linkage


```python
plt.figure(figsize=(15, 10))
single_linkage = linkage(corr_dist_matrix, method='single')
d_single = dendrogram(single_linkage, labels=tissues.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_11_13_0.png)


Here single linkage has managed to clearly separate the classes perfectly


```python
# cluster 1
sorted(d_single['ivl'][ : d_single['ivl'].index(3) + 1])
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]




```python
%pprint
```

    Pretty printing has been turned OFF



```python
# cluster 2
sorted(d_single['ivl'][d_single['ivl'].index(3) + 1 : ])
```




    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]



#### Complete linkage


```python
plt.figure(figsize=(15, 10))
complete_linkage = linkage(corr_dist_matrix, method='complete')
d_complete = dendrogram(complete_linkage, labels=tissues.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_11_19_0.png)


Here complete linkage has not managed to clearly separate the classes perfectly. If we cut the dendrogram to get two classes we get clusters


```python
# cluster 1
sorted(d_complete['ivl'][ : d_complete['ivl'].index(17) + 1])
```




    [1, 4, 6, 8, 9, 12, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]




```python
%pprint
```

    Pretty printing has been turned OFF



```python
# cluster 2
sorted(d_complete['ivl'][d_complete['ivl'].index(17) + 1 : ])
```




    [0, 2, 3, 5, 7, 10, 11, 13, 14, 19]



Note however, that if we cut to get three clusters, and merge the two that were not merged, we do get perfect class separation


```python
# cluster 1
sorted(d_complete['ivl'][ : d_complete['ivl'].index(33) + 1])
```




    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]




```python
# cluster 1
sorted(d_complete['ivl'][d_complete['ivl'].index(33) + 1 : ])
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]



#### Average linkage


```python
plt.figure(figsize=(15, 10))
avg_linkage = linkage(corr_dist_matrix, method='average')
d_avg = dendrogram(avg_linkage, labels=tissues.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_11_28_0.png)


Average linkage does a poorer job of separating the classes

#### Centroid linkage


```python
plt.figure(figsize=(15, 10))
cent_linkage = linkage(corr_dist_matrix, method='centroid')
d_cent = dendrogram(cent_linkage, labels=tissues.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_11_31_0.png)


Centroid linkage separates the classes perfectly


```python
# cluster 1
sorted(d_cent['ivl'][ : d_cent['ivl'].index(3) + 1])
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]




```python
# cluster 2
sorted(d_cent['ivl'][d_cent['ivl'].index(3) + 1 : ])
```




    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]



## c. Which genes differ the most across the two groups?

We could answer this question using classical statistical inference. For example, we could do confidence intervals or  hypothesis tests for the difference of means of the expression each gene across the healthy and diseased tissue samples. However, it seems likely the authors intended us to use the techniques of this chapter.

Recall that the first principal component is the direction along which the data has the greatest variance. The loadings $\phi_{1, j}$ are the weights of the variables $X_j$ along this direction, so the magnitude of $\phi_{1, j}$ can be taken as a measure of the degree to which the variable $X_j$ varies across the dataset.

To answer the question of which genes $X_j$ differ across the two groups we can:
- Find $\phi_{1, j}$ the first PCA component loading for the full dataset, and take this as the degree to which the gene $X_j$ varies across all tissue samples.
- Find $\phi^{h}_{1, j}, \phi^{d}_{1, j}$ the first PCA component loadings for the healthy and diseased datasets, respectively, and take each as the degree to which the gene $X_j$ varies within the health and diseased groups, respectively.
- Reason that a high magnitude for $\phi_{1, j}$ will indicate large variance across all tissue samples, while low magnitudes for $\phi^{h}_{1, j}$, $\phi^{d}_{1, j}$ will indicate low variances within the respective tissue sample groups, and conclude that such a $X_j$ differs in its expression across the two groups.
-  Calculate some quantity which allows us to rank $X_j$ in this fashion. We choose
$$ |\phi_{1, j}| - \max\{|\phi^{h}_{1, j}|, |\phi^{c}_{1, j}|\}$$



```python
from sklearn.decomposition import PCA

pca_full, pca_h, pca_c = PCA(), PCA(), PCA()

pca_full.fit(tissues)
pca_h.fit(tissues.loc[:19, :])
pca_d.fit(tissues.loc[20:, :])

phi_full = pca_full.components_.transpose()
phi_h = pca_h.components_transpose()
phi_d = pca_d.components_transpose()

diff_rank = np.abs(phi_full, phi_h, phi_c)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)


{% endkatexmm %}