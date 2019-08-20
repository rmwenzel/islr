---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

# Exercise 9: Hierarchical Clustering on `USArrests` dataset

<h1><span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-cluster-with-complete-linkage-and-euclidean-distance" data-toc-modified-id="a.-Cluster-with-complete-linkage-and-Euclidean-distance-2">a. Cluster with complete linkage and Euclidean distance</a></span></li><li><span><a href="#b-cut-dendrogram-to-get-3-clusters" data-toc-modified-id="b.-Cut-dendrogram-to-get-3-clusters-3">b. Cut dendrogram to get 3 clusters</a></span></li><li><span><a href="#c-repeat-clustering-after-scaling" data-toc-modified-id="c.-Repeat-clustering-after-scaling-4">c. Repeat clustering after scaling</a></span></li><li><span><a href="#d-what-effect-does-scaling-have" data-toc-modified-id="d.-What-effect-does-scaling-have?-5">d. What effect does scaling have?</a></span></li></ul></div>

## Preparing the data

Information on the dataset can be [found here](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/USArrests.html)


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


## a. Cluster with complete linkage and Euclidean distance


```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(arrests, method='complete', metric='Euclidean')

plt.figure(figsize=(10, 8))
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
d_1 = dendrogram(Z, labels=arrests.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_09_7_0.png)


## b. Cut dendrogram to get 3 clusters

The clusters are


```python
cluster_1 = d_1['ivl'][:d_1['ivl'].index('Nevada') + 1]
cluster_1
```




    ['Florida',
     'North Carolina',
     'Delaware',
     'Alabama',
     'Louisiana',
     'Alaska',
     'Mississippi',
     'South Carolina',
     'Maryland',
     'Arizona',
     'New Mexico',
     'California',
     'Illinois',
     'New York',
     'Michigan',
     'Nevada']




```python
cluster_2 = d_!['ivl'][d_1['ivl'].index('Nevada') + 1: d_1['ivl'].index('New Jersey') + 1]
cluster_2
```




    ['Missouri',
     'Arkansas',
     'Tennessee',
     'Georgia',
     'Colorado',
     'Texas',
     'Rhode Island',
     'Wyoming',
     'Oregon',
     'Oklahoma',
     'Virginia',
     'Washington',
     'Massachusetts',
     'New Jersey']




```python
cluster_3 = d_1['ivl'][d_1['ivl'].index('New Jersey') + 1 :]
cluster_3
```




    ['Ohio',
     'Utah',
     'Connecticut',
     'Pennsylvania',
     'Nebraska',
     'Kentucky',
     'Montana',
     'Idaho',
     'Indiana',
     'Kansas',
     'Hawaii',
     'Minnesota',
     'Wisconsin',
     'Iowa',
     'New Hampshire',
     'West Virginia',
     'Maine',
     'South Dakota',
     'North Dakota',
     'Vermont']



## c. Repeat clustering after scaling


```python
arrests_sc = arrests/arrests.std()

Z = linkage(arrests_sc, method='complete', metric='Euclidean')

plt.figure(figsize=(10, 8))
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
d_2 = dendrogram(Z, labels=arrests_sc.index, leaf_rotation=90)
```


![png]({{site.baseurl}}/assets/images/ch10_exercise_09_14_0.png)


## d. What effect does scaling have?

Scaling seemingly results in a more "balanced" clustering. In general we know that data should be scaled if variables are measured on incomparable scales (see conceptual exercise 5 for an example). In this case, while `Murder`, `Assault` and `Rape` are measured in the same units, `Urban` is measured in a percentage. 

Thus we conclude the data should be scaled bofore clustering in this case.

{% endkatexmm %}
