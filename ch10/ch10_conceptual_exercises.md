---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

## Exercise 1: Part of the argument that the $K$-means clustering algorithm 10.1 reaches a local optimum

<div class="toc"><ul class="toc-item"><li><span><a href="#exercise-1-part-of-the-argument-that-the-k-means-clustering-algorithm-101-reaches-a-local-optimum" data-toc-modified-id="Exercise-1:-Part-of-the-argument-that-the-K-means-clustering-algorithm-10.1-reaches-a-local-optimum-1">Exercise 1: Part of the argument that the $K$-means clustering algorithm 10.1 reaches a local optimum</a></span></li><li><span><a href="#a-proving-identity-1012" data-toc-modified-id="a.-Proving-identity-(10.12)-2">a. Proving identity (10.12)</a></span></li><li><span><a href="#b-arguing-the-algorithm-decreases-the-objective" data-toc-modified-id="b.-Arguing-the-algorithm-decreases-the-objective-3">b. Arguing the algorithm decreases the objective</a></span></li><li><span><a href="#exercise-2-sketching-a-dendrogram" data-toc-modified-id="Exercise-2:-Sketching-a-dendrogram-4">Exercise 2: Sketching a dendrogram</a></span></li><li><span><a href="#a-complete-linkage-dendrogram" data-toc-modified-id="a.-Complete-linkage-dendrogram-5">a. Complete linkage dendrogram</a></span></li><li><span><a href="#b-single-linkage-dendrogram" data-toc-modified-id="b.-Single-linkage-dendrogram-6">b. Single linkage dendrogram</a></span></li><li><span><a href="#c-clusters-from-the-complete-linkage-dendrogram" data-toc-modified-id="c.--Clusters-from-the-complete-linkage-dendrogram-7">c.  Clusters from the complete linkage dendrogram</a></span></li><li><span><a href="#d" data-toc-modified-id="d.-8">d.</a></span></li><li><span><a href="#e" data-toc-modified-id="e.-9">e.</a></span></li><li><span><a href="#exercise-3-manual-example-of-k-means-clustering-for-k2-n6-p2" data-toc-modified-id="Exercise-3:-Manual-example-of-K-means-clustering-for-k2-n6-p2-10">Exercise 3: Manual example of $K$-means clustering for $K=2$, $n=6$, $p=2$.</a></span><ul class="toc-item"><li><span><a href="#a-plot-the-observations" data-toc-modified-id="a.-Plot-the-observations-10.1">a. Plot the observations</a></span></li><li><span><a href="#b-initialize-with-random-cluster-assignment" data-toc-modified-id="b.-Initialize-with-random-cluster-assignment-10.2">b. Initialize with random cluster assignment</a></span></li><li><span><a href="#c-compute-the-centroid-for-each-cluster" data-toc-modified-id="c.-Compute-the-centroid-for-each-cluster-10.3">c. Compute the centroid for each cluster</a></span></li><li><span><a href="#d-assign-observations-to-clusters-by-centroid" data-toc-modified-id="d.-Assign-observations-to-clusters-by-centroid-10.4">d. Assign observations to clusters by centroid</a></span></li><li><span><a href="#e-iterate-c-d-until-cluster-assignments-are-stable" data-toc-modified-id="e.-Iterate-c.,-d.-until-cluster-assignments-are-stable-10.5">e. Iterate c., d. until cluster assignments are stable</a></span></li><li><span><a href="#f-plot-clusters" data-toc-modified-id="f.-Plot-clusters-10.6">f. Plot clusters</a></span></li></ul></li><li><span><a href="#exercise-4-comparing-single-and-complete-linkage" data-toc-modified-id="Exercise-4:-Comparing-single-and-complete-linkage-11">Exercise 4: Comparing single and complete linkage</a></span><ul class="toc-item"><li><span><a href="#a.-For-clusters-123-and-45" data-toc-modified-id="a.-For-clusters-123-and-45-11.1">a. For clusters $\{1, 2, 3\}$ and $\{4, 5\}$.</a></span><ul class="toc-item"><li><span><a href="#An-example-of-hcomp-greater-hsing" data-toc-modified-id="An-example-of-hcomp-greater-hsing-11.1.1">An example of $h_{comp} > h_{sing}$</a></span></li><li><span><a href="#An-example-of-hcomp-less-hsing" data-toc-modified-id="An-example-of-hcomp-less-hsing-11.1.2">An example of $h_{comp} < h_{sing}$</a></span></li></ul></li></ul></li><li><span><a href="#b-for-clusters-5-6" data-toc-modified-id="b.-For-clusters-5-6-12">b. For clusters $\{5\}$, $\{6\}$</a></span></li><li><span><a href="#exercise-5-k-means-clustering-in-fig-1014" data-toc-modified-id="Exercise-5:-K-means-clustering-in-fig-10.14-13">Exercise 5: $K$-means clustering in fig 10.14</a></span><ul class="toc-item"><li><span><a href="#left-hand-scaling" data-toc-modified-id="Left-hand-scaling-13.1">Left-hand scaling</a></span></li><li><span><a href="#middle-scaling" data-toc-modified-id="Middle-scaling-13.2">Middle scaling</a></span></li><li><span><a href="#right-hand-scaling" data-toc-modified-id="Right-hand-scaling-13.3">Right-hand scaling</a></span></li></ul></li><li><span><a href="#exercise-6-a-case-of-pca" data-toc-modified-id="Exercise-6:-A-case-of-PCA-14">Exercise 6: A case of PCA</a></span><ul class="toc-item"><li><span><a href="#a." data-toc-modified-id="a.-14.1">a.</a></span></li></ul></li></ul></div>

## a. Proving identity (10.12)

We'll prove this for the case $p = 1$, from which the result follows easily by induction. 

Starting with the left-hand side of (10.12), we have

$$
\begin{aligned}
\frac{1}{|C_k|}\sum_{i, i' \in C_k} (x_i - x_{i'})^2 &= \frac{1}{|C_k|}\sum_{i \neq i' \in C_k} (x_i - x_{i'})^2\\
& = \frac{2}{|C_k|}\sum_{i < i'} (x_i - x_{i'})^2
\end{aligned}
$$

Now, we work with the right-hand side. 

$$
\begin{aligned}
2\sum_{i  \in C_k} (x_i - \overline{x}_k)^2 &= 2 \sum_{i  \in C_k} \left(x_i - \frac{1}{|C_k|}\sum_{i' \in C_k} x_{i'}\right)^2\\
&= \frac{2}{N^2} \sum_{i } \left((N - 1)x_i - \sum_{i' \neq i} x_{i'}\right)^2\\
\end{aligned}
$$

Where we have let $N = |C_k|$ for readability.

Furthermore

$$
\begin{aligned}
& \left((N - 1)x_i - \sum_{i' \neq i} x_{i'}\right)^2 \\
&= (N - 1)^2 x_i^2 - \sum_{i' \neq i} 2(N - 1)x_ix_{i'} + \sum_{i' \neq i} x_{i'}^2 + \sum_{i', i''\neq i} 2 x_{i'} x_{i''}
\end{aligned}
$$

Thus (4) becomes

$$
\begin{aligned}
& \frac{2}{N^2}\sum_i \left((N - 1)^2 x_i^2 + \sum_{i' \neq i} 2(N - 1)x_ix_{i'} + \sum_{i' \neq i} x_{i'}^2 + \sum_{i', i''\neq i} 2 x_{i'}
x_{i''}\right)\\
&= \frac{2}{N^2}\sum_{i < i'}\Bigg((N - 1)^2 x_i^2  - 2(N - 1)x_ix_{i'} + (N - 1) x_i^2 + 2x_i x_{i'}\Bigg)\\
&= \frac{2}{N^2}\sum_{i < i'} \Bigg(N(N - 1)x_{i}^2 - 2Nx_ix_{i'}\Bigg)\\
&= \frac{2}{N^2}\sum_{i < i'} \Bigg(N x_{i}^2 - 2Nx_ix_{i'} + Nx_{i'}^2\Bigg)\\
&= \frac{2}{N^2}(N)\sum_{i < i'} \Bigg(x_{i}^2 - 2x_ix_{i'} + x_{i'}^2\Bigg)\\
&= \frac{2}{|C_k|}\sum_{i < i'} \left(x_{i} - x_{i'}\right)^2
\end{aligned}
$$

And (12) is the same as (2)

## b. Arguing the algorithm decreases the objective

The identity (10.12) shows that the objective (10.11) is equal to 

$$ \sum_{k =1}^K \left(2\sum_{i\in C_k} ||x_i - \mu_{k}||^2\right) $$

where $\mu_k = (\overline{x}_{k1}, \dots, \overline{x}_{kp})$ is the $k$-th cluster centroid.

At each iteration of the algorithm, observations $x_i$ are reassigned to the cluster $k$ whose centroid is closest, i.e. such that $|| x_i - \mu_k ||^2$ is minimal over $k$. That is, if $k_m$ denotes the cluster assigned to observation $x_i$ on iteration $m$, and $\mu_{k_m}$ the centroid of cluster $x_m$ then 

$$ || x_i - \mu_{k_{m + 1}} ||^2 \leqslant || x_i - \mu_{k_{m}} ||^2 $$

for all $i$. Thus (1) decreases at each iteration.

## Exercise 2: Sketching a dendrogram

## a. Complete linkage dendrogram


```python
%matplotlib inline
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np
```


```python
dist_sq = np.array([[0, 0.3, 0.4, 0.7], [0.3, 0, 0.5, 0.8],
                       [0.4, 0.5, 0, 0.45], [0.7, 0.8, 0.45, 0]])
y = squareform(dist_sq)
```


```python
Z = linkage(y, method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x1', 'x2', 'x3', 'x4'], leaf_rotation=45)
```




    {'icoord': [[5.0, 5.0, 15.0, 15.0],
      [25.0, 25.0, 35.0, 35.0],
      [10.0, 10.0, 30.0, 30.0]],
     'dcoord': [[0.0, 0.3, 0.3, 0.0],
      [0.0, 0.45, 0.45, 0.0],
      [0.3, 0.8, 0.8, 0.45]],
     'ivl': ['x1', 'x2', 'x3', 'x4'],
     'leaves': [0, 1, 2, 3],
     'color_list': ['g', 'r', 'b']}




![png](ch10_conceptual_exercises_9_1.png)


## b. Single linkage dendrogram


```python
Z = linkage(y, method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x1', 'x2', 'x3', 'x4'], leaf_rotation=90)
```




    {'icoord': [[25.0, 25.0, 35.0, 35.0],
      [15.0, 15.0, 30.0, 30.0],
      [5.0, 5.0, 22.5, 22.5]],
     'dcoord': [[0.0, 0.3, 0.3, 0.0],
      [0.0, 0.4, 0.4, 0.3],
      [0.0, 0.45, 0.45, 0.4]],
     'ivl': ['x4', 'x3', 'x1', 'x2'],
     'leaves': [3, 2, 0, 1],
     'color_list': ['g', 'b', 'b']}




![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_11_1.png)


## c.  Clusters from the complete linkage dendrogram

The clusters are

$$\{x_1, x_2\}, \{x_3, x_4\}$$

## d. 

The clusters are

$$\{x_1, x_2, x_3\}, \{x_4\}$$

## e. 

Just exchange $ x_1 \mapsto x_3, x_2 \mapsto x_4$ in the diagram in part [a.](#a.-Complete-linkage-dendrogram)

## Exercise 3: Manual example of $K$-means clustering for $K=2$, $n=6$, $p=2$. 


```python
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')

df = pd.DataFrame({'X1': [1, 1, 0, 5, 6, 4], 'X2': [4, 3, 4, 1, 2, 0]},
                    index=range(1, 7))
df
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
      <th>X1</th>
      <th>X2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### a. Plot the observations


```python
sns.scatterplot(x='X1', y='X2', data=df, color='r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x8168b3d30>




![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_21_1.png)


### b. Initialize with random cluster assignment


```python
np.random.seed(33)
df['cluster'] = np.random.choice([1, 2], replace=True, size=6)
df['cluster']
```




    1    1
    2    2
    3    1
    4    1
    5    1
    6    2
    Name: cluster, dtype: int64




```python
sns.scatterplot(x='X1', y='X2', data=df, hue='cluster', legend='full', 
                palette=sns.color_palette(n_colors=2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x8168bad68>




![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_24_1.png)


### c. Compute the centroid for each cluster


```python
def get_centroids(df):
    # compute centroids
    c1, c2 = df[df['cluster'] == 1], df[df['cluster'] == 2]
    cent_1 = [c1['X1'].mean(), c1['X2'].mean()]
    cent_2 = [c2['X1'].mean(), c2['X2'].mean()]

    return (cent_1, cent_2)
```


```python
cent_1, cent_2 = get_centroids(df)
```

### d. Assign observations to clusters by centroid


```python
def d_cent(cent):
    def f(x):
        return np.linalg.norm(x - cent)
    return f


def assign_to_centroid(cent_1, cent_2):
    def f(x):
        d_1, d_2 = d_cent(cent_1)(x), d_cent(cent_2)(x)
        return 1 if d_1 < d_2 else 2
    return f

def assign_to_clusters(df):
    cent_1, cent_2 = get_centroids(df)
    df = df.drop(columns=['cluster'])
    return df.apply(assign_to_centroid(cent_1, cent_2), axis=1)
```


```python
assign_to_clusters(df)
```




    1    1
    2    1
    3    1
    4    2
    5    1
    6    2
    dtype: int64



### e. Iterate c., d. until cluster assignments are stable


```python
def get_final_clusters(df):
    cl_before, cl_after = df['cluster'], assign_to_clusters(df)
    while not (cl_before == cl_after).any():
        df.loc[:, 'cluster'] = cl_after
        get_final_clusters(df)
    return cl_after
```


```python
get_final_clusters(df)
```




    1    1
    2    1
    3    1
    4    2
    5    1
    6    2
    dtype: int64



### f. Plot clusters


```python
sns.scatterplot(x='X1', y='X2', data=df, hue='cluster', legend='full',
                palette=sns.color_palette(n_colors=2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a191f0198>




![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_35_1.png)


## Exercise 4: Comparing single and complete linkage

### a. For clusters $\{1, 2, 3\}$ and $\{4, 5\}$.

In both linkage diagrams, the clusters $A = \{1, 2, 3\}$ and $B = \{4, 5\}$ fuse when they are the most similar pair of clusters among all pairs at that height.

In the simple linkage diagram, this occurs at the height $h_{sing}(A, B)$ when the *minimum* dissimilarity $d(a, b)$ over pairs $(a, b) \in A \times B$ is less than for any other clusters $C$.
In the complete linkage diagram, this occurs at the height $h_{comp}(A, B)$ when the *maximum* dissimilarity over pairs $d(a, b) \in A \times B$ is less than for any other clusters $C$.

It is possible to have the maximum over other clusters $C$ less than or equal to the minimum, and vice versa. In the former case, $h_{comp} <= h_{sing}$ and in the latter, $h_{comp} >= h_{sing}$. So there is not enough information to tell.

To make this argument a bit more concrete here are some examples:

#### An example of $h_{comp} > h_{sing}$

Consider the following possible subset of observations. Suppose all other observations are far away.


```python
array = np.array([[0, 0], [0, 1], [0, 2], 
                  [4, 1], [4, 2], 
                  [7, 2], [11, 2]])
df1 = pd.DataFrame(array, columns=['X1', 'X2'], index=range(1, 8))
df1
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
      <th>X1</th>
      <th>X2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot(x='X1', y='X2', data=df1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b1abc88>




![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_42_1.png)



```python
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot single linkage dendrogram
plt.subplot(1, 2, 1)
Z = linkage(df1[['X1', 'X2']], method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x'+stri. for i in range(1, 8)], leaf_rotation=45)

# plot complete linkage dendrogram
plt.subplot(1, 2, 2)
Z = linkage(df1[['X1', 'X2']], method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x'+stri. for i in range(1, 8)], leaf_rotation=45)

fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_43_0.png)


In this example, $A = {1, 2, 3}$ and $B ={4, 5}$ are indeed both clusters -- they are formed at height 1 in the simple linkage dendrogram and at height 2 in the complete linkage dendrogram. 

But in this case the height that they are fused is greater for the complete linkage than for the single linkage

$$h_{comp}(A, B) = 11 > h_{sing}(A, B) = 4$$

#### An example of $h_{comp} < h_{sing}$

We use the same observations from the last example, except the $x_7$ point changes $[11, 2] \mapsto [9, 2]$


```python
df1.loc[7, 'X1'] = 9
df1
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
      <th>X1</th>
      <th>X2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot single linkage dendrogram
plt.subplot(1, 2, 1)
Z = linkage(df1[['X1', 'X2']], method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x'+stri. for i in range(1, 8)], leaf_rotation=45)

# plot complete linkage dendrogram
plt.subplot(1, 2, 2)
Z = linkage(df1[['X1', 'X2']], method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=['x'+stri. for i in range(1, 8)], leaf_rotation=45)

fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_48_0.png)


Again in this example, $A = {1, 2, 3}$ and $B ={4, 5}$ are indeed both clusters -- again they are formed at height 1 in the simple linkage dendrogram and at height 2 in the complete linkage dendrogram. 

But in this case the height that they are fused is greater for the single linkage than for the complete linkage

$$h_{comp}(A, B) = 2 < h_{sing}(A, B) = 4$$

## b. For clusters $\{5\}$, $\{6\}$

In this case, with singleton clusters $A = \{5\}$, $B = \{6\}$

$$d_{sing}(A, B) = d_{comp}(A, B) = d(5, 6)$$

so $A, B$ are fused when in the single linkage diagram when 

$$d(5, 6) \leqslant \underset{clusters\ C}{\min}\{\min{d(5, c)} | c \in C\}$$

and in the complete linkage diagram when 

$$d(5, 6) \leqslant \underset{clusters\ C}{\min}\{\max{d(5, c)} | c \in C\}$$

so necessarily

$$h_{comp}(A, B) \geqslant h_{sing}(A, B)$$

## Exercise 5: $K$-means clustering in fig 10.14

For the left-hand scaling, we would expect the orange customer in a cluster by themselves, and the other 7 customers in the other. The orange customer has a minimum distance of 3 from any other customer, and all other 7 customers are a distance one away from some other customer in that group.

For the middle scaling, we would expect to see the customers that bought computers (yellow, blue, red, magenta) in one cluster and the others in the other.

For the right-hand scaling, we would expect to see the same as for the middle scaling.

We can verify these expectations with a computation

### Left-hand scaling


```python
customers = ['black', 'orange', 'lt_blue', 'green', 'yellow', 'dk_blue',
             'red', 'magenta']
purchases = [[8, 0], [11, 0], [7, 0], [6, 0], [5, 1], [6, 1], [7, 1], 
               [8, 1]]
df_left = pd.DataFrame(purchases, columns=['socks', 'computers'], index=customers)
df_left
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
      <th>socks</th>
      <th>computers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>black</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>orange</th>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lt_blue</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>green</th>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>yellow</th>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dk_blue</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>red</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>magenta</th>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot single linkage dendrogram
plt.subplot(1, 2, 1)
Z = linkage(df_left, method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

# plot complete linkage dendrogram
plt.subplot(1, 2, 2)
Z = linkage(df_left, method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_56_0.png)


For both single and complete linkages, the clusters were as expected

### Middle scaling

We don't get the same plot as the book whether we scale by the standard deviation (or the variance) so we'll code this by hand again (with a bit of guesswork)


```python
df_mid = df/df.std()
df_mid.loc[:, 'socks'] = [1.0, 1.4, 0.9, 0.78, 0.62, 0.78, 0.9, 1.0]
df_mid.loc[:, 'computers'] = [0, 0, 0, 0, 1.4, 1.4, 1.4, 1.4]
df_mid
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
      <th>socks</th>
      <th>computers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>black</th>
      <td>1.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>orange</th>
      <td>1.40</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>lt_blue</th>
      <td>0.90</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>green</th>
      <td>0.78</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>yellow</th>
      <td>0.62</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>dk_blue</th>
      <td>0.78</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0.90</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>magenta</th>
      <td>1.00</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot single linkage dendrogram
plt.subplot(1, 2, 1)
Z = linkage(df_mid, method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

# plot complete linkage dendrogram
plt.subplot(1, 2, 2)
Z = linkage(df_mid, method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_61_0.png)


For both single and complete linkages, the clusters were again as expected

### Right-hand scaling

We'll assume $\$2$ per pair of socks, and $\$2000$ per computer


```python
df_right = df_left.copy()
df_right.loc[:, 'socks'] = 2 * df_right['socks']
df_right.loc[:, 'computers'] = 2000 * df_right['computers']
df_right
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
      <th>socks</th>
      <th>computers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>black</th>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>orange</th>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>lt_blue</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>green</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>yellow</th>
      <td>10</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>dk_blue</th>
      <td>12</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>red</th>
      <td>14</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>magenta</th>
      <td>16</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot single linkage dendrogram
plt.subplot(1, 2, 1)
Z = linkage(df_right, method='single', metric='Euclidean')
plt.title('Single Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

# plot complete linkage dendrogram
plt.subplot(1, 2, 2)
Z = linkage(df_right, method='complete', metric='Euclidean')
plt.title('Complete Linkage Dendrogram')
plt.xlabel('sample')
plt.ylabel('Euclidean Distance')
dendrogram(Z, labels=customers, leaf_rotation=45)

fig.tight_layout()
```


![png]({{site.baseurl}}/assets/images/ch10_conceptual_exercises_66_0.png)


Again for both single and complete linkages, the clusters were as expected

## Exercise 6: A case of PCA

### a.

This means that the proportion of variance explained by the first principal component is 0.10. That is, the ratio of the (sample) variance of the first component to the total (sample) variance is 0.10

{% endkatexmm %}