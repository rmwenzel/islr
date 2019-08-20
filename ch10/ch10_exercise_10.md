---
layout: page
title: 10. Unsupervised Learning
---

{% katexmm %}

# Exercise 10: PCA and K-means on simulated data

<div class="toc"><ul class="toc-item"><li><span><a href="#a-generate-data" data-toc-modified-id="a.-Generate-data-1">a. Generate data</a></span></li><li><span><a href="#b-pca" data-toc-modified-id="b.-PCA-2">b. PCA</a></span></li><li><span><a href="#c-k-means-clustering-on-full-dataset-with-k-3" data-toc-modified-id="c.-K-means-clustering-on-full-dataset-with-K-=-3-3">c. K-means clustering on full dataset with K = 3</a></span></li><li><span><a href="#d-k-means-clustering-on-full-dataset-with-k-2" data-toc-modified-id="d.-K-means-clustering-on-full-dataset-with-K-=-2-4">d. K-means clustering on full dataset with K = 2</a></span></li><li><span><a href="#e-k-means-clustering-on-full-dataset-with-k-4" data-toc-modified-id="e.-K-means-clustering-on-full-dataset-with-K-=-4-5">e. K-means clustering on full dataset with K = 4</a></span></li><li><span><a href="#f-k-means-clustering-on-first-two-principal-components-with-k-3" data-toc-modified-id="f.-K-means-clustering-on-first-two-principal-components-with-K-=-3-6">f. K-means clustering on first two principal components with K = 3</a></span></li><li><span><a href="#g-k-means-clustering-on-scaled-dataset-with-k-3" data-toc-modified-id="g.-K-means-clustering-on-scaled-dataset-with-K-=-3-7">g. K-means clustering on scaled dataset with K = 3</a></span></li></ul></div>

## a. Generate data


```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('whitegrid')
```


```python
class_1 = np.random.uniform(low=0, high=1, size=1000).reshape(20, 50)
class_2 = np.random.uniform(low=0, high=2, size=1000).reshape(20, 50)
class_3 = np.random.uniform(low=0, high=3, size=1000).reshape(20, 50)
data = np.concatenate((class_1, class_2, class_3))
data = pd.DataFrame(data, columns=['X'+stri. for i in range(1, 51)])
data['class'] = 20*[1] + 20*[2] + 20*[3]
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
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X42</th>
      <th>X43</th>
      <th>X44</th>
      <th>X45</th>
      <th>X46</th>
      <th>X47</th>
      <th>X48</th>
      <th>X49</th>
      <th>X50</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.206085</td>
      <td>0.548891</td>
      <td>0.483345</td>
      <td>0.430228</td>
      <td>0.084926</td>
      <td>0.457959</td>
      <td>0.313941</td>
      <td>0.796469</td>
      <td>0.899392</td>
      <td>0.140730</td>
      <td>...</td>
      <td>0.803371</td>
      <td>0.575263</td>
      <td>0.868178</td>
      <td>0.887051</td>
      <td>0.659624</td>
      <td>0.130191</td>
      <td>0.271975</td>
      <td>0.206966</td>
      <td>0.272497</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.549837</td>
      <td>0.654331</td>
      <td>0.679153</td>
      <td>0.088327</td>
      <td>0.920253</td>
      <td>0.721935</td>
      <td>0.079815</td>
      <td>0.816096</td>
      <td>0.702062</td>
      <td>0.390909</td>
      <td>...</td>
      <td>0.082494</td>
      <td>0.681500</td>
      <td>0.621130</td>
      <td>0.090764</td>
      <td>0.630456</td>
      <td>0.398784</td>
      <td>0.879749</td>
      <td>0.637703</td>
      <td>0.653399</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.834361</td>
      <td>0.807387</td>
      <td>0.186267</td>
      <td>0.497567</td>
      <td>0.470530</td>
      <td>0.965813</td>
      <td>0.960205</td>
      <td>0.917021</td>
      <td>0.269370</td>
      <td>0.246568</td>
      <td>...</td>
      <td>0.649654</td>
      <td>0.079688</td>
      <td>0.064122</td>
      <td>0.702819</td>
      <td>0.312329</td>
      <td>0.822282</td>
      <td>0.177533</td>
      <td>0.227268</td>
      <td>0.157189</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.485799</td>
      <td>0.830802</td>
      <td>0.311301</td>
      <td>0.787556</td>
      <td>0.177693</td>
      <td>0.451556</td>
      <td>0.901479</td>
      <td>0.970518</td>
      <td>0.929537</td>
      <td>0.944158</td>
      <td>...</td>
      <td>0.504830</td>
      <td>0.045807</td>
      <td>0.168133</td>
      <td>0.188069</td>
      <td>0.998266</td>
      <td>0.684840</td>
      <td>0.206639</td>
      <td>0.748141</td>
      <td>0.650031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.323639</td>
      <td>0.559594</td>
      <td>0.433228</td>
      <td>0.981336</td>
      <td>0.235406</td>
      <td>0.479690</td>
      <td>0.232493</td>
      <td>0.950931</td>
      <td>0.596846</td>
      <td>0.113195</td>
      <td>...</td>
      <td>0.502297</td>
      <td>0.820850</td>
      <td>0.542294</td>
      <td>0.151900</td>
      <td>0.723607</td>
      <td>0.603645</td>
      <td>0.119756</td>
      <td>0.913848</td>
      <td>0.466831</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>



## b. PCA


```python
from sklearn.decomposition import PCA

pca = PCA()
data_trans = pd.DataFrame(pca.fit_transform(data.drop(columns=['class'])), 
                          columns=['Z'+stri. for i in range(1, 51)])
data_trans['class'] = 20*[1] + 20*[2] + 20*[3]
data_trans.head()
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
      <th>Z1</th>
      <th>Z2</th>
      <th>Z3</th>
      <th>Z4</th>
      <th>Z5</th>
      <th>Z6</th>
      <th>Z7</th>
      <th>Z8</th>
      <th>Z9</th>
      <th>Z10</th>
      <th>...</th>
      <th>Z42</th>
      <th>Z43</th>
      <th>Z44</th>
      <th>Z45</th>
      <th>Z46</th>
      <th>Z47</th>
      <th>Z48</th>
      <th>Z49</th>
      <th>Z50</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-3.197494</td>
      <td>0.205622</td>
      <td>-0.441883</td>
      <td>-0.256300</td>
      <td>0.203960</td>
      <td>-0.353133</td>
      <td>0.472105</td>
      <td>0.034966</td>
      <td>0.026696</td>
      <td>0.186835</td>
      <td>...</td>
      <td>0.113655</td>
      <td>0.031458</td>
      <td>0.110456</td>
      <td>-0.094083</td>
      <td>-0.093648</td>
      <td>0.194276</td>
      <td>0.256539</td>
      <td>0.055143</td>
      <td>-0.003031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.377112</td>
      <td>-0.388694</td>
      <td>0.082136</td>
      <td>0.362302</td>
      <td>0.228215</td>
      <td>0.261574</td>
      <td>-0.501082</td>
      <td>-0.484285</td>
      <td>-0.150148</td>
      <td>-0.440468</td>
      <td>...</td>
      <td>0.057817</td>
      <td>-0.178044</td>
      <td>0.209686</td>
      <td>0.150724</td>
      <td>-0.132578</td>
      <td>0.062330</td>
      <td>-0.014915</td>
      <td>-0.030767</td>
      <td>0.006091</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.962193</td>
      <td>0.178716</td>
      <td>0.212917</td>
      <td>0.404123</td>
      <td>-0.071486</td>
      <td>0.267885</td>
      <td>0.302552</td>
      <td>-0.940849</td>
      <td>-0.250239</td>
      <td>-0.387476</td>
      <td>...</td>
      <td>-0.383477</td>
      <td>0.267378</td>
      <td>-0.250597</td>
      <td>-0.173644</td>
      <td>0.058662</td>
      <td>-0.110632</td>
      <td>0.174202</td>
      <td>-0.053465</td>
      <td>0.030769</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.973925</td>
      <td>0.110673</td>
      <td>-0.380835</td>
      <td>0.118737</td>
      <td>-0.252083</td>
      <td>-0.154925</td>
      <td>0.080041</td>
      <td>-0.069936</td>
      <td>-0.091114</td>
      <td>0.446387</td>
      <td>...</td>
      <td>0.076662</td>
      <td>0.028786</td>
      <td>0.345715</td>
      <td>0.146093</td>
      <td>0.171473</td>
      <td>-0.248217</td>
      <td>-0.094597</td>
      <td>-0.127434</td>
      <td>0.150054</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-3.645312</td>
      <td>0.102917</td>
      <td>-0.161366</td>
      <td>0.346589</td>
      <td>0.374630</td>
      <td>0.161183</td>
      <td>-0.680470</td>
      <td>0.095888</td>
      <td>0.020735</td>
      <td>0.602909</td>
      <td>...</td>
      <td>0.176197</td>
      <td>0.056955</td>
      <td>0.274409</td>
      <td>0.044277</td>
      <td>0.098239</td>
      <td>0.230063</td>
      <td>0.273771</td>
      <td>-0.051816</td>
      <td>-0.029336</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
sns.scatterplot(x=data_trans['Z1'], y=data_trans['Z2'], data=data,
                hue='class', palette=sns.color_palette(n_colors=3))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1aa72ac8>




![png]({{site.baseurl}}/assets/images/ch10_exercise_10_6_1.png)


## c. $K$-means clustering on full dataset with $K$=3


```python
from sklearn.cluster import KMeans

kmeans3 = KMeans(n_clusters=3, random_state=0)
kmeans3.fit(data.drop(columns=['class']))
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
        random_state=0, tol=0.0001, verbose=0)




```python
class_labels = pd.DataFrame({'true': data['class'], 
                             'K=3, full': kmeans3.labels_})
class_labels['K=3, full'].value_counts()
```




    0    21
    1    20
    2    19
    Name: K=3, full, dtype: int64



It appears that `kmeans` labeled clusters differently, so we'll adjust. There's a bit of guesswork but it appears the true cluster label $\rightarrow$ kmeans cluster label mapping is $1\mapsto 1, 2 \mapsto 0, 3 \mapsto 2$.


```python
map = {0:2, 1:1, 2:3}
class_labels.loc[:, 'K=3, full'] = [map[label] for label 
                                 in class_labels['K=3, full']]
```


```python
# percentage of class labels kmeans got right
np.sum(class_labels['true'] == class_labels['K=3, full'])/60
```




    0.9833333333333333



## d. $K$-means clustering on full dataset with $K$=2


```python
kmeans2 = KMeans(n_clusters=2, random_state=0)
kmeans2.fit(data.drop(columns=['class']))
```


```python
class_labels['K=2, full'] = kmeans2.labels_
class_labels['K=2, full'].value_counts()
```




    1    31
    0    29
    Name: K=2, full, dtype: int64




```python
class_labels[class_labels['true'] == 1]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 2]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 2]['K=2, full'].value_counts()
```




    1    11
    0     9
    Name: K=2, full, dtype: int64




```python
class_labels[class_labels['true'] == 3]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



For $K=2$, the algorithm correctly separated classes 1 and 3, and assigned class 2 to class 1 and 3 in roughly equal proportion.

## e. $K$-means clustering on full dataset with $K$=4


```python
kmeans4 = KMeans(n_clusters=4, random_state=0)
kmeans4.fit(data.drop(columns=['class']))
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
        random_state=0, tol=0.0001, verbose=0)




```python
class_labels['K=4, full'] = kmeans4.labels_
class_labels['K=4, full'].value_counts()
```




    0    20
    2    19
    3    18
    1     3
    Name: K=4, full, dtype: int64




```python
class_labels[class_labels['K=4, full'] == 0]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['K=4, full'] == 1]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['K=4, full'] == 2]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['K=4, full'] == 3]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>51</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>54</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>57</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



$K=4$ almost correctly identified clusters 1, 2, and 3. The exception was a fourth cluster formed from 3 observations from class 3 (presumably very close to each other).

## f. $K$-means clustering on first two principal components with $K$=3


```python
kmeans3_pca = KMeans(n_clusters=3, random_state=0)
kmeans3_pca.fit(data_trans[['Z1','Z2']])
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
        random_state=0, tol=0.0001, verbose=0)




```python
class_labels['K=3, pca'] = kmeans3_pca.labels_
class_labels['K=3, pca'].value_counts()
```




    2    20
    1    20
    0    20
    Name: K=3, pca, dtype: int64




```python
class_labels[class_labels['true'] == 1]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 2]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 3]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This time $K=3$ perfectly separated the classes

## g. $K$-means clustering on scaled dataset with $K$=3 


```python
data_sc = data/data.std()
kmeans3_sc = KMeans(n_clusters=3, random_state=0)
kmeans3_sc.fit(data_sc.drop(columns=['class']))
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
        random_state=0, tol=0.0001, verbose=0)




```python
class_labels['K=3, scaled'] = kmeans3_sc.labels_
class_labels['K=3, scaled'].value_counts()
```




    0    27
    2    22
    1    11
    Name: K=3, scaled, dtype: int64




```python
class_labels[class_labels['true'] == 1]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
      <th>K=3, scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 2]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
      <th>K=3, scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_labels[class_labels['true'] == 3]
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
      <th>true</th>
      <th>K=3, full</th>
      <th>K=2, full</th>
      <th>K=4, full</th>
      <th>K=3, pca</th>
      <th>K=3, scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



These results are much worse than in part b.. However, this algorithm is highly sensitive to the random initialization - in practice setting a different random seed produces wildly different results. An averaging should be undertaken to reliably compare performance.

{% endkatexmm %}
