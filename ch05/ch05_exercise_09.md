---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Estimate of standard error of sample mean of `medv` in `Boston` data set

<div class="toc"><ul class="toc-item"><li><span><a href="#prepare-the-data" data-toc-modified-id="Prepare-the-data-1">Prepare the data</a></span></li><li><span><a href="#a.-Sample-mean-as-estimator-of-population-mean" data-toc-modified-id="a.-Sample-mean-as-estimator-of-population-mean-2">a. Sample mean as estimator of population mean</a></span></li><li><span><a href="#b-plug-in-estimate-of-standard-error-of-the-sample-mean" data-toc-modified-id="b.-Plug-in-estimate-of-standard-error-of-the-sample-mean-3">b. Plug-in estimate of standard error of the sample mean</a></span></li><li><span><a href="#c-boostrap-estimate-of-standard-error-of-sample-mean" data-toc-modified-id="c.-Boostrap-estimate-of-standard-error-of-sample-mean-4">c. Boostrap estimate of standard error of sample mean</a></span></li><li><span><a href="#d-a-95-confidence-interval-for-population-mean" data-toc-modified-id="d.-A-$95\%$-confidence-interval2-for-population-mean-5">d. A 95% confidence interval for population mean</a></span></li><li><span><a href="#e-sample-median-as-estimator-of-population-median" data-toc-modified-id="e.-Sample-median-as-estimator3-of-population-median-6">e. Sample median as estimatorof population median</a></span></li><li><span><a href="#f-boostrap-estimate-of-standard-error-of-sample-median" data-toc-modified-id="f.-Boostrap-estimate-of-standard-error-of-sample-median-7">f. Boostrap estimate of standard error of sample median</a></span></li><li><span><a href="#g-sample-quantile-as-estimator-of-population-quantile" data-toc-modified-id="g.-Sample-quantile4-as-estimator-of-population-quantile-8">g. Sample quantile as estimator of population quantile</a></span></li><li><span><a href="#h-boostrap-estimate-of-standard-error-of-sample-quantile" data-toc-modified-id="h.-Boostrap-estimate-of-standard-error-of-sample-quantile-9">h. Bootstrap estimate of standard error of sample quantile</a></span></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-10">Footnotes</a></span></li></ul></div>

## Prepare the data


```python
import pandas as pd

boston = pd.read_csv('../../datasets/Boston.csv', index_col=0)
boston.head()
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
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
boston.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 506 entries, 1 to 506
    Data columns (total 14 columns):
    crim       506 non-null float64
    zn         506 non-null float64
    indus      506 non-null float64
    chas       506 non-null int64
    nox        506 non-null float64
    rm         506 non-null float64
    age        506 non-null float64
    dis        506 non-null float64
    rad        506 non-null int64
    tax        506 non-null int64
    ptratio    506 non-null float64
    black      506 non-null float64
    lstat      506 non-null float64
    medv       506 non-null float64
    dtypes: float64(11), int64(3)
    memory usage: 59.3 KB


## a. Sample mean as estimator of population mean

Estimate using the sample mean [^1], $\overline{X} = \hat{\mu}$, where $X=$`medv`


```python
mu_hat = boston.medv.mean()
mu_hat
```




    22.532806324110698



## b. Plug-in estimate of standard error of the sample mean

By the Central Limit Theorem, 

$$\mathbf{se}(\hat{\mu}) = \frac{\sigma}{\sqrt{n}}$$

where $\sigma$ is the population deviation. So we have the plug-in estimate [^2]

$$\hat{\mathbf{se}}(\hat{\mu}) = \frac{s}{\sqrt{n}}$$

where $s$ is the sample standard deviation


```python
import numpy as np

mu_hat_se_hat = boston.medv.std()/np.sqrt(len(boston))
mu_hat_se_hat
```




    0.4088611474975351



## c. Boostrap estimate of standard error of sample mean


```python
boot_means = np.array([np.random.choice(boston.medv, size=len(boston.medv), replace=True).mean()
                       for i in range(100)])
boot_means.std()
```




    0.3195039288174233



Very close to the plug in estimate

## d. A $95\%$ confidence interval [^3] for population mean


```python
(mu_hat - 2*mu_hat_se_hat, mu_hat + 2*mu_hat_se_hat)
```




    (21.715084029115626, 23.35052861910577)



## e. Sample median as estimator [^4] of population median


```python
m_hat = boston.medv.median()
m_hat
```




    21.2



## f. Boostrap estimate of standard error of sample median


```python
boot_meds = np.array([np.median(np.random.choice(boston.medv, size=len(boston.medv), replace=True))
                       for i in range(100)])
boot_meds.std()
```




    0.38405045241478325



Since we don't have another estimate to compare this too, we really can't say anything about the accuracy of this one.

## g. Sample quantile [^5] as estimator of population quantile


```python
q_hat = boston.medv.quantile(0.1)
q_hat
```




    12.75



## h. Boostrap estimate of standard error of sample quantile


```python
boot_quantiles = np.array([np.quantile(np.random.choice(boston.medv, size=len(boston.medv), replace=True), 0.1)
                       for i in range(100)])
boot_quantiles.std()
```




    0.42275170017399105



Again since we don't have another estimate to compare this too, we really can't say anything about the accuracy of this one.

## Footnotes

[^1]: The sample mean is the plug-in estimator of the population mean (the plug-in estimate comes from estimating the population cdf with the empirical cdf see [All of Statistics ch 7](http://www.stat.cmu.edu/~larry/all-of-statistics/).

By linearity of expectation, the sample mean is an unbiased estimator of the population mean. By the Law of Large Numbers it is consitent, and by the Central Limit Theorm it is asymptotically normal

[^2]: The sample deviation is the plug-in estimate of the population deviation.

[^3]: This is a normal-based interval, its accuracy relies on the fact that the sample mean is asymptotically normal.

[^4]: The sample median is the plug-in estimate of the poulation median.

[^5]: The sample quantile is the plug-in estimate of the poulation quantile.

{% endkatexmm %}

