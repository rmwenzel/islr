---
layout: page
title: 4. Logistic Regression
---

{% katexmm %}
# Exercise 10: Classifying `Direction` in the `Weekly` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#reparing-the-data" data-toc-modified-id="Preparing-the-Data-1">Preparing the Data</a></span><ul class="toc-item"><li><span><a href="#import" data-toc-modified-id="Import-1.1">Import</a></span></li><li><span><a href="#preprocessing" data-toc-modified-id="Preprocessing-1.2">Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Converting-qualitative-variables-to-quantitative" data-toc-modified-id="Converting-qualitative-variables-to-quantitative-1.2.1">Converting qualitative variables to quantitative</a></span></li></ul></li></ul></li><li><span><a href="#a-numerical-and-graphical-summaries" data-toc-modified-id="a.-Numerical-and-graphical-summaries-2">a. Numerical and graphical summaries</a></span><ul class="toc-item"><li><span><a href="#the-return-variables-are-concentrated" data-toc-modified-id="The-return-variables-are-concentrated-2.1">The return variables are concentrated</a></span></li><li><span><a href="#return-variables-are-nearly-uncorrelated" data-toc-modified-id="Return-variables-are-nearly-uncorrelated-2.2">Return variables are nearly uncorrelated</a></span></li><li><span><a href="#volume-correlates-with-year" data-toc-modified-id="Volume-correlates-with-Year-2.3"><code>Volume</code> correlates with <code>Year</code></a></span></li></ul></li><li><span><a href="#b-logistic-regression-classification-of-direction-using-lag-and-volume-predictors" data-toc-modified-id="b.-Logistic-Regression-Classification-of-Direction-using-Lag-and-Volume-predictors-3">b. Logistic Regression Classification of <code>Direction</code> using <code>Lag</code> and <code>Volume</code> predictors</a></span></li><li><span><a href="#c-confusion-matrix" data-toc-modified-id="c.-Confusion-Matrix-4">c. Confusion Matrix</a></span><ul class="toc-item"><li><span><a href="#performance-rates-of-interest-from-the-confusion-matrix" data-toc-modified-id="Performance-rates-of-interest-from-the-confusion-matrix-4.1">Performance rates of interest from the confusion matrix</a></span></li><li><span><a href="#analyzing-model-performance-rates" data-toc-modified-id="Analyzing-model-performance-rates-4.2">Analyzing model performance rates</a></span><ul class="toc-item"><li><span><a href="#observations" data-toc-modified-id="Observations-4.2.1">Observations</a></span></li></ul></li></ul></li><li><span><a href="#d-logistic-regression-classification-of-direction-using-lag2-predictor" data-toc-modified-id="d.-Logistic-Regression-Classification-of-Direction-using-Lag2-predictor-5">d. Logistic Regression Classification of <code>Direction</code> using <code>Lag2</code> predictor</a></span></li><li><span><a href="#e--other-classification-models-of-direction-using-lag2-predictor" data-toc-modified-id="e.--Other-classification-models-of-Direction-using-Lag2-predictor-6">e.  Other classification models of <code>Direction</code> using <code>Lag2</code> predictor</a></span><ul class="toc-item"><li><span><a href="#lda" data-toc-modified-id="LDA-6.1">LDA</a></span></li><li><span><a href="#qda" data-toc-modified-id="QDA-6.2">QDA</a></span></li><li><span><a href="#knn" data-toc-modified-id="KNN-6.3">KNN</a></span></li><li><span><a href="comparing-results" data-toc-modified-id="Comparing-results-6.4">Comparing results</a></span></li></ul></li><li><span><a href="#h-which-method-has-the-best-results?" data-toc-modified-id="h.-Which-method-has-the-best-results?-7">h. Which method has the best results?</a></span></li><li><span><a href="#i-feature-and-model-selection" data-toc-modified-id="i.-Feature-and-Model-Selection-8">i. Feature and Model Selection</a></span><ul class="toc-item"><li><span><a href="#get-all-predictor-interactions" data-toc-modified-id="Get-all-predictor-interactions-8.1">Get all predictor interactions</a></span></li><li><span><a href="#choose-some-transformations" data-toc-modified-id="Choose-some-transformations-8.2">Choose some transformations</a></span></li><li><span><a href="#Rrndom-data-tweak" data-toc-modified-id="Random-data-tweak-8.3">Random data tweak</a></span></li><li><span><a href="#comparison-of-logit-lda-qda-and-knn-models-on-a-single-random-data-tweak" data-toc-modified-id="Comparison-of-Logit,-LDA,-QDA,-and-KNN-models-on-a-single-random-data-tweak-8.4">Comparison of Logit, LDA, QDA, and KNN models on a single random data tweak</a></span></li><li><span><a href="#comparison-of-logit-lda-qda-and-knn-models-over-$n$-data-tweaks" data-toc-modified-id="Comparison-of-Logit,-LDA,-QDA,-and-KNN-models-over-$n$-data-tweaks-8.5">Comparison of Logit, LDA, QDA, and KNN models over $n$ data tweaks</a></span></li><li><span><a href="#analysis-of-comparisons" data-toc-modified-id="Analysis-of-Comparisons-8.6">Analysis of Comparisons</a></span><ul class="toc-item"><li><span><a href="#analyzing-accuracy-across-models" data-toc-modified-id="Analyzing-accuracy-across-models-8.6.1">Analyzing accuracy across models</a></span><ul class="toc-item"><li><span><a href="#summary-statistics" data-toc-modified-id="Summary-statistics-8.6.1.1">Summary statistics</a></span></li><li><span><a href="#distributions-of-accuracy-across-models" data-toc-modified-id="Distributions-of-accuracy-across-models-8.6.1.2">Distributions of accuracy across models</a></span></li></ul></li></ul></li></ul></li></ul></div>

## Preparing the Data

### Import


```python
import pandas as pd

weekly = pd.read_csv('../../datasets/Weekly.csv', index_col=0)
weekly.head()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1990</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>-3.484</td>
      <td>0.154976</td>
      <td>-0.270</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>Down</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>Up</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>Up</td>
    </tr>
  </tbody>
</table>
</div>




```python
weekly.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1089 entries, 1 to 1089
    Data columns (total 9 columns):
    Year         1089 non-null int64
    Lag1         1089 non-null float64
    Lag2         1089 non-null float64
    Lag3         1089 non-null float64
    Lag4         1089 non-null float64
    Lag5         1089 non-null float64
    Volume       1089 non-null float64
    Today        1089 non-null float64
    Direction    1089 non-null object
    dtypes: float64(7), int64(1), object(1)
    memory usage: 85.1+ KB


### Preprocessing

#### Converting qualitative variables to quantitative

Don't see any null values but let's check


```python
weekly.isna().sum().sum()
```




    0



`Direction` is a qualitative variable encoded as a string; let's encode it numerically 


```python
import sklearn.preprocessing as skl_preprocessing

# create and fit label encoder
direction_le = skl_preprocessing.LabelEncoder()
direction_le.fit(weekly.Direction)

# replace string encoding with numeric
weekly['Direction_num'] = direction_le.transform(weekly.Direction)
weekly.Direction_num.head()
```




    1    0
    2    0
    3    1
    4    1
    5    1
    Name: Direction_num, dtype: int64




```python
direction_le.classes_
```




    array(['Down', 'Up'], dtype=object)




```python
direction_le.transform(direction_le.classes_)
```




    array([0, 1])



So the encoding is {`Down`:0, `Up`:1}

## a. Numerical and graphical summaries

Here's a description of the dataset from the `R` documentation

```
Weekly S&P Stock Market Data

Description

Weekly percentage returns for the S&P 500 stock index between 1990 and 2010.

Usage

Weekly
Format

A data frame with 1089 observations on the following 9 variables.

Year
The year that the observation was recorded

Lag1
Percentage return for previous week

Lag2
Percentage return for 2 weeks previous

Lag3
Percentage return for 3 weeks previous

Lag4
Percentage return for 4 weeks previous

Lag5
Percentage return for 5 weeks previous

Volume
Volume of shares traded (average number of daily shares traded in billions)

Today
Percentage return for this week

Direction
A factor with levels Down and Up indicating whether the market had a positive or negative return on a given week

Source

Raw values of the S&P 500 were obtained from Yahoo Finance and then converted to percentages and lagged.
```

Let's look at summary statistics


```python
weekly.describe()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
      <td>1089.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2000.048669</td>
      <td>0.150585</td>
      <td>0.151079</td>
      <td>0.147205</td>
      <td>0.145818</td>
      <td>0.139893</td>
      <td>1.574618</td>
      <td>0.149899</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.033182</td>
      <td>2.357013</td>
      <td>2.357254</td>
      <td>2.360502</td>
      <td>2.360279</td>
      <td>2.361285</td>
      <td>1.686636</td>
      <td>2.356927</td>
      <td>0.497132</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>-18.195000</td>
      <td>0.087465</td>
      <td>-18.195000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1995.000000</td>
      <td>-1.154000</td>
      <td>-1.154000</td>
      <td>-1.158000</td>
      <td>-1.158000</td>
      <td>-1.166000</td>
      <td>0.332022</td>
      <td>-1.154000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2000.000000</td>
      <td>0.241000</td>
      <td>0.241000</td>
      <td>0.241000</td>
      <td>0.238000</td>
      <td>0.234000</td>
      <td>1.002680</td>
      <td>0.241000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2005.000000</td>
      <td>1.405000</td>
      <td>1.409000</td>
      <td>1.409000</td>
      <td>1.409000</td>
      <td>1.405000</td>
      <td>2.053727</td>
      <td>1.405000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2010.000000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>12.026000</td>
      <td>9.328214</td>
      <td>12.026000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- All the variables ranges look good (e.g. no negative values for volume)

- The `Lag` variables and `Today` all have very similar summary statistics, as expected.

- Of particular interest is `Direction_num`, which has a mean of $\approx 0.56$ and a standard deviation of $\approx 0.50$! 
That's what we'll be trying to predict later in the exercise, which will be interesting.

Let's look at distributions


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-white')
sns.set_style('white')

import warnings  
warnings.filterwarnings('ignore')

import numpy as np
```


```python
sns.pairplot(weekly, hue='Direction')
```




    <seaborn.axisgrid.PairGrid at 0x1a2515d208>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_22_1.png)


### The return variables are concentrated

The return variables  (i.e. `Lag` and `Today` variables) are "tight" instead of spread out, i.e. fairly concentrated about their mean.

Here are the deviations of all variables as percentages of the magnitude of their ranges


```python
weekly_num = weekly.drop('Direction', axis=1)
round(100 * (weekly_num.std() / (weekly_num.max() - weekly_num.min())), 2)
```




    Year             30.17
    Lag1              7.80
    Lag2              7.80
    Lag3              7.81
    Lag4              7.81
    Lag5              7.81
    Volume           18.25
    Today             7.80
    Direction_num    49.71
    dtype: float64



### Return variables are nearly uncorrelated

In the pairplot there is a visible lack of pairwise sample correlation among the return variables.


```python
weekly.corr()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year</th>
      <td>1.000000</td>
      <td>-0.032289</td>
      <td>-0.033390</td>
      <td>-0.030006</td>
      <td>-0.031128</td>
      <td>-0.030519</td>
      <td>0.841942</td>
      <td>-0.032460</td>
      <td>-0.022200</td>
    </tr>
    <tr>
      <th>Lag1</th>
      <td>-0.032289</td>
      <td>1.000000</td>
      <td>-0.074853</td>
      <td>0.058636</td>
      <td>-0.071274</td>
      <td>-0.008183</td>
      <td>-0.064951</td>
      <td>-0.075032</td>
      <td>-0.050004</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>-0.033390</td>
      <td>-0.074853</td>
      <td>1.000000</td>
      <td>-0.075721</td>
      <td>0.058382</td>
      <td>-0.072499</td>
      <td>-0.085513</td>
      <td>0.059167</td>
      <td>0.072696</td>
    </tr>
    <tr>
      <th>Lag3</th>
      <td>-0.030006</td>
      <td>0.058636</td>
      <td>-0.075721</td>
      <td>1.000000</td>
      <td>-0.075396</td>
      <td>0.060657</td>
      <td>-0.069288</td>
      <td>-0.071244</td>
      <td>-0.022913</td>
    </tr>
    <tr>
      <th>Lag4</th>
      <td>-0.031128</td>
      <td>-0.071274</td>
      <td>0.058382</td>
      <td>-0.075396</td>
      <td>1.000000</td>
      <td>-0.075675</td>
      <td>-0.061075</td>
      <td>-0.007826</td>
      <td>-0.020549</td>
    </tr>
    <tr>
      <th>Lag5</th>
      <td>-0.030519</td>
      <td>-0.008183</td>
      <td>-0.072499</td>
      <td>0.060657</td>
      <td>-0.075675</td>
      <td>1.000000</td>
      <td>-0.058517</td>
      <td>0.011013</td>
      <td>-0.018168</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>0.841942</td>
      <td>-0.064951</td>
      <td>-0.085513</td>
      <td>-0.069288</td>
      <td>-0.061075</td>
      <td>-0.058517</td>
      <td>1.000000</td>
      <td>-0.033078</td>
      <td>-0.017995</td>
    </tr>
    <tr>
      <th>Today</th>
      <td>-0.032460</td>
      <td>-0.075032</td>
      <td>0.059167</td>
      <td>-0.071244</td>
      <td>-0.007826</td>
      <td>0.011013</td>
      <td>-0.033078</td>
      <td>1.000000</td>
      <td>0.720025</td>
    </tr>
    <tr>
      <th>Direction_num</th>
      <td>-0.022200</td>
      <td>-0.050004</td>
      <td>0.072696</td>
      <td>-0.022913</td>
      <td>-0.020549</td>
      <td>-0.018168</td>
      <td>-0.017995</td>
      <td>0.720025</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Indeed the magnitudes of the sample correlations for these pairs are quite small.

### `Volume` correlates with `Year`


```python
sns.pairplot(weekly, vars=['Year', 'Volume'], hue='Direction')
```




    <seaborn.axisgrid.PairGrid at 0x1a28503780>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_31_1.png)



```python
sns.regplot(weekly.Year, weekly.Volume, lowess=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2a543be0>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_32_1.png)


## b. Logistic Regression Classification of `Direction` using `Lag` and `Volume` predictors


```python
import statsmodels.api as sm

# predictor labels
predictors = ['Lag' + stri. for i in range(1, 6)]
predictors += ['Volume']

# fit and summarize model

sm_logit_model_full = sm.Logit(weekly.Direction_num, sm.add_constant(weekly[predictors]))
sm_logit_model_full.fit().summary()
```

    Optimization terminated successfully.
             Current function value: 0.682441
             Iterations 4





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>Direction_num</td>  <th>  No. Observations:  </th>  <td>  1089</td> 
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>  1082</td> 
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     6</td> 
</tr>
<tr>
  <th>Date:</th>          <td>Mon, 26 Nov 2018</td> <th>  Pseudo R-squ.:     </th> <td>0.006580</td>
</tr>
<tr>
  <th>Time:</th>              <td>17:27:33</td>     <th>  Log-Likelihood:    </th> <td> -743.18</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -748.10</td>
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th>  <td>0.1313</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>  <td>    0.2669</td> <td>    0.086</td> <td>    3.106</td> <td> 0.002</td> <td>    0.098</td> <td>    0.435</td>
</tr>
<tr>
  <th>Lag1</th>   <td>   -0.0413</td> <td>    0.026</td> <td>   -1.563</td> <td> 0.118</td> <td>   -0.093</td> <td>    0.010</td>
</tr>
<tr>
  <th>Lag2</th>   <td>    0.0584</td> <td>    0.027</td> <td>    2.175</td> <td> 0.030</td> <td>    0.006</td> <td>    0.111</td>
</tr>
<tr>
  <th>Lag3</th>   <td>   -0.0161</td> <td>    0.027</td> <td>   -0.602</td> <td> 0.547</td> <td>   -0.068</td> <td>    0.036</td>
</tr>
<tr>
  <th>Lag4</th>   <td>   -0.0278</td> <td>    0.026</td> <td>   -1.050</td> <td> 0.294</td> <td>   -0.080</td> <td>    0.024</td>
</tr>
<tr>
  <th>Lag5</th>   <td>   -0.0145</td> <td>    0.026</td> <td>   -0.549</td> <td> 0.583</td> <td>   -0.066</td> <td>    0.037</td>
</tr>
<tr>
  <th>Volume</th> <td>   -0.0227</td> <td>    0.037</td> <td>   -0.616</td> <td> 0.538</td> <td>   -0.095</td> <td>    0.050</td>
</tr>
</table>



Only the intercept and `Lag2` and appear to be statistically significant

## c. Confusion Matrix


```python
import sklearn.linear_model as skl_linear_model

# fit model
skl_logit_model_full = skl_linear_model.LogisticRegression()
X, Y = sm.add_constant(weekly[predictors]).values, weekly.Direction_num.values
skl_logit_model_full.fit(X, Y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)




```python
# check paramaters are close in the two models
abs(skl_logit_model_full.coef_ - sm_logit_model_full.fit().params.values)
```

    Optimization terminated successfully.
             Current function value: 0.682441
             Iterations 4





    array([[1.33953314e-01, 6.15092984e-05, 3.94405141e-06, 4.06643634e-05,
            5.96004698e-05, 3.92562567e-05, 3.22507048e-04]])




```python
import sklearn.metrics as skl_metrics

# confusion matrix
confusion_array = skl_metrics.confusion_matrix(Y, skl_logit_model_full.predict(X))
confusion_array
```




    array([[ 54, 430],
           [ 47, 558]])




```python
# confusion data frame
col_index = pd.MultiIndex.from_product([['Pred'], [0, 1]])
row_index = pd.MultiIndex.from_product([['Actual'], [0, 1]])
confusion_df = pd.DataFrame(confusion_array, columns=col_index, index=row_index)
confusion_df.loc[: ,('Pred','Total')] = confusion_df.Pred[0] + confusion_df.Pred[1]
confusion_df.loc[('Actual','Total'), :] = (confusion_df.loc[('Actual',0), :] + 
                                        confusion_df.loc[('Actual',1), :])
confusion_df.astype('int32')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-aligned: middle;
    }

    .dataframe tbody tr th {
        vertical-aligned: top;
    }

    .dataframe thead tr th {
        text-aligned: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" haligned="left">Pred</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valigned="top">Actual</th>
      <th>0</th>
      <td>54</td>
      <td>430</td>
      <td>484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>558</td>
      <td>605</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>101</td>
      <td>988</td>
      <td>1089</td>
    </tr>
  </tbody>
</table>
</div>



### Performance rates of interest from the confusion matrix

Recall that for a binary classifier the confusion matrix shows

$$
\begin{matrix}
\text{Pred} \\
\text{Actual}
\begin{pmatrix}
TN & FP & ActNeg\\
FN & TP & ActPos\\
PredNeg & PredPos & n
\end{pmatrix}
\end{matrix}
$$

where

$$
\begin{aligned}
TP & = \text{True Positives}\\
FP & = \text{False Positives}\\
FN & = \text{False Negatives}\\
TN & = \text{True Negatives}\\
ActNeg &= \text{Total Actual Negatives} = TN + FP\\
ActPos &= \text{Total Actual Positives} = FN + TP\\
PredNeg &= \text{Total Predicted Negatives} = TN + FN\\
PredPos &= \text{Total Predicted Positives} = FP + TP\\
\end{aligned}
$$

Also recall the following rates of interest

- The ***accuracy*** of the classifier is the proportion of correctly classified observations 

$$ \frac{TP + TN}{n} $$

i.e., "How often is the model right?"

- The ***misclassification rate*** (or ***error rate***) is the proportion of incorrectly classified observations 

$$ \frac{FP + FN}{n}$$

i.e., "How often is the model wrong?"

- The ***null error rate*** is the proportion of the majority class

$$ \frac{\max\{ActNeg, ActPos\}}{n} $$

i.e. "How often would we be wrong if we always predicted the majority class" 

- The ***true positive rate*** (or ***sensitivity*** or ***recall***) is the ratio of true positives to actual positives

$$ \frac{TP}{ActPos} $$

, "How often is the model right for actual positives ($Y=1$)?"

- The ***false positive rate*** is the ratio of false positives to actual negatives

$$ \frac{FP}{ActNeg}  = $$

i.e., "How often is the model wrong for actual positives?"

- The ***true negative rate*** or ***specificity*** is the ratio of true negatives to actual negatives

$$ \frac{TN}{ActNeg} $$

i.e., "How often is the model right for actual negatives ($Y=0$)?"

- The ***false negative rate*** is the ratio of true negatives to actual negatives

$$ \frac{FN}{ActNeg} $$

i.e., "How often is the model wrong for actual negatives ($Y=0$)?"

- The ***precision*** is the ratio of true positives to predicted positives

$$ \frac{TP}{PredPos} $$

i.e., "How often are the models' positive predictions right?"

- The ***prevalence*** is the ratio of actual positives to total observations

$$ \frac{FN + TP}{n} $$

i.e., "How often do actual positives occur in the sample?"



### Analyzing model performance rates


```python
# necessary variables
n = confusion_df.loc[('Actual', 'Total'), ('Pred', 'Total')]
TN = confusion_df.loc[('Actual', 0), ('Pred', 0)]
FP = confusion_df.loc[('Actual', 0), ('Pred', 1)]
FN = confusion_df.loc[('Actual', 1), ('Pred', 0)]
TP = confusion_df.loc[('Actual', 1), ('Pred', 1)]

# compute rates
rates = {}
rates['accuracy'] = (TP + TN) / n
rates['error rate'] = (FP + FN) / n
rates['null error'] = max(FN + TP, FP + TN) / n
rates['TP rate'] = TP / (FN + TP)
rates['FP rate'] = FP / (FN + TP)
rates['TN rate'] = TN / (FP + TN)
rates['FN rate'] = FN / (FP + TN)
rates['precision'] = TP / (FP + TP)
rates['prevalence'] = (FN + TP) / n


# store results
model_perf_rates_df = pd.DataFrame(rates, index=[0])
model_perf_rates_df
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.561983</td>
      <td>0.438017</td>
      <td>0.555556</td>
      <td>0.922314</td>
      <td>0.710744</td>
      <td>0.11157</td>
      <td>0.097107</td>
      <td>0.564777</td>
      <td>0.555556</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations

- The accuracy is $\approx 56\%$ so the model is right a bit more than half the time

- The error rate is $\approx 44\%$ so the model is wrong a bit less than half the time

- The "null error rate" is the error rate of the "null classifier" which always predicts the majority class, which in this case is $\approx 56\%$. Thus our model is about as accurate as the null classifier.

- The true positive and false positive rates are relatively high. The true negative and false negative rates are relatively low. This makes sense -- inspection of the confusion matrix shows that the model predicts positives almost an order of magnitude more often

- The precision is $\approx 56\%$ so the model correctly predicts positives a little more than half the time

- The prevalence is also $\approx 56\%$ so positives occur in the sample a little more than half the time

We'll package this functionality for later use:


```python
def confusion_results(X, y, skl_model_fit):
    # get confusion array
    confusion_array = skl_metrics.confusion_matrix(y, skl_model_fit.predict(X))

    # necessary variables
    n = np.sum(confusion_array)
    TN = confusion_array[0, 0]
    FP = confusion_array[0, 1]
    FN = confusion_array[1, 0]
    TP = confusion_array[1, 1]

    # compute rates
    rates = {}
    rates['accuracy'] = (TP + TN) / n
    rates['error rate'] = (FP + FN) / n
    rates['null error'] = max(FN + TP, FP + TN) / n
    rates['TP rate'] = TP / (FN + TP)
    rates['FP rate'] = FP / (FN + TP)
    rates['TN rate'] = TN / (FP + TN)
    rates['FN rate'] = FN / (FP + TN)
    rates['precision'] = TP / (FP + TP)
    rates['prevalence'] = (FN + TP) / n


    # return results
    return pd.DataFrame(rates, index=[0])
```


```python
confusion_results(X, Y, skl_logit_model_full.fit(X, Y))
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.561983</td>
      <td>0.438017</td>
      <td>0.555556</td>
      <td>0.922314</td>
      <td>0.710744</td>
      <td>0.11157</td>
      <td>0.097107</td>
      <td>0.564777</td>
      <td>0.555556</td>
    </tr>
  </tbody>
</table>
</div>



## d. Logistic Regression Classification of `Direction` using `Lag2` predictor

In this section we do some feature selection. Since in b. we found that only `Lag2` was a significant feature, we'll eliminate the others. Further, we'll train on the years 1990 to 2008 and test on 2009 to 2010


```python
# train/test split
weekly_test, weekly_train = weekly[weekly['Year'] <= 2008], weekly[weekly['Year'] > 2008]
X_train, y_train = sm.add_constant(weekly_train['Lag2']), weekly_train['Direction_num']
X_test, y_test = sm.add_constant(weekly_test['Lag2']), weekly_test['Direction_num']

# fit new model
skl_logit_model_lag2 = skl_linear_model.LogisticRegression()
```


```python
# confusion matrix
confusion_results(X_test, y_test, skl_logit_model_lag2.fit(X_train, y_train))
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.550254</td>
      <td>0.449746</td>
      <td>0.552284</td>
      <td>0.963235</td>
      <td>0.777574</td>
      <td>0.040816</td>
      <td>0.045351</td>
      <td>0.553326</td>
      <td>0.552284</td>
    </tr>
  </tbody>
</table>
</div>



The accuracy is actually slightly worse than the full model

## e.  Other classification models of `Direction` using `Lag2` predictor

### LDA


```python
import sklearn.discriminant_analysis as skl_discriminant_analysis

skl_LDA_model_lag2 = skl_discriminant_analysis.LinearDiscriminantAnalysis()
```


```python
# confusion matrix
confusion_results(X_test, y_test, skl_LDA_model_lag2.fit(X_train, y_train))
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.550254</td>
      <td>0.449746</td>
      <td>0.552284</td>
      <td>0.965074</td>
      <td>0.779412</td>
      <td>0.038549</td>
      <td>0.043084</td>
      <td>0.553214</td>
      <td>0.552284</td>
    </tr>
  </tbody>
</table>
</div>




```python
# accuracy
skl_metrics.accuracy_score(y_test, skl_LDA_model_lag2.predict(X_test))
```




    0.550253807106599



Nearly identical to `skl_logit_model_lag2`!

### QDA


```python
skl_QDA_model_lag2 = skl_discriminant_analysis.QuadraticDiscriminantAnalysis()
```


```python
# confusion matrix
confusion_results(X_test, y_test, skl_QDA_model_lag2.fit(X_train, y_train))
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.447716</td>
      <td>0.552284</td>
      <td>0.552284</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.23356</td>
      <td>NaN</td>
      <td>0.552284</td>
    </tr>
  </tbody>
</table>
</div>



Worse accuracy than Logistic Regression and LDA

### KNN


```python
import sklearn.neighbors as skl_neighbors

skl_KNN_model_lag2 = skl_neighbors.KNeighborsClassifier(n_neighbors=1)
```


```python
# confusion matrix
confusion_results(X_test, y_test, skl_KNN_model_lag2.fit(X_train, y_train))
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.518782</td>
      <td>0.481218</td>
      <td>0.552284</td>
      <td>0.626838</td>
      <td>0.498162</td>
      <td>0.385488</td>
      <td>0.460317</td>
      <td>0.55719</td>
      <td>0.552284</td>
    </tr>
  </tbody>
</table>
</div>



### Comparing results

For neatness and ease of comparison, we'll package all these results


```python
def confusion_comparison(X, y, models):
    df = pd.concat([confusion_results(X, y, models[model_name]) 
                      for model_name in models])
    df['Model'] = list(models.keys())
    return df.set_index('Model')
```


```python
models = {'Logit': skl_logit_model_lag2, 'LDA': skl_LDA_model_lag2, 
          'QDA': skl_QDA_model_lag2, 'KNN': skl_KNN_model_lag2}
```


```python
for model_name in models:
    models[model_name] = models[model_name].fit(X_train, y_train)

conf_comp_df = confusion_comparison(X_test, y_test, models)
conf_comp_df
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logit</th>
      <td>0.550254</td>
      <td>0.449746</td>
      <td>0.552284</td>
      <td>0.963235</td>
      <td>0.777574</td>
      <td>0.040816</td>
      <td>0.045351</td>
      <td>0.553326</td>
      <td>0.552284</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.550254</td>
      <td>0.449746</td>
      <td>0.552284</td>
      <td>0.965074</td>
      <td>0.779412</td>
      <td>0.038549</td>
      <td>0.043084</td>
      <td>0.553214</td>
      <td>0.552284</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>0.447716</td>
      <td>0.552284</td>
      <td>0.552284</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.233560</td>
      <td>NaN</td>
      <td>0.552284</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.518782</td>
      <td>0.481218</td>
      <td>0.552284</td>
      <td>0.626838</td>
      <td>0.498162</td>
      <td>0.385488</td>
      <td>0.460317</td>
      <td>0.557190</td>
      <td>0.552284</td>
    </tr>
  </tbody>
</table>
</div>



## h. Which method has the best results?

As measured by accuracy, Logit and LDA models are tied, with KNN not too far behind and QDA a more distant third.

With respect to other confusion metrics, Logit and LDA are nearly identical

## i. Feature and Model Selection

In this section, we'll experiment to try to find improved performance on the test data. We'll try:

- Different subsets of the predictors
- Interactions among the predictors
- Transformations of the predictors
- Different values of $K$ for KNN

### Get all predictor interactions


```python
from itertools import combinations

# all pairs of columns in weekly except year and direction
df = weekly.drop(['Year', 'Direction', 'Direction_num'], axis=1)
col_pairs = combinations(df.columns, 2)

# assemble interactions in dataframe
interaction_df = pd.DataFrame({col1 + ':' + col2: weekly[col1]*weekly[col2] 
                               for (col1, col2) in col_pairs})

# concat data frames
dir_df = weekly[['Direction', 'Direction_num']]
weekly_interact = pd.concat([weekly.drop(['Direction', 'Direction_num'], axis=1),
                             interaction_df, dir_df], axis=1)
weekly_interact.head()
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
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Lag1:Lag2</th>
      <th>Lag1:Lag3</th>
      <th>...</th>
      <th>Lag3:Volume</th>
      <th>Lag3:Today</th>
      <th>Lag4:Lag5</th>
      <th>Lag4:Volume</th>
      <th>Lag4:Today</th>
      <th>Lag5:Volume</th>
      <th>Lag5:Today</th>
      <th>Volume:Today</th>
      <th>Direction</th>
      <th>Direction_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1990</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>-3.484</td>
      <td>0.154976</td>
      <td>-0.270</td>
      <td>1.282752</td>
      <td>-3.211776</td>
      <td>...</td>
      <td>-0.609986</td>
      <td>1.062720</td>
      <td>0.797836</td>
      <td>-0.035490</td>
      <td>0.061830</td>
      <td>-0.539936</td>
      <td>0.940680</td>
      <td>-0.041844</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>-0.229</td>
      <td>0.148574</td>
      <td>-2.576</td>
      <td>-0.220320</td>
      <td>-0.424440</td>
      <td>...</td>
      <td>0.233558</td>
      <td>-4.049472</td>
      <td>0.901344</td>
      <td>-0.584787</td>
      <td>10.139136</td>
      <td>-0.034023</td>
      <td>0.589904</td>
      <td>-0.382727</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1990</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>-3.936</td>
      <td>0.159837</td>
      <td>3.514</td>
      <td>0.695520</td>
      <td>-2.102016</td>
      <td>...</td>
      <td>0.130427</td>
      <td>2.867424</td>
      <td>-6.187392</td>
      <td>0.251265</td>
      <td>5.524008</td>
      <td>-0.629120</td>
      <td>-13.831104</td>
      <td>0.561669</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1990</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>1.572</td>
      <td>0.161630</td>
      <td>0.712</td>
      <td>-9.052064</td>
      <td>-0.948780</td>
      <td>...</td>
      <td>-0.043640</td>
      <td>-0.192240</td>
      <td>1.282752</td>
      <td>0.131890</td>
      <td>0.580992</td>
      <td>0.254082</td>
      <td>1.119264</td>
      <td>0.115081</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1990</td>
      <td>0.712</td>
      <td>3.514</td>
      <td>-2.576</td>
      <td>-0.270</td>
      <td>0.816</td>
      <td>0.153728</td>
      <td>1.178</td>
      <td>2.501968</td>
      <td>-1.834112</td>
      <td>...</td>
      <td>-0.396003</td>
      <td>-3.034528</td>
      <td>-0.220320</td>
      <td>-0.041507</td>
      <td>-0.318060</td>
      <td>0.125442</td>
      <td>0.961248</td>
      <td>0.181092</td>
      <td>Up</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### Choose some transformations




```python
# list of transformations

from numpy import sqrt, sin, log, exp, power

# power functions with odd exponent to preserve sign of inputs
def power_3(array):
    return np.power(array, 3)

def power_5(array):
    return np.power(array, 3)

def power_7(array):
    return np.power(array, 4)

# transformations are functions with domain all real numbers
transforms = [power_3, power_5, power_7, sin, exp]
transforms
```




    [<function __main__.power_3(array)>,
     <function __main__.power_5(array)>,
     <function __main__.power_7(array)>,
     <ufunc 'sin'>,
     <ufunc 'exp'>]



### Random data tweak

We'll write a simple function which returns a dataset which has

- as predictors a random subset of the predictors and interactions of the original dataset (i.e. a random subset of the columns of `weekly_interact`)
- a random subset of its predictors transformed by a random choice of transformations from `transform`


We call such a dataset a "random data tweak"


```python
import numpy as np

def random_data_tweak(weekly_interact, transforms):
    # drop undersirable columns
    weekly_drop = weekly_interact.drop(['Year', 'Direction', 'Direction_num'], axis=1)

    # choose a random subset of the predictors
    predictor_labels = np.random.choice(weekly_drop.columns, 
                                  size=np.random.randint(1, high=weekly_drop.shape[1] + 1),
                                  replace=False)


    # choose a random subset of these to transform
    trans_predictor_labels = np.random.choice(predictor_labels, 
                                          size=np.random.randint(0, len(predictor_labels)),
                                          replace=False)

    # choose random transforms
    some_transforms = np.random.choice(transforms,
                                  size=len(trans_predictor_labels),
                                  replace=True)

    # create the df
    df = weekly_interact[predictor_labels].copy()

    # transform appropriate columns
    for i in range(0, len(trans_predictor_labels)):
        # do transformation
        df.loc[ : , trans_predictor_labels[i]] = some_transforms[i](df[trans_predictor_labels[i]])
        # rename to reflect which transformation was used
        df = df.rename({trans_predictor_labels[i]: 
                    some_transforms[i].__name__ + '(' + trans_predictor_labels[i] + ')' }, axis='columns')

    return pd.concat([df, weekly_interact['Direction_num']], axis=1)
```


```python
random_data_tweak(weekly_interact, transforms).head()
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
      <th>Lag3-Lag5</th>
      <th>power_7(Volume-Today)</th>
      <th>power_5(Lag5)</th>
      <th>Lag2-Lag5</th>
      <th>power_3(Lag4-Lag5)</th>
      <th>sin(Lag4)</th>
      <th>Lag2-Today</th>
      <th>exp(Lag1-Lag5)</th>
      <th>Lag1</th>
      <th>power_5(Lag4-Volume)</th>
      <th>exp(Lag1-Today)</th>
      <th>Lag4-Today</th>
      <th>sin(Lag5-Volume)</th>
      <th>Lag3-Today</th>
      <th>power_3(Lag3-Lag4)</th>
      <th>exp(Lag2)</th>
      <th>Direction_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13.713024</td>
      <td>0.000003</td>
      <td>-42.289684</td>
      <td>-5.476848</td>
      <td>0.507856</td>
      <td>-0.227004</td>
      <td>-0.424440</td>
      <td>0.058254</td>
      <td>0.816</td>
      <td>-0.000045</td>
      <td>0.802262</td>
      <td>0.061830</td>
      <td>-0.514081</td>
      <td>1.062720</td>
      <td>0.732271</td>
      <td>4.816271</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.359988</td>
      <td>0.021456</td>
      <td>-0.012009</td>
      <td>-0.186864</td>
      <td>0.732271</td>
      <td>0.713448</td>
      <td>-2.102016</td>
      <td>1.063781</td>
      <td>-0.270</td>
      <td>-0.199983</td>
      <td>2.004751</td>
      <td>10.139136</td>
      <td>-0.034017</td>
      <td>-4.049472</td>
      <td>-236.877000</td>
      <td>2.261436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.211776</td>
      <td>0.099523</td>
      <td>-60.976890</td>
      <td>1.062720</td>
      <td>-236.877000</td>
      <td>0.999999</td>
      <td>-0.948780</td>
      <td>25314.585232</td>
      <td>-2.576</td>
      <td>0.015863</td>
      <td>0.000117</td>
      <td>5.524008</td>
      <td>-0.588434</td>
      <td>2.867424</td>
      <td>2.110708</td>
      <td>0.763379</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.424440</td>
      <td>0.000175</td>
      <td>3.884701</td>
      <td>-4.049472</td>
      <td>2.110708</td>
      <td>0.728411</td>
      <td>-1.834112</td>
      <td>250.637582</td>
      <td>3.514</td>
      <td>0.002294</td>
      <td>12.206493</td>
      <td>0.580992</td>
      <td>0.251357</td>
      <td>-0.192240</td>
      <td>-0.010695</td>
      <td>0.076078</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.102016</td>
      <td>0.001075</td>
      <td>0.543338</td>
      <td>2.867424</td>
      <td>-0.010695</td>
      <td>-0.266731</td>
      <td>4.139492</td>
      <td>1.787811</td>
      <td>0.712</td>
      <td>-0.000072</td>
      <td>2.313441</td>
      <td>-0.318060</td>
      <td>0.125113</td>
      <td>-3.034528</td>
      <td>0.336456</td>
      <td>33.582329</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Comparison of Logit, LDA, QDA, and KNN models on a single random data tweak


```python
import sklearn.model_selection as skl_model_selection

def model_comparison():
    # tweak data
    tweak_df = random_data_tweak(weekly_interact, transforms)
    
    # train test split
    X, y = sm.add_constant(tweak_df.drop(['Direction_num'], axis=1).values), tweak_df['Direction_num'].values
    X_train, X_test, y_train, y_test = skl_model_selection.train_test_split(X, y)
    
    # dict for models
    models = {}

    # train param models
    models['Logit'] = skl_linear_model.LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    models['LDA'] = skl_discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train, y_train)
    models['QDA'] =  skl_discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    
    # train KNN models for K = 1,..,5
    for i in range(1, 6):
        models['KNN' + stri.] = skl_neighbors.KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    
    
    return {'predictors': tweak_df.columns, 'comparison': confusion_comparison(X_test, y_test, models)}    
```


```python
model_comparison()
```

### Comparison of Logit, LDA, QDA, and KNN models over $n$ data tweaks


```python
def model_comparisons(n):
    # running list of predictors and dfs from each comparison
    predictors_list = []
    dfs = []
    
    # iterate comparisons
    for i in range(n):
        # get comparison result
        result = model_comparison()
        predictors_list += [result['predictors']]
        
        # set MultiIndex
        df = result['comparison']
        df['Instance'] = i
        df = df.reset_index()
        df  = df.set_index(['Instance', 'Model'])
        
        # add results to running lists
        dfs += [df]
    
    return {'predictors': predictors_list, 'comparisons': pd.concat(dfs)}
```


```python
results = model_comparisons(1000)
```

Here are the first 3 comparisons:


```python
comparisons_df = results['comparisons']
comparisons_df.head(21)
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
      <th></th>
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
    <tr>
      <th>Instance</th>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valigned="top">0</th>
      <th>Logit</th>
      <td>0.560440</td>
      <td>0.439560</td>
      <td>0.549451</td>
      <td>0.666667</td>
      <td>0.466667</td>
      <td>0.430894</td>
      <td>0.406504</td>
      <td>0.588235</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.948718</td>
      <td>0.051282</td>
      <td>0.549451</td>
      <td>1.000000</td>
      <td>0.093333</td>
      <td>0.886179</td>
      <td>0.000000</td>
      <td>0.914634</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>0.450549</td>
      <td>0.549451</td>
      <td>0.549451</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.219512</td>
      <td>NaN</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>KNN1</th>
      <td>0.714286</td>
      <td>0.285714</td>
      <td>0.549451</td>
      <td>0.760000</td>
      <td>0.280000</td>
      <td>0.658537</td>
      <td>0.292683</td>
      <td>0.730769</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>KNN2</th>
      <td>0.688645</td>
      <td>0.311355</td>
      <td>0.549451</td>
      <td>0.553333</td>
      <td>0.120000</td>
      <td>0.853659</td>
      <td>0.544715</td>
      <td>0.821782</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>KNN3</th>
      <td>0.703297</td>
      <td>0.296703</td>
      <td>0.549451</td>
      <td>0.740000</td>
      <td>0.280000</td>
      <td>0.658537</td>
      <td>0.317073</td>
      <td>0.725490</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>KNN4</th>
      <td>0.699634</td>
      <td>0.300366</td>
      <td>0.549451</td>
      <td>0.653333</td>
      <td>0.200000</td>
      <td>0.756098</td>
      <td>0.422764</td>
      <td>0.765625</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th>KNN5</th>
      <td>0.703297</td>
      <td>0.296703</td>
      <td>0.549451</td>
      <td>0.760000</td>
      <td>0.300000</td>
      <td>0.634146</td>
      <td>0.292683</td>
      <td>0.716981</td>
      <td>0.549451</td>
    </tr>
    <tr>
      <th rowspan="8" valigned="top">1</th>
      <th>Logit</th>
      <td>0.937729</td>
      <td>0.062271</td>
      <td>0.523810</td>
      <td>0.979021</td>
      <td>0.097902</td>
      <td>0.892308</td>
      <td>0.023077</td>
      <td>0.909091</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.695971</td>
      <td>0.304029</td>
      <td>0.523810</td>
      <td>0.965035</td>
      <td>0.545455</td>
      <td>0.400000</td>
      <td>0.038462</td>
      <td>0.638889</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>0.476190</td>
      <td>0.523810</td>
      <td>0.523810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.100000</td>
      <td>NaN</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>KNN1</th>
      <td>0.706960</td>
      <td>0.293040</td>
      <td>0.523810</td>
      <td>0.769231</td>
      <td>0.328671</td>
      <td>0.638462</td>
      <td>0.253846</td>
      <td>0.700637</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>KNN2</th>
      <td>0.710623</td>
      <td>0.289377</td>
      <td>0.523810</td>
      <td>0.587413</td>
      <td>0.139860</td>
      <td>0.846154</td>
      <td>0.453846</td>
      <td>0.807692</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>KNN3</th>
      <td>0.706960</td>
      <td>0.293040</td>
      <td>0.523810</td>
      <td>0.790210</td>
      <td>0.349650</td>
      <td>0.615385</td>
      <td>0.230769</td>
      <td>0.693252</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>KNN4</th>
      <td>0.695971</td>
      <td>0.304029</td>
      <td>0.523810</td>
      <td>0.650350</td>
      <td>0.230769</td>
      <td>0.746154</td>
      <td>0.384615</td>
      <td>0.738095</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th>KNN5</th>
      <td>0.725275</td>
      <td>0.274725</td>
      <td>0.523810</td>
      <td>0.825175</td>
      <td>0.349650</td>
      <td>0.615385</td>
      <td>0.192308</td>
      <td>0.702381</td>
      <td>0.523810</td>
    </tr>
    <tr>
      <th rowspan="5" valigned="top">2</th>
      <th>Logit</th>
      <td>0.454212</td>
      <td>0.545788</td>
      <td>0.545788</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.201613</td>
      <td>NaN</td>
      <td>0.545788</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.967033</td>
      <td>0.032967</td>
      <td>0.545788</td>
      <td>0.959732</td>
      <td>0.020134</td>
      <td>0.975806</td>
      <td>0.048387</td>
      <td>0.979452</td>
      <td>0.545788</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>0.454212</td>
      <td>0.545788</td>
      <td>0.545788</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.201613</td>
      <td>NaN</td>
      <td>0.545788</td>
    </tr>
    <tr>
      <th>KNN1</th>
      <td>0.501832</td>
      <td>0.498168</td>
      <td>0.545788</td>
      <td>0.510067</td>
      <td>0.422819</td>
      <td>0.491935</td>
      <td>0.588710</td>
      <td>0.546763</td>
      <td>0.545788</td>
    </tr>
    <tr>
      <th>KNN2</th>
      <td>0.487179</td>
      <td>0.512821</td>
      <td>0.545788</td>
      <td>0.308725</td>
      <td>0.248322</td>
      <td>0.701613</td>
      <td>0.830645</td>
      <td>0.554217</td>
      <td>0.545788</td>
    </tr>
  </tbody>
</table>
</div>



Here are the predictors used for the first 3 comparisons:


```python
predictors_list = results['predictors']

predictors_list[0:3]
```




    [Index(['Lag2-Today', 'Lag4-Volume', 'Lag2-Lag3', 'Volume-Today', 'Lag2',
            'power_7(Lag1-Volume)', 'Lag1-Lag2', 'Lag2-Volume', 'Lag4-Today',
            'sin(Lag3-Today)', 'Lag3-Lag4', 'Lag5-Volume', 'Today',
            'power_5(Lag3-Lag5)', 'Lag1-Lag5', 'Lag4', 'Direction_num'],
           dtype='object'),
     Index(['Lag3-Lag5', 'Lag5-Today', 'Lag1', 'sin(Lag2-Volume)', 'Lag2-Today',
            'power_5(Volume)', 'Lag4-Lag5', 'Lag1-Lag2', 'power_5(Lag2-Lag3)',
            'Lag3-Today', 'Lag3-Volume', 'Volume-Today', 'Lag1-Today',
            'Direction_num'],
           dtype='object'),
     Index(['power_3(Lag2-Lag3)', 'Lag1-Today', 'Lag4-Volume', 'sin(Lag1-Lag2)',
            'power_7(Lag2-Today)', 'Lag1', 'exp(Lag2-Lag4)', 'power_3(Volume)',
            'exp(Lag3-Lag5)', 'exp(Lag5)', 'power_5(Lag1-Lag5)', 'Today',
            'Lag1-Lag3', 'Lag4-Today', 'power_5(Lag5-Today)', 'exp(Lag3-Lag4)',
            'power_7(Lag3-Today)', 'sin(Lag2)', 'sin(Volume-Today)',
            'power_3(Lag2-Volume)', 'Lag1-Lag4', 'power_7(Lag4-Lag5)',
            'exp(Lag5-Volume)', 'Direction_num'],
           dtype='object')]



### Analysis of Comparisons

#### Analyzing accuracy across models

##### Summary statistics


```python
grouped_by_model = comparisons_df.groupby(level=1)
grouped_by_model.mean()
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
      <th>accuracy</th>
      <th>error rate</th>
      <th>null error</th>
      <th>TP rate</th>
      <th>FP rate</th>
      <th>TN rate</th>
      <th>FN rate</th>
      <th>precision</th>
      <th>prevalence</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN1</th>
      <td>0.626147</td>
      <td>0.373853</td>
      <td>0.556381</td>
      <td>0.668314</td>
      <td>0.342757</td>
      <td>0.574424</td>
      <td>0.418527</td>
      <td>0.662625</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>KNN2</th>
      <td>0.601912</td>
      <td>0.398088</td>
      <td>0.556381</td>
      <td>0.463810</td>
      <td>0.180901</td>
      <td>0.775631</td>
      <td>0.676137</td>
      <td>0.707457</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>KNN3</th>
      <td>0.629945</td>
      <td>0.370055</td>
      <td>0.556381</td>
      <td>0.690867</td>
      <td>0.358510</td>
      <td>0.555069</td>
      <td>0.390503</td>
      <td>0.661137</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>KNN4</th>
      <td>0.615769</td>
      <td>0.384231</td>
      <td>0.556381</td>
      <td>0.544830</td>
      <td>0.237261</td>
      <td>0.705825</td>
      <td>0.574505</td>
      <td>0.691635</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>KNN5</th>
      <td>0.631011</td>
      <td>0.368989</td>
      <td>0.556381</td>
      <td>0.706117</td>
      <td>0.371881</td>
      <td>0.538619</td>
      <td>0.371593</td>
      <td>0.658446</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.726355</td>
      <td>0.273645</td>
      <td>0.556381</td>
      <td>0.942929</td>
      <td>0.437290</td>
      <td>0.456703</td>
      <td>0.073307</td>
      <td>0.717898</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>Logit</th>
      <td>0.551216</td>
      <td>0.448784</td>
      <td>0.556381</td>
      <td>0.368049</td>
      <td>0.176416</td>
      <td>0.782584</td>
      <td>0.797881</td>
      <td>0.697051</td>
      <td>0.55615</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>0.462040</td>
      <td>0.537960</td>
      <td>0.556381</td>
      <td>0.085262</td>
      <td>0.053065</td>
      <td>0.933012</td>
      <td>1.151742</td>
      <td>0.619088</td>
      <td>0.55615</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_by_model.mean().accuracy.sort_values(ascending=False)
```




    Model
    LDA      0.726355
    KNN5     0.631011
    KNN3     0.629945
    KNN1     0.626147
    KNN4     0.615769
    KNN2     0.601912
    Logit    0.551216
    QDA      0.462040
    Name: accuracy, dtype: float64



Let's look at summary statistics for accuracy


```python
grouped_by_model.accuracy.describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN1</th>
      <td>1000.0</td>
      <td>0.626147</td>
      <td>0.111154</td>
      <td>0.435897</td>
      <td>0.545788</td>
      <td>0.600733</td>
      <td>0.684982</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>KNN2</th>
      <td>1000.0</td>
      <td>0.601912</td>
      <td>0.112694</td>
      <td>0.421245</td>
      <td>0.516484</td>
      <td>0.578755</td>
      <td>0.659341</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>KNN3</th>
      <td>1000.0</td>
      <td>0.629945</td>
      <td>0.110973</td>
      <td>0.443223</td>
      <td>0.549451</td>
      <td>0.604396</td>
      <td>0.682234</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>KNN4</th>
      <td>1000.0</td>
      <td>0.615769</td>
      <td>0.113840</td>
      <td>0.410256</td>
      <td>0.531136</td>
      <td>0.589744</td>
      <td>0.670330</td>
      <td>0.996337</td>
    </tr>
    <tr>
      <th>KNN5</th>
      <td>1000.0</td>
      <td>0.631011</td>
      <td>0.112222</td>
      <td>0.439560</td>
      <td>0.549451</td>
      <td>0.600733</td>
      <td>0.692308</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>1000.0</td>
      <td>0.726355</td>
      <td>0.177813</td>
      <td>0.476190</td>
      <td>0.556777</td>
      <td>0.699634</td>
      <td>0.941392</td>
      <td>0.996337</td>
    </tr>
    <tr>
      <th>Logit</th>
      <td>1000.0</td>
      <td>0.551216</td>
      <td>0.186576</td>
      <td>0.369963</td>
      <td>0.439560</td>
      <td>0.468864</td>
      <td>0.553114</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>QDA</th>
      <td>1000.0</td>
      <td>0.462040</td>
      <td>0.072413</td>
      <td>0.369963</td>
      <td>0.428571</td>
      <td>0.446886</td>
      <td>0.468864</td>
      <td>0.967033</td>
    </tr>
  </tbody>
</table>
</div>



Interesting -- all models were able to get very close to or achieve 100% accuracy at least once. 

*TO DO: Try to find similarities in predictors/transformations for the instances which gave maximum classification accuracy*

Let's look at some distributions.

##### Distributions of accuracy across models

###### Parametric models


```python
import seaborn as sns
sns.set_style('white')

param_models = ['Logit', 'LDA', 'QDA']

for model_name in param_models:
    sns.distplot(grouped_by_model.accuracy.get_group(model_name), label=model_name)

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a254db860>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_105_1.png)


Observations

- There are some interesting peaks here -- the distributions look bimodal.
- Both Logit and QDA models are highly concentrated in low accuracy, but LDA is more spread out

*TO DO: Try to find similarities in predictors/transformations for the instances clustered around these two modes*

###### KNN models


```python
sns.distplot(grouped_by_model.accuracy.get_group('KNN1'), label='KNN1')

plt.legend()
```




    <matplotlib.legend.Legend at 0x1a252924a8>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_108_1.png)



```python
sns.distplot(grouped_by_model.accuracy.get_group('KNN2'), label='KNN2')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2a6e6940>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_109_1.png)



```python
sns.distplot(grouped_by_model.accuracy.get_group('KNN3'), label='KNN3')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25ce2cc0>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_110_1.png)



```python
sns.distplot(grouped_by_model.accuracy.get_group('KNN4'), label='KNN4')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2569ad30>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_111_1.png)



```python
sns.distplot(grouped_by_model.accuracy.get_group('KNN5'), label='KNN5')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25ca3dd8>




![png]({{site.baseurl}}/assets/images/ch04_exercise_10_112_1.png)


The distributions are all very similar, although they appear to become more concentrated as $N$ increases. 

{% endkatexmm %}
