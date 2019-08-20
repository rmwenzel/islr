---
layout: page
title: 3. Linear Regression
---

# Exercise 9: Multiple regression of `mpg` on numerical features in `auto`

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-dataset" data-toc-modified-id="Preparing-the-dataset-1">Preparing the dataset</a></span></li><li><span><a href="#a.-Scatterplot-matrix-of-auto" data-toc-modified-id="a.-Scatterplot-matrix-of-auto-2">a. Scatterplot matrix of <code>auto</code></a></span></li><li><span><a href="#b-correlation-matrix-of-auto" data-toc-modified-id="b.-Correlation-matrix-of-auto-3">b. Correlation matrix of <code>auto</code></a></span></li><li><span><a href="#c-fitting-the-model" data-toc-modified-id="c.-Fitting-the-model-4">c. Fitting the model</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#i-is-there-a-relationship-between-the-predictors-and-the-mpg?" data-toc-modified-id="i.-Is-there-a-relationship-between-the-predictors-and-the-mpg?-4.0.1">i. Is there a relationship between the predictors and the <code>mpg</code>?</a></span></li><li><span><a href="#ii-which-predictors-appear-to-have-a-statistically-significant-relationship-to-the-response?" data-toc-modified-id="ii.-Which-predictors-appear-to-have-a-statistically-significant-relationship-to-the-response?-4.0.2">ii. Which predictors appear to have a statistically significant relationship to the response?</a></span></li><li><span><a href="#iii-what-does-the-coefficient-for-the-year-variable-suggest?" data-toc-modified-id="iii.-What-does-the-coefficient-for-the-year-variable-suggest?-4.0.3">iii. What does the coefficient for the year variable suggest?</a></span></li></ul></li></ul></li><li><span><a href="#d-diagnostic-plots" data-toc-modified-id="d.-Diagnostic-plots-5">d. Diagnostic plots</a></span><ul class="toc-item"><li><span><a href="#standardized-residuals-vs-fitted-value" data-toc-modified-id="Standardized-residuals-vs-fitted-value-5.1">Standardized residuals vs fitted value</a></span></li><li><span><a href="#standardized-residuals-QQ-plot" data-toc-modified-id="Standardized-residuals-QQ-plot-5.2">Standardized residuals QQ-plot</a></span></li><li><span><a href="#Scale-location-plot" data-toc-modified-id="Scale-location-plot-5.3">Scale-location plot</a></span></li><li><span><a href="#influence-plot" data-toc-modified-id="Influence-Plot-5.4">Influence Plot</a></span></li></ul></li><li><span><a href="#e-interaction-effects" data-toc-modified-id="e.-Interaction-effects-6">e. Interaction effects</a></span></li><li><span><a href="#f-variable-transformations" data-toc-modified-id="f.-Variable-transformations-7">f. Variable transformations</a></span><ul class="toc-item"><li><span><a href="#the-log-model" data-toc-modified-id="The-log-model-7.1">The $\log(X)$ model</a></span></li><li><span><a href="#the-square-root-model" data-toc-modified-id="The-square-root-model-7.2">The $\sqrt{X}$ model</a></span></li><li><span><a href="#the-squared-model" data-toc-modified-id="The-squared-model-7.3">The $X^2$ model</a></span></li></ul></li></ul></div>

{% katexmm %}
## Preparing the dataset

Import pandas, load the `Auto` dataset, and inspect


```python
import numpy as np
import pandas as pd

auto = pd.read_csv('../../datasets/Auto.csv')
auto.head()
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
      <th>Unnamed: 0</th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
auto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 392 entries, 0 to 391
    Data columns (total 10 columns):
    Unnamed: 0      392 non-null int64
    mpg             392 non-null float64
    cylinders       392 non-null int64
    displacement    392 non-null float64
    horsepower      392 non-null int64
    weight          392 non-null int64
    acceleration    392 non-null float64
    year            392 non-null int64
    origin          392 non-null int64
    name            392 non-null object
    dtypes: float64(3), int64(6), object(1)
    memory usage: 30.7+ KB


There are missing values represented by `'?'` in  `horsepower`. We'll impute these by using mean values for the `cylinders` class


```python
# replace `?` with nans
auto.loc[:, 'horsepower'].apply(lambda x: np.nan if x == '?' else x)

# cast horsepower to numeric dtype
auto.loc[:, 'horsepower'] = pd.to_numeric(auto.horsepower)

# now impute values
auto.loc[:, 'horsepower'] = auto.horsepower.fillna(auto.horsepower.mean())
```


```python
auto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 392 entries, 0 to 391
    Data columns (total 10 columns):
    Unnamed: 0      392 non-null int64
    mpg             392 non-null float64
    cylinders       392 non-null int64
    displacement    392 non-null float64
    horsepower      392 non-null int64
    weight          392 non-null int64
    acceleration    392 non-null float64
    year            392 non-null int64
    origin          392 non-null int64
    name            392 non-null object
    dtypes: float64(3), int64(6), object(1)
    memory usage: 30.7+ KB


## a. Scatterplot matrix of `auto`


```python
# setup
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn-white')
sns.set_style('white')
```


```python
sns.pairplot(auto.dropna())
```

![png]({{site.baseurl}}/assets/images/ch03_exercise_09_10_1.png)


## b. Correlation matrix of `auto`

Compute the correlation matrix of the numerical variables


```python
auto.corr()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.776260</td>
      <td>-0.804443</td>
      <td>-0.776230</td>
      <td>-0.831739</td>
      <td>0.422297</td>
      <td>0.581469</td>
      <td>0.563698</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.776260</td>
      <td>1.000000</td>
      <td>0.950920</td>
      <td>0.843640</td>
      <td>0.897017</td>
      <td>-0.504061</td>
      <td>-0.346717</td>
      <td>-0.564972</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>-0.804443</td>
      <td>0.950920</td>
      <td>1.000000</td>
      <td>0.897584</td>
      <td>0.933104</td>
      <td>-0.544162</td>
      <td>-0.369804</td>
      <td>-0.610664</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.776230</td>
      <td>0.843640</td>
      <td>0.897584</td>
      <td>1.000000</td>
      <td>0.864320</td>
      <td>-0.688223</td>
      <td>-0.415617</td>
      <td>-0.451925</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.831739</td>
      <td>0.897017</td>
      <td>0.933104</td>
      <td>0.864320</td>
      <td>1.000000</td>
      <td>-0.419502</td>
      <td>-0.307900</td>
      <td>-0.581265</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.422297</td>
      <td>-0.504061</td>
      <td>-0.544162</td>
      <td>-0.688223</td>
      <td>-0.419502</td>
      <td>1.000000</td>
      <td>0.282901</td>
      <td>0.210084</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.581469</td>
      <td>-0.346717</td>
      <td>-0.369804</td>
      <td>-0.415617</td>
      <td>-0.307900</td>
      <td>0.282901</td>
      <td>1.000000</td>
      <td>0.184314</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.563698</td>
      <td>-0.564972</td>
      <td>-0.610664</td>
      <td>-0.451925</td>
      <td>-0.581265</td>
      <td>0.210084</td>
      <td>0.184314</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## c. Fitting the model


```python
import statsmodels.api as sm

# drop non-numerical columns and rows with null entries
model_df = auto.drop(['name'], axis=1).dropna()
X, Y = model_df.drop(['mpg'], axis=1), model_df.mpg

# add constant
X = sm.add_constant(X)

# create and fit model
model = sm.OLS(Y, X).fit()

# show results summary
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.822</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.819</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   256.4</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 28 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>1.89e-141</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:28:06</td>     <th>  Log-Likelihood:    </th> <td> -1037.2</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   397</td>      <th>  AIC:               </th> <td>   2090.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   389</td>      <th>  BIC:               </th> <td>   2122.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>  -18.0900</td> <td>    4.629</td> <td>   -3.908</td> <td> 0.000</td> <td>  -27.191</td> <td>   -8.989</td>
</tr>
<tr>
  <th>cylinders</th>    <td>   -0.4560</td> <td>    0.322</td> <td>   -1.414</td> <td> 0.158</td> <td>   -1.090</td> <td>    0.178</td>
</tr>
<tr>
  <th>displacement</th> <td>    0.0196</td> <td>    0.008</td> <td>    2.608</td> <td> 0.009</td> <td>    0.005</td> <td>    0.034</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -0.0136</td> <td>    0.014</td> <td>   -0.993</td> <td> 0.321</td> <td>   -0.040</td> <td>    0.013</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.0066</td> <td>    0.001</td> <td>  -10.304</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.005</td>
</tr>
<tr>
  <th>acceleration</th> <td>    0.0998</td> <td>    0.098</td> <td>    1.021</td> <td> 0.308</td> <td>   -0.092</td> <td>    0.292</td>
</tr>
<tr>
  <th>year</th>         <td>    0.7587</td> <td>    0.051</td> <td>   14.969</td> <td> 0.000</td> <td>    0.659</td> <td>    0.858</td>
</tr>
<tr>
  <th>origin</th>       <td>    1.4199</td> <td>    0.277</td> <td>    5.132</td> <td> 0.000</td> <td>    0.876</td> <td>    1.964</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>30.088</td> <th>  Durbin-Watson:     </th> <td>   1.294</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  48.301</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.511</td> <th>  Prob(JB):          </th> <td>3.25e-11</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.370</td> <th>  Cond. No.          </th> <td>8.58e+04</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 8.58e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



#### i. Is there a relationship between the predictors and the `mpg`?

This question is answered by the $p$-value of the $F$-statistic


```python
model.f_pvalue
```




    1.8936359873496686e-141



This is effectively zero, so the answer is yes

#### ii. Which predictors appear to have a statistically significant relationship to the response?

This is answered by the $p$-values of the individual predictors


```python
model.pvalues
```




    const           1.097017e-04
    cylinders       1.580259e-01
    displacement    9.455004e-03
    horsepower      3.212038e-01
    weight          3.578587e-22
    acceleration    3.077592e-01
    year            2.502539e-40
    origin          4.530034e-07
    dtype: float64



A common cutoff is a $p$-value of 0.05, so by this standard, the predictors with a statistically significant relationship to `mpg` are


```python
is_stat_sig = model.pvalues < 0.05
model.pvalues[is_stat_sig]
```




    const           1.097017e-04
    displacement    9.455004e-03
    weight          3.578587e-22
    year            2.502539e-40
    origin          4.530034e-07
    dtype: float64



And those which do not are


```python
model.pvalues[~ is_stat_sig]
```




    cylinders       0.158026
    horsepower      0.321204
    acceleration    0.307759
    dtype: float64



This is surprising, since we found a statistically significant relationship between `horsepower` and `mpg` in [exercise 8]({{site.baseurl}}/ch03/ch03_exercises/ch03_exercise_08). 

#### iii. What does the coefficient for the year variable suggest?

That fuel efficiency has been improving over time

## d. Diagnostic plots

First we assemble the results in a dataframe and clean up a bit


```python
# get full prediction results
pred_df = model.get_prediction().summary_frame()

# rename columns to avoid `mean` name conflicts and other confusions
new_names = {}
for name in pred_df.columns:
    if 'mean' in name:
        new_names[name] = name.replace('mean', 'mpg_pred')
    elif 'obs_ci' in name:
        new_names[name] = name.replace('obs_ci', 'mpg_pred_pi')
    else:
        new_names[name] = name
pred_df = pred_df.rename(new_names, axis='columns')

# concat into final df
model_df = pd.concat([model_df, pred_df], axis=1)
```


```python
model_df.head()
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>mpg_pred</th>
      <th>mpg_pred_se</th>
      <th>mpg_pred_ci_lower</th>
      <th>mpg_pred_ci_upper</th>
      <th>mpg_pred_pi_lower</th>
      <th>mpg_pred_pi_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>14.966498</td>
      <td>0.506952</td>
      <td>13.969789</td>
      <td>15.963208</td>
      <td>8.338758</td>
      <td>21.594239</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>14.028743</td>
      <td>0.446127</td>
      <td>13.151621</td>
      <td>14.905865</td>
      <td>7.417930</td>
      <td>20.639557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>15.262507</td>
      <td>0.487309</td>
      <td>14.304418</td>
      <td>16.220595</td>
      <td>8.640464</td>
      <td>21.884549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>15.107684</td>
      <td>0.493468</td>
      <td>14.137487</td>
      <td>16.077882</td>
      <td>8.483879</td>
      <td>21.731490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>14.948273</td>
      <td>0.535264</td>
      <td>13.895900</td>
      <td>16.000646</td>
      <td>8.311933</td>
      <td>21.584612</td>
    </tr>
  </tbody>
</table>
</div>



Now we plot the 4 diagnostic plots returned by R's `lm()` function (see [[exercise 8]({{site.baseurl}}/ch03/ch03_exercises/ch03_exercise_08))

### Standardized residuals vs fitted value


```python
# add residuals to df
model_df['resid'] = model.resid


# plot
plt.ylabel('standardized resid')
sns.regplot(model_df.mpg_pred, model_df.resid/model_df.resid.std(), lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_09_31_1.png)


### Standardized residuals QQ-plot


```python
sm.qqplot(model_df.resid/model_df.resid.std(), color='grey', alpha=0.5, xlabel='')
plt.ylabel('studentized resid quantiles')
plt.xlabel('standard normal quantiles')
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_09_33_1.png)


### Scale-location plot


```python
plt.ylabel('√|standardized resid|')
sns.regplot(model_df.mpg_pred, np.sqrt(np.abs(model_df.resid/model_df.resid.std())), lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_09_35_1.png)


### Influence Plot


```python
# influence plot
axes = sns.regplot(model.get_influence().hat_matrix_diag, model_df.resid/model_df.resid.std(), 
            lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('leverage')
plt.ylabel('studentized resid')
x = np.linspace(0.01, axes.get_xlim()[1], 50)
plt.plot(x, np.sqrt(0.5*(1 - x)/x), color='red', linestyle='dashed')
plt.plot(x, np.sqrt((1 - x)/x), color='red', linestyle='dashed')
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_09_37_1.png)


From these diagnostic plots we conclude

- There is non-linearity in the data
- There are a handful of outliers (studentized residual $\geqslant$ 3)
- The normality assumption is appropriate
- The data shows heteroscedasticity
- There are no high influence points

## e. Interaction effects

We are told to use the `*` and `~` R operators to investigate interaction effects. Thankfully [statmodels has support for these](http://www.statsmodels.org/devel/example_formulas.html). 

To use `:
`, we will fit a model consisting of only pairwise interaction terms $X_iX_j$


```python
import itertools as it
import statsmodels.formula.api as smf


# generate formula for interaction terms 
names = list(auto.columns.drop('name').drop('mpg'))
pairs = list(it.product(names, names))
terms  = [name1 + ' : ' + name2 for (name1, name2) in pairs if name1 != name2]
formula = 'mpg ~ '

for term in terms:
    formula += term + ' + '
formula = formula[:-3]
formula
```




    'mpg ~ cylinders : displacement + cylinders : horsepower + cylinders : weight + cylinders : acceleration + cylinders : year + cylinders : origin + displacement : cylinders + displacement : horsepower + displacement : weight + displacement : acceleration + displacement : year + displacement : origin + horsepower : cylinders + horsepower : displacement + horsepower : weight + horsepower : acceleration + horsepower : year + horsepower : origin + weight : cylinders + weight : displacement + weight : horsepower + weight : acceleration + weight : year + weight : origin + acceleration : cylinders + acceleration : displacement + acceleration : horsepower + acceleration : weight + acceleration : year + acceleration : origin + year : cylinders + year : displacement + year : horsepower + year : weight + year : acceleration + year : origin + origin : cylinders + origin : displacement + origin : horsepower + origin : weight + origin : acceleration + origin : year'




```python
# fit a regression model with only interaction terms
pair_int_model = smf.ols(formula=formula, data=auto).fit()
```

And find the statisitcally significant interactions


```python
# show interactions with p value less than 0.05
pair_int_model.pvalues[pair_int_model.pvalues < 5e-2]
```




    Intercept                    0.005821
    cylinders:year               0.014595
    displacement:acceleration    0.010101
    displacement:year            0.000036
    displacement:origin          0.015855
    weight:acceleration          0.005177
    acceleration:year            0.000007
    year:origin                  0.045881
    dtype: float64



Now to use `+` we fit a model consisting of all features $X_i$ and all possible interactions 


```python
# generate formula for interaction terms 
names = list(auto.columns.drop('name').drop('mpg'))
formula = 'mpg ~ '
for name in names:
    formula += name + '*'
formula = formula[:-1]
formula
```




    'mpg ~ cylinders*displacement*horsepower*weight*acceleration*year*origin'




```python
# fit a regression model with all features and all possible interaction terms
full_int_model = smf.ols(formula=formula, data=auto).fit()
```

Finally, we find the statistically significant terms


```python
full_int_model.pvalues[full_int_model.pvalues < 0.05]
```




    Series([], dtype: float64)



In this case, including all possible interactions has led to none of them being statistically significant, even the pairwise interactions.

## f. Variable transformations

We'll try the suggested variable transformations $\log(X), \sqrt{X}, X^2$.


```python
# drop constant before transformation, else const for log(X) will be zero
X = X.drop('const', axis=1)
```


```python
import numpy as np
import statsmodels.api as sm

# transform data
log_X = np.log(X)
sqrt_X = np.sqrt(X)
X_sq = X**2

# fit models with constants
log_model = sm.OLS(Y, sm.add_constant(log_X)).fit()
sqrt_model = sm.OLS(Y, sm.add_constant(sqrt_X)).fit()
sq_model = sm.OLS(Y, sm.add_constant(X_sq)).fit()
```

Now we'll look at each of these models individually:

### The log model


```python
log_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.848</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.845</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   310.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 28 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>6.92e-155</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:28:07</td>     <th>  Log-Likelihood:    </th> <td> -1005.5</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   397</td>      <th>  AIC:               </th> <td>   2027.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   389</td>      <th>  BIC:               </th> <td>   2059.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>  -67.0838</td> <td>   17.433</td> <td>   -3.848</td> <td> 0.000</td> <td> -101.358</td> <td>  -32.810</td>
</tr>
<tr>
  <th>cylinders</th>    <td>    1.8114</td> <td>    1.658</td> <td>    1.093</td> <td> 0.275</td> <td>   -1.448</td> <td>    5.070</td>
</tr>
<tr>
  <th>displacement</th> <td>   -1.0935</td> <td>    1.540</td> <td>   -0.710</td> <td> 0.478</td> <td>   -4.121</td> <td>    1.934</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -6.2631</td> <td>    1.528</td> <td>   -4.100</td> <td> 0.000</td> <td>   -9.267</td> <td>   -3.259</td>
</tr>
<tr>
  <th>weight</th>       <td>  -13.4966</td> <td>    2.185</td> <td>   -6.178</td> <td> 0.000</td> <td>  -17.792</td> <td>   -9.201</td>
</tr>
<tr>
  <th>acceleration</th> <td>   -4.3687</td> <td>    1.577</td> <td>   -2.770</td> <td> 0.006</td> <td>   -7.469</td> <td>   -1.268</td>
</tr>
<tr>
  <th>year</th>         <td>   55.5963</td> <td>    3.540</td> <td>   15.704</td> <td> 0.000</td> <td>   48.636</td> <td>   62.557</td>
</tr>
<tr>
  <th>origin</th>       <td>    1.5763</td> <td>    0.506</td> <td>    3.118</td> <td> 0.002</td> <td>    0.582</td> <td>    2.570</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>39.413</td> <th>  Durbin-Watson:     </th> <td>   1.381</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  76.214</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.576</td> <th>  Prob(JB):          </th> <td>2.82e-17</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.812</td> <th>  Cond. No.          </th> <td>1.36e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.36e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Very large $R^2$ and very low $p$-value for the $F$-statistic suggest this is a useful model. Interestingly, this model gives very large $p$-values for the features `cylinders` and `displacement`.

The statistically significant features of the the original and log models and their p-values


```python
stat_sig_df = pd.concat([model.pvalues[is_stat_sig], log_model.pvalues[is_stat_sig]], join='outer', axis=1, sort=False)
stat_sig_df = stat_sig_df.rename({0 : 'model_pval', 1: 'log_model_pval'}, axis='columns')
stat_sig_df
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
      <th>model_pval</th>
      <th>log_model_pval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>1.097017e-04</td>
      <td>1.390585e-04</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>9.455004e-03</td>
      <td>4.780250e-01</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>3.578587e-22</td>
      <td>1.641584e-09</td>
    </tr>
    <tr>
      <th>year</th>
      <td>2.502539e-40</td>
      <td>2.150054e-43</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>4.530034e-07</td>
      <td>1.958604e-03</td>
    </tr>
  </tbody>
</table>
</div>



The insignificant features and p-values are


```python
stat_sig_df = pd.concat([model.pvalues[~ is_stat_sig], log_model.pvalues[~ is_stat_sig]], join='outer', axis=1, sort=False)
stat_sig_df = stat_sig_df.rename({0 : 'model_pval', 1: 'log_model_pval'}, axis='columns')
stat_sig_df
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
      <th>model_pval</th>
      <th>log_model_pval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cylinders</th>
      <td>0.158026</td>
      <td>0.275162</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0.321204</td>
      <td>0.000050</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.307759</td>
      <td>0.005869</td>
    </tr>
  </tbody>
</table>
</div>



So the original and log models are in total agreement about which features are significant! 

Let's look at prediction accuracy.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# split the data
X_train, X_test, y_train, y_test = train_test_split(auto.drop(['name', 'mpg'], axis=1).dropna(), auto.mpg)

# transform
log_X_train, log_X_test = np.log(X_train), np.log(X_test)

# train models
reg_model = LinearRegression().fit(X_train, y_train)
log_model = LinearRegression().fit(log_X_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
log_train_mse = mean_squared_error(y_train, log_model.predict(log_X_train))

print("The reg model train mse is {} and the log model train mse is {}".format(round(reg_train_mse, 3), round(log_train_mse, 3)))

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
log_test_mse = mean_squared_error(y_test, log_model.predict(log_X_test))

print("The reg model test mse is {} and the log model test mse is {}".format(round(reg_test_mse, 3), round(log_test_mse, 3)))
```

    The reg model train mse is 11.111 and the log model train mse is 9.496
    The reg model test mse is 10.434 and the log model test mse is 8.83


From a prediction standpoint, the $log(X)$ model is an improvement

### The square root model


```python
sqrt_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.834</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.831</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   279.5</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 28 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>1.76e-147</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:29:40</td>     <th>  Log-Likelihood:    </th> <td> -1023.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   397</td>      <th>  AIC:               </th> <td>   2062.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   389</td>      <th>  BIC:               </th> <td>   2094.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>  -51.9765</td> <td>    9.138</td> <td>   -5.688</td> <td> 0.000</td> <td>  -69.942</td> <td>  -34.011</td>
</tr>
<tr>
  <th>cylinders</th>    <td>   -0.0144</td> <td>    1.535</td> <td>   -0.009</td> <td> 0.993</td> <td>   -3.031</td> <td>    3.003</td>
</tr>
<tr>
  <th>displacement</th> <td>    0.2176</td> <td>    0.229</td> <td>    0.948</td> <td> 0.344</td> <td>   -0.234</td> <td>    0.669</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -0.6775</td> <td>    0.303</td> <td>   -2.233</td> <td> 0.026</td> <td>   -1.274</td> <td>   -0.081</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.6471</td> <td>    0.078</td> <td>   -8.323</td> <td> 0.000</td> <td>   -0.800</td> <td>   -0.494</td>
</tr>
<tr>
  <th>acceleration</th> <td>   -0.5983</td> <td>    0.821</td> <td>   -0.729</td> <td> 0.467</td> <td>   -2.212</td> <td>    1.016</td>
</tr>
<tr>
  <th>year</th>         <td>   12.9347</td> <td>    0.854</td> <td>   15.139</td> <td> 0.000</td> <td>   11.255</td> <td>   14.614</td>
</tr>
<tr>
  <th>origin</th>       <td>    3.2448</td> <td>    0.763</td> <td>    4.253</td> <td> 0.000</td> <td>    1.745</td> <td>    4.745</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>38.601</td> <th>  Durbin-Watson:     </th> <td>   1.306</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  69.511</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.589</td> <th>  Prob(JB):          </th> <td>8.05e-16</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.677</td> <th>  Cond. No.          </th> <td>3.30e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.3e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



The $R^2$ value is slightly less than for the log model, but not much, and the F-statistic p-value is comparable. 
This model doesn't like `cylinder` and `displacement` just like the regular and log models, but also rejects `acceleration`. 

Now we'll check prediction accuracy


```python
# transform
sqrt_X_train, sqrt_X_test = np.sqrt(X_train), np.sqrt(X_test)

# train sqrt model
sqrt_model = LinearRegression().fit(sqrt_X_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
sqrt_train_mse = mean_squared_error(y_train, sqrt_model.predict(sqrt_X_train))

print("The reg model train mse is {} and the sqrt model train mse is {}".format(round(reg_train_mse, 3), round(sqrt_train_mse, 3)))

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
sqrt_test_mse = mean_squared_error(y_test, sqrt_model.predict(sqrt_X_test))

print("The reg model test mse is {} and the sqrt model test mse is {}".format(round(reg_test_mse, 3), round(sqrt_test_mse, 3)))
```

    The reg model train mse is 11.111 and the sqrt model train mse is 10.365
    The reg model test mse is 10.434 and the sqrt model test mse is 9.635


Again, the $\sqrt{X}$ model is better at prediction

### The squared model


```python
sq_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.798</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.794</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   219.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 28 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>8.35e-131</td>
</tr>
<tr>
  <th>Time:</th>                 <td>19:38:32</td>     <th>  Log-Likelihood:    </th> <td> -1062.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   397</td>      <th>  AIC:               </th> <td>   2141.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   389</td>      <th>  BIC:               </th> <td>   2172.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>    0.9215</td> <td>    2.352</td> <td>    0.392</td> <td> 0.695</td> <td>   -3.702</td> <td>    5.546</td>
</tr>
<tr>
  <th>cylinders</th>    <td>   -0.0864</td> <td>    0.025</td> <td>   -3.431</td> <td> 0.001</td> <td>   -0.136</td> <td>   -0.037</td>
</tr>
<tr>
  <th>displacement</th> <td> 5.672e-05</td> <td> 1.39e-05</td> <td>    4.092</td> <td> 0.000</td> <td> 2.95e-05</td> <td>  8.4e-05</td>
</tr>
<tr>
  <th>horsepower</th>   <td>-2.945e-05</td> <td> 4.98e-05</td> <td>   -0.591</td> <td> 0.555</td> <td>   -0.000</td> <td> 6.85e-05</td>
</tr>
<tr>
  <th>weight</th>       <td>-9.535e-07</td> <td> 8.95e-08</td> <td>  -10.653</td> <td> 0.000</td> <td>-1.13e-06</td> <td>-7.77e-07</td>
</tr>
<tr>
  <th>acceleration</th> <td>    0.0066</td> <td>    0.003</td> <td>    2.466</td> <td> 0.014</td> <td>    0.001</td> <td>    0.012</td>
</tr>
<tr>
  <th>year</th>         <td>    0.0050</td> <td>    0.000</td> <td>   14.360</td> <td> 0.000</td> <td>    0.004</td> <td>    0.006</td>
</tr>
<tr>
  <th>origin</th>       <td>    0.4110</td> <td>    0.069</td> <td>    5.956</td> <td> 0.000</td> <td>    0.275</td> <td>    0.547</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>20.163</td> <th>  Durbin-Watson:     </th> <td>   1.296</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  27.033</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.421</td> <th>  Prob(JB):          </th> <td>1.35e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.961</td> <th>  Cond. No.          </th> <td>1.45e+08</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.45e+08. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



Slightly lower $R^2$ and higher $F$-statistic $p$-value than previous, but seems negligible (in all cases the $F$-statistic $p$-value is effectively zero)

Let's check prediction accuracy


```python
# transform
X_sq_train, X_sq_test = X_train**2, X_test**2

# train sqrt model
sq_model = LinearRegression().fit(X_sq_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
sq_train_mse = mean_squared_error(y_train, sq_model.predict(X_sq_train))

print("The reg model train mse is {} and the sq model train mse is {}".format(round(reg_train_mse, 3), round(sq_train_mse, 3)))

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
sq_test_mse = mean_squared_error(y_test, sq_model.predict(sq_X_test))

print("The reg model test mse is {} and the sq model test mse is {}".format(round(reg_test_mse, 3), round(sq_test_mse, 3)))
```

    The reg model train mse is 11.111 and the sq model train mse is 12.571
    The reg model test mse is 10.434 and the sq model test mse is 12.0


So the $X^2$ model is not as good at predicting as any of the other models

{% endkatexmm %}
