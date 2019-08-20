---
layout: page
title: 3. Linear Regression
---

# Exercise 8: Simple regression of `mpg` on `horsepower` in `auto` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-dataset" data-toc-modified-id="Preparing-the-dataset-1">Preparing the dataset</a></span></li><li><span><a href="#a-fitting-the-model" data-toc-modified-id="a.-Fitting-the-model-2">a. Fitting the model</a></span><ul class="toc-item"><li><span><a href="#i-is-there-a-relationship-between-horsepower-and-mpg" data-toc-modified-id="i.-Is-there-a-relationship-between-horsepower-and-mpg?-2.1">i. Is there a relationship between <code>horsepower</code> and <code>mpg</code>?</a></span></li><li><span><a href="#ii-how-strong-is-the-relationship" data-toc-modified-id="ii.-How-strong-is-the-relationship?-2.2">ii. How strong is the relationship?</a></span></li><li><span><a href="#iii-is-the-relationship-positive-or-negative" data-toc-modified-id="iii.-Is-the-relationship-positive-or-negative?-2.3">iii. Is the relationship positive or negative?</a></span></li><li><span><a href="#iv-what-is-the-predicted-mpg-associated-with-a-horsepower-of-98-what-are-the-associated-95-confidence-and-prediction-intervals" data-toc-modified-id="iv.-What-is-the-predicted-mpg-associated-with-a-horsepower-of-98?-What-are-the-associated-95-%-confidence-and-prediction-intervals?-2.4">iv. What is the predicted <code>mpg</code> associated with a <code>horsepower</code> of 98? What are the associated 95 % confidence and prediction intervals?</a></span></li></ul></li><li><span><a href="#b-scatterplot-and-least-squares-line-plot" data-toc-modified-id="b.-Scatterplot-and-least-squares-line-plot-3">b. Scatterplot and least squares line plot</a></span></li><li><span><a href="#c-diagnostic-plots" data-toc-modified-id="c-diagnostic-plots-4">c. Diagnostic plots</a></span><ul class="toc-item"><li><span><a href="#studentized-residuals-vs-fitted-plot" data-toc-modified-id="Studentized-Residuals-vs.-Fitted-plot-4.1">Studentized Residuals vs. Fitted plot</a></span></li><li><span><a href="#qq-plot-of-residuals" data-toc-modified-id="QQ-plot-of-Residuals-4.2">QQ-plot of Residuals</a></span></li><li><span><a href="#scale-location-plot" data-toc-modified-id="Scale-location-plot-4.3">Scale-location plot</a></span></li><li><span><a href="#influence-plot" data-toc-modified-id="Influence-Plot-4.4">Influence Plot</a></span></li></ul></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-5">Footnotes</a></span></li></ul></div>

## Preparing the dataset

Import pandas, load the `Auto` dataset, and inspect


```python
import pandas as pd

auto = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Auto.csv')
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
    RangeIndex: 397 entries, 0 to 396
    Data columns (total 9 columns):
    mpg             397 non-null float64
    cylinders       397 non-null int64
    displacement    397 non-null float64
    horsepower      397 non-null object
    weight          397 non-null int64
    acceleration    397 non-null float64
    year            397 non-null int64
    origin          397 non-null int64
    name            397 non-null object
    dtypes: float64(3), int64(4), object(2)
    memory usage: 28.0+ KB


All the dtypes look good except `horsepower`.


```python
auto.horsepower = pd.to_numeric(auto.horsepower, errors='coerce')
auto.horsepower.dtype
```




    dtype('float64')



##  a. Fitting the model

There are lots of way to [do simple linear regression with Python](https://medium.freecodecamp.org/data-science-with-python-8-ways-to-do-linear-regression-and-measure-their-speed-b5577d75f8b). For statistical analysis, `statsmodel` is useful.


```python
import statsmodels.api as sm
```


```python
# filter out null entries
X = auto.horsepower[auto.mpg.notna() & auto.horsepower.notna()]
Y = auto.mpg[auto.mpg.notna() & auto.horsepower.notna()]
X

# add constant
X = sm.add_constant(X)

# create and fit model
model = sm.OLS(Y, X)
model = model.fit()

# show results summary
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.606</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.605</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   599.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 27 Oct 2018</td> <th>  Prob (F-statistic):</th> <td>7.03e-81</td>
</tr>
<tr>
  <th>Time:</th>                 <td>08:23:54</td>     <th>  Log-Likelihood:    </th> <td> -1178.7</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2361.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   390</td>      <th>  BIC:               </th> <td>   2369.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>   39.9359</td> <td>    0.717</td> <td>   55.660</td> <td> 0.000</td> <td>   38.525</td> <td>   41.347</td>
</tr>
<tr>
  <th>horsepower</th> <td>   -0.1578</td> <td>    0.006</td> <td>  -24.489</td> <td> 0.000</td> <td>   -0.171</td> <td>   -0.145</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.432</td> <th>  Durbin-Watson:     </th> <td>   0.920</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  17.305</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.492</td> <th>  Prob(JB):          </th> <td>0.000175</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.299</td> <th>  Cond. No.          </th> <td>    322.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Now to answer the questions

### i. Is there a relationship between `horsepower` and `mpg`?

{% katexmm %}
This question is answered by testing the hypothesis

$$H_0: \beta_1 = 0$$
$$H_a: \beta_1 \neq 0$$

In the results summary table above, the value `P>|t|` in the row `horsepower` is the p-value for our hypothesis test. Since it's $< 0.5e-3$, we reject $H_0$ and conclude there is a relationship between `mpg` and `hp`
{% endkatexmm %}

### ii. How strong is the relationship?

{% katexmm %}
This question is answered by checking the $R^2$ value.


```python
model.rsquared
```




    0.6059482578894348



It's hard to interpret this based on the current state of my knowledge about the data. Interpretation is discussed on page 70 of the book, but it's not clear where this problem fits into that discussion. 

Given $\min(R^2) = 0$ indicates no relationship and $\max(R^2) = 1$ indicates a perfect (linear) relationship, I'll say this is a somewhat strong relationship.
{% endkatexmm %}


### iii. Is the relationship positive or negative?

{% katexmm %}
This is given by the sign of $\beta_1$


```python
model.params
```




    const         39.935861
    horsepower    -0.157845
    dtype: float64



Since $\beta_1 = -0.157845$, the relationship is negative
{% endkatexmm %}

### iv. What is the predicted `mpg` associated with a `horsepower` of 98? What are the associated 95/ confidence and prediction intervals?



```python
prediction = model.get_prediction([1, 98])
pred_df = prediction.summary_frame()
pred_df.head()
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
      <th>mean</th>
      <th>mean_se</th>
      <th>mean_ci_lower</th>
      <th>mean_ci_upper</th>
      <th>obs_ci_lower</th>
      <th>obs_ci_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.467077</td>
      <td>0.251262</td>
      <td>23.973079</td>
      <td>24.961075</td>
      <td>14.809396</td>
      <td>34.124758</td>
    </tr>
  </tbody>
</table>
</div>



The predicted value for `mpg`=98 is


```python
pred_df['mean']
```




    0    24.467077
    Name: mean, dtype: float64



The confidence interval is


```python
(pred_df['mean_ci_lower'].values[0], pred_df['mean_ci_upper'].values[0])
```




    (23.97307896070394, 24.961075344320907)



While the prediction interval is


```python
(pred_df['obs_ci_lower'].values[0], pred_df['obs_ci_upper'].values[0])
```




    (14.809396070967116, 34.12475823405773)



## b. Scatterplot and least squares line plot


```python
# setup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline
plt.style.use('seaborn-white')
sns.set_style('white')
```

For convenience, assemble the results in a new dataframe


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

# concat mpg, horsepower and prediction results in dataframe
model_df = pd.concat([X, Y, pred_df], axis=1)
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
      <th>const</th>
      <th>horsepower</th>
      <th>mpg</th>
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
      <td>1.0</td>
      <td>130.0</td>
      <td>18.0</td>
      <td>19.416046</td>
      <td>0.297444</td>
      <td>18.831250</td>
      <td>20.000841</td>
      <td>9.753295</td>
      <td>29.078797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>165.0</td>
      <td>15.0</td>
      <td>13.891480</td>
      <td>0.462181</td>
      <td>12.982802</td>
      <td>14.800158</td>
      <td>4.203732</td>
      <td>23.579228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>150.0</td>
      <td>18.0</td>
      <td>16.259151</td>
      <td>0.384080</td>
      <td>15.504025</td>
      <td>17.014277</td>
      <td>6.584598</td>
      <td>25.933704</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>150.0</td>
      <td>16.0</td>
      <td>16.259151</td>
      <td>0.384080</td>
      <td>15.504025</td>
      <td>17.014277</td>
      <td>6.584598</td>
      <td>25.933704</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>140.0</td>
      <td>17.0</td>
      <td>17.837598</td>
      <td>0.337403</td>
      <td>17.174242</td>
      <td>18.500955</td>
      <td>8.169775</td>
      <td>27.505422</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot
sns.scatterplot(model_df.horsepower, model_df.mpg, facecolors='grey', alpha=0.5)
sns.lineplot(model_df.horsepower, model_df.mpg_pred, color='r')
```

![png]({{site.baseurl}}/assets/images/ch03_exercise_08_27_1.png)


## c. Diagnostic plots 

This exercise uses R's [`plot()` function](https://www.rdocumentation.org/packages/graphics/versions/3.5.1/topics/plot), which by default returns [four diagnostic plots](https://data.library.virginia.edu/diagnostic-plots/). We'll recreate those plots in python [^1]

### Studentized Residuals vs. Fitted plot

This helps identify non-linearity


```python
# add studentized residuals to the dataframe
model_df['resid'] = model.resid

# studentized residuals vs. predicted values plot
sns.regplot(model_df.mpg_pred, model_df.resid/model_df.resid.std(), lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.ylabel('studentized resid')
```

    Text(0,0.5,'studentized resid')

![png]({{site.baseurl}}/assets/images/ch03_exercise_08_30_1.png)

This is a pretty clear indication of non-linearity (see p93) of text). We can also see some outliers

### QQ-plot of Residuals

This tests the assumption that the errors are normally distributed


```python
# plot standardized residuals against a standard normal distribution
sm.qqplot(model_df.resid/model_df.resid.std(), color='grey', alpha=0.5, xlabel='')
plt.ylabel('studentized resid quantiles')
plt.xlabel('standard normal quantiles')
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_08_33_1.png)


In this case there's good agreement with the normality assumption

### Scale-location plot

This tests the assumption of homoscedasticity (equal variance) of the errors


```python
sns.regplot(model_df.mpg_pred, np.sqrt(np.abs(model_df.resid/model_df.resid.std())), lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.ylabel('√|studentized resid|')
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_08_36_1.png)


In this case, the assumptions seems unjustified.

### Influence Plot

This helps identify influence points, i.e. points with an "outsize" effect on the model [^2]


```python
# scatterplot of leverage vs studentized residuals
axes = sns.regplot(model.get_influence().hat_matrix_diag, model_df.resid/model_df.resid.std(), 
            lowess=True, 
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('leverage')
plt.ylabel('studentized resid')

# plot Cook's distance contours for D = 0.5, D = 1
x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 50)
plt.plot(x, np.sqrt(0.5*(1 - x)/x), color='red', linestyle='dashed')
plt.plot(x, np.sqrt((1 - x)/x), color='red', linestyle='dashed')
```


![png]({{site.baseurl}}/assets/images/ch03_exercise_08_39_1.png)


No point in this plot has both high leverage and high residual, and all the points in this plot are within the Cook's distance contours, so we conclude that there are no high influence points

## Footnotes

[^1]: [This Medium article](https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034) addresses the same issue.

[^2]: [This Cross-Validated question](https://stats.stackexchange.com/questions/266597/plotting-cooks-distance-lines) was helpful in figuring out how to plot the Cook's distance.




