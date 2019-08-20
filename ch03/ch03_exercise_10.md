---
layout: page
title: 3. Linear Regression
---

# Exercise 10: Multiple regression of `Sales` on `Price`, `Urban`, and `US ` features in `Carseats` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-dataset" data-toc-modified-id="Preparing-the-dataset-1">Preparing the dataset</a></span></li><li><span><a href="#a-fitting-the-model" data-toc-modified-id="a.-Fitting-the-model-2">a. Fitting the model</a></span></li><li><span><a href="#b-interpreting-the-coefficients" data-toc-modified-id="b.-Interpreting-the-coefficients-3">b. Interpreting the coefficients</a></span></li><li><span><a href="#c-the-formal-model" data-toc-modified-id="c.-The-formal-model-4">c. The formal model</a></span></li><li><span><a href="#d.-Significant-Variables" data-toc-modified-id="d.-Significant-Variables-5">d. Significant Variables</a></span></li><li><span><a href="#e-fitting-a-second-model-with-price,-US-features" data-toc-modified-id="e.-Fitting-a-second-model-with-price,-US-features-6">e. Fitting a second model with <code>price</code>, <code>US</code> features</a></span></li><li><span><a href="#f-goodness-of-fit-for-the-two-models" data-toc-modified-id="f.-Goodness-of-fit-for-the-two-models-7">f. Goodness-of-fit for the two models</a></span></li><li><span><a href="#g-confidence-intervals-for-coefficients-in-the-second-model" data-toc-modified-id="g.-Confidence-intervals-for-coefficients-in-the-second-model-8">g. Confidence intervals for coefficients in the second model</a></span></li><li><span><a href="#h-outliers-and-high-leverage-points-in-the-second-model" data-toc-modified-id="h.-Outliers-and-high-leverage-points-in-the-second-model-9">h. Outliers and high leverage points in the second model</a></span></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-10">Footnotes</a></span></li></ul></div>

## Preparing the dataset

{

The `Carseats` dataset is part of the `ISLR` R package. We'll use [`rpy2` to import `ISLR` into python](https://rpy2.readthedocs.io/en/version_2.8.x/introduction.html)


```python
import numpy as np
import pandas as pd

carseats = pd.read_csv('../../datasets/Carseats.csv')
```

We'll check for null entries


```python
carseats.isna().sum().sum()
```

    0



## a. Fitting the model

Both `Urban` and `US` are binary class variables, so we can represent both by indicator variables (see [section 3.3.1]({{site.baseurl}}/ch03/notes/#qualitative-predictors).

To encode the qualitative variables `Urban` and `US`, we'll use a `LabelEncoder` from `sklearn`


```python
from sklearn.preprocessing import LabelEncoder

# instantiate and encode labels
urban_le, us_le = LabelEncoder(), LabelEncoder()
urban_le.fit(carseats.Urban.unique())
us_le.fit(carseats.US.unique())
```




    LabelEncoder()




```python
urban_le.classes_
```




    array(['No', 'Yes'], dtype=object)



Now we'll create a dataframe with the qualitative variables numerically encoded


```python
# copy
carseats_enc = carseats.copy()

# transform columns
carseats_enc.loc[ :, 'Urban'] = urban_le.transform(carseats['Urban'])
carseats_enc.loc[ :, 'US'] = us_le.transform(carseats['US'])

carseats_enc[['Urban', 'US']].head()
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
      <th>Urban</th>
      <th>US</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now we can fit the model


```python
import statsmodels.formula.api as smf

model = smf.ols('Sales ~ Urban + US + Price', data=carseats_enc).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.239</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.234</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   41.52</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.39e-23</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:41:28</td>     <th>  Log-Likelihood:    </th> <td> -927.66</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   1863.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   396</td>      <th>  BIC:               </th> <td>   1879.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   13.0435</td> <td>    0.651</td> <td>   20.036</td> <td> 0.000</td> <td>   11.764</td> <td>   14.323</td>
</tr>
<tr>
  <th>Urban</th>     <td>   -0.0219</td> <td>    0.272</td> <td>   -0.081</td> <td> 0.936</td> <td>   -0.556</td> <td>    0.512</td>
</tr>
<tr>
  <th>US</th>        <td>    1.2006</td> <td>    0.259</td> <td>    4.635</td> <td> 0.000</td> <td>    0.691</td> <td>    1.710</td>
</tr>
<tr>
  <th>Price</th>     <td>   -0.0545</td> <td>    0.005</td> <td>  -10.389</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.044</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.676</td> <th>  Durbin-Watson:     </th> <td>   1.912</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.713</td> <th>  Jarque-Bera (JB):  </th> <td>   0.758</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.093</td> <th>  Prob(JB):          </th> <td>   0.684</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.897</td> <th>  Cond. No.          </th> <td>    628.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## b. Interpreting the coefficients


```python
model.params
```




    Intercept    13.043469
    Urban        -0.021916
    US            1.200573
    Price        -0.054459
    dtype: float64



`Carseats` has carseat sales data for 400 stores [^1]

can interpret these coefficients as follows

{% katexmm %}

1. The intercept $\hat{\beta}_0 = 13.04$ is the (estimated/predicted) average sales for carseats sold for free (`Price=0`) outside the US (`US='No'`) and outside a city (`Urban='No'`). 
2. $\hat{\beta}_0 + \hat{\beta}_1 = 13.02$ is the estimated average sales for carseats sold in cities outside the US
3. $\hat{\beta}_0 + \hat{\beta}_2 = 14.24$ is the estimated average sales for carseats sold outside cities in the US
4. $\hat{\beta}_0 + \hat{\beta}_1 + \hat{\beta}_2 = 14.22$ is the estimated average sales for carseats sold in cities in the US
5. $\hat{\beta}_3 = -0.05$ so a \$1 increase in price is estimated to decrease sales by $-0.05$
{% endkatexmm %}

## c. The formal model

{% katexmm %}
Here we write the model in equation form. Let $X_1$ be `Urban`, $X_2$ be `US`, $X_3$ be `Price`. Then the model is

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3
$$

For the $i$-th observation, we have
$$ 
\begin{aligned}
y_i &= \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3}\\
&= \begin{cases} \beta_0 + \beta_3 x_{i3} & \text{i-th carseat was sold outside a city, and outside the US}\\
\beta_0 + \beta_1 + \beta_3 x_{i3} & \text{i-th carseat was sold in a city outside the US}\\
\beta_0 + \beta_2 + \beta_3 x_{i3} & \text{i-th carseat was sold outside a city in the US}\\
\beta_0 + \beta_1 + \beta_2 + \beta_3 x_{i3} & \text{i-th carseat was sold in a US city}\\
\end{cases}
\end{aligned}
$$
{% endkatexmm %}

## d. Significant Variables


```python
is_stat_sig = model.pvalues < 0.05
model.pvalues[is_stat_sig]
```




    Intercept    3.626602e-62
    US           4.860245e-06
    Price        1.609917e-22
    dtype: float64


{% katexmm %}
So we reject the null hypothesis $H_0: \beta_i = 0$ for $i=2, 3$, that is for the variables `US` and `price`.


```python
model.pvalues[~ is_stat_sig]
```




    Urban    0.935739
    dtype: float64



 We fail to reject the null hypothesis for $i=1$, that is for the variables `Urban`
{% endkatexmm %}

## e. Fitting a second model with `price`, `US` features

On the basis of d., we fit a new model with only the `price` and `US` variables


```python
model_2 = smf.ols('Sales ~ US + Price', data=carseats_enc).fit()
model_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Sales</td>      <th>  R-squared:         </th> <td>   0.239</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.235</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   62.43</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 11 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>2.66e-24</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:41:28</td>     <th>  Log-Likelihood:    </th> <td> -927.66</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   400</td>      <th>  AIC:               </th> <td>   1861.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   397</td>      <th>  BIC:               </th> <td>   1873.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
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
  <th>Intercept</th> <td>   13.0308</td> <td>    0.631</td> <td>   20.652</td> <td> 0.000</td> <td>   11.790</td> <td>   14.271</td>
</tr>
<tr>
  <th>US</th>        <td>    1.1996</td> <td>    0.258</td> <td>    4.641</td> <td> 0.000</td> <td>    0.692</td> <td>    1.708</td>
</tr>
<tr>
  <th>Price</th>     <td>   -0.0545</td> <td>    0.005</td> <td>  -10.416</td> <td> 0.000</td> <td>   -0.065</td> <td>   -0.044</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.666</td> <th>  Durbin-Watson:     </th> <td>   1.912</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.717</td> <th>  Jarque-Bera (JB):  </th> <td>   0.749</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.092</td> <th>  Prob(JB):          </th> <td>   0.688</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.895</td> <th>  Cond. No.          </th> <td>    607.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## f. Goodness-of-fit for the two models

One way of determining how well the models fit the data is the $R^2$ values


```python
print("The first model has R^2 = {}".format(round(model.rsquared, 3)))
print("The second model has R^2 = {}".format(round(model_2.rsquared, 3)))
```

    The first model has R^2 = 0.239
    The second model has R^2 = 0.239

{% katexmm %}
So as far as the $R^2$ value is concerned, the models are equivalent. To choose one over the other, we might look at prediction accuracy, comparing mean squared errors, for example
{% endkatexmm %}

```python
print("The first model has model mean squared error {}".format(round(model.mse_model, 3)))
print("The first model has residual mean squared error {}".format(round(model.mse_resid, 3)))
print("The first model has total mean squared error {}".format(round(model.mse_total, 3)))
```

    The first model has model mean squared error 253.813
    The first model has residual mean squared error 6.113
    The first model has total mean squared error 7.976



```python
print("The second model has model mean squared error {}".format(round(model_2.mse_model, 3)))
print("The second model has residual mean squared error {}".format(round(model_2.mse_resid, 3)))
print("The second model has total mean squared error {}".format(round(model_2.mse_total, 3)))
```

    The second model has model mean squared error 380.7
    The second model has residual mean squared error 6.098
    The second model has total mean squared error 7.976



```python
print("The second model's model mse is {}% of the first".format(
      round(((model_2.mse_model - model.mse_model) * 100) / model.mse_model, 3)))
print("The second model's residual mse is {}% of the first".format(
      round(((model_2.mse_resid - model.mse_resid) * 100) / model.mse_resid, 3)))
print("The second model's total mse is {}% of the first".format(
      round(((model_2.mse_total - model.mse_total) * 100) / model.mse_total, 3)))
```

    The second model's model mse is 49.992% of the first
    The second model's residual mse is -0.25% of the first
    The second model's total mse is 0.0% of the first


The `statsmodel` documentation has [descriptions of these mean squared errors](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html)[^2]. 

- The model mean squared error is "the explained sum of squares divided by the model degrees of freedom" 
- The residual mean squared error is "the sum of squared residuals divided by the residual degrees of freedom" 
- The total mean squared error is "the uncentered total sum of squares divided by n the number of observations"

{% katexmm %}
In OLS multiple regression with $n$ observations and $p$ predictors, the number of model degrees of freedom are $p$, the number of residual degrees of freedom are $n - p - 1$, so we have

$$\mathbf{mse}_{model} = \frac{1}{p}\left(\sum_{i = 1}^n (\hat{y}_i - \overline{y})^2\right)$$
$$\mathbf{mse}_{resid} = \frac{1}{n - p - 1}\left(\sum_{i = 1}^n (\hat{y}_i - y_i)^2\right)$$
$$\mathbf{mse}_{total} = \frac{1}{n}\left(\sum_{i = 1}^n (y_i - \overline{y})^2\right)$$

From this we can draw some conclusions

- It's clear why $\mathbf{mse}_{total}$ for the two models in this problem
- The error $\mathbf{mse}_{resid}$ is slightly smaller ($0.25\%$) for the second model, so it's slightly at predicting the response.
- The error $\mathbf{mse}_{resid}$ is much larger ($50\%$) for the second model, so for this model the response predictions have a much greater deviation from average

From this perspective, the first model seems preferable, but it's difficult to say. The tiebreaker should come from their performance on test data.
{% endkatexmm %}

## g. Confidence intervals for coefficients in the second model


```python
model_2.conf_int()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>11.79032</td>
      <td>14.271265</td>
    </tr>
    <tr>
      <th>US</th>
      <td>0.69152</td>
      <td>1.707766</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>-0.06476</td>
      <td>-0.044195</td>
    </tr>
  </tbody>
</table>
</div>



## h. Outliers and high leverage points in the second model

To check for outliers we look at a standardized residuals vs fitted values plot


```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.regplot(model_2.get_prediction().summary_frame()['mean'], model_2.resid/model_2.resid.std(),
            lowess=True,
            line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'grey', 'edgecolors':'grey', 'alpha':0.4})
plt.xlabel('fitted values')
plt.ylabel('studentized resid')
```




    Text(0,0.5,'studentized resid')




![png]({{site.baseurl}}/assets/images/ch03_exercise_10_40_1.png)


No outliers -- all the studentized residual values are in the interval


```python
((model_2.resid/model_2.resid.std()).min(), (model_2.resid/model_2.resid.std()).max())
```




    (-2.812135116227541, 2.8627420897506766)



To check for high influence points we do an influence plot


```python
# scatterplot of leverage vs studentized residuals
axes = sns.regplot(model_2.get_influence().hat_matrix_diag, model_2.resid/model_2.resid.std(), 
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




    [<matplotlib.lines.Line2D at 0x1a181ff6a0>]




![png]({{site.baseurl}}/assets/images/ch03_exercise_10_44_1.png)


No high influence points. There is one rather high leverage point, but it has a low residual.

___
## Footnotes

{% katexmm %}
[^1]: For the authors' exploration of this dataset, see section 3.6.6 in the text 

[^2]: Related quantities in the book are ESS, RSS and TSS, respectively. The three are related by $TSS = ESS + RSS$. The book doesn't mention ESS explicitly, but $ESS = TSS - RSS$
{% endkatexmm %}

