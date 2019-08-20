---
layout: page
title: 5. Resampling Methods
---

{% katexmm %}

# Exercise 6: Bootstrap estimates of standard errors of logistic regression coefficient estimates

<div class="toc"><ul class="toc-item"><li><span><a href="#get-estimates-of-standard-errors-from-statsmodels" data-toc-modified-id="Get-estimates-of-standard-errors-from-statsmodels-1">Get estimates of standard errors from <code>statsmodels</code></a></span></li><li><span><a href="#get-boostrap-estimates-of-standard-errors" data-toc-modified-id="Get-boostrap-estimates-of-standard-errors-2">Get boostrap estimates of standard errors</a></span></li></ul></div>

Given the differences between R and Python in this case, I'm not following the structure of this exercise

## Get estimates of standard errors from `statsmodels`


```python
import pandas as pd
import statsmodels.formula.api as smf


#import data
default = pd.read_csv("../../datasets/Default.csv", index_col=0)

# add constant
default['const'] = 1
columns = list(default.columns)
columns.remove('const')
default = default[['const'] + columns]

# convert to numeric
default['default'] = [int(value=='Yes') for value in default['default']]
default['student'] = [int(value=='Yes') for value in default['student']]

# fit model
logit = smf.logit(formula='default ~ income + balance', 
                  data=default).fit(disp=0)
```


```python
logit.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>default</td>     <th>  No. Observations:  </th>   <td> 10000</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>   <td>  9997</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>   <td>     2</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Mon, 03 Dec 2018</td> <th>  Pseudo R-squ.:     </th>   <td>0.4594</td>  
</tr>
<tr>
  <th>Time:</th>              <td>09:29:05</td>     <th>  Log-Likelihood:    </th>  <td> -789.48</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th>  <td> -1460.3</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>4.541e-292</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -11.5405</td> <td>    0.435</td> <td>  -26.544</td> <td> 0.000</td> <td>  -12.393</td> <td>  -10.688</td>
</tr>
<tr>
  <th>income</th>    <td> 2.081e-05</td> <td> 4.99e-06</td> <td>    4.174</td> <td> 0.000</td> <td>  1.1e-05</td> <td> 3.06e-05</td>
</tr>
<tr>
  <th>balance</th>   <td>    0.0056</td> <td>    0.000</td> <td>   24.835</td> <td> 0.000</td> <td>    0.005</td> <td>    0.006</td>
</tr>
</table><br/><br/>Possibly complete quasi-separation: A fraction 0.14 of observations can be<br/>perfectly predicted. This might indicate that there is complete<br/>quasi-separation. In this case some parameters will not be identified.



The estimated standard errors of the coefficient estimates are


```python
logit.bse
```




    Intercept    0.434772
    income       0.000005
    balance      0.000227
    dtype: float64



## Get boostrap estimates of standard errors


```python
from sklearn.utils import resample

boot_std_errs = {}
n_boot_samples = 1000

for i in range(n_boot_samples):
    default_boot_sample = resample(default)
    logit = smf.logit(formula='default ~ income + balance', 
                      data=default_boot_sample).fit(disp=0)
    boot_std_errs[i] = logit.bse
```


```python
df = pd.DataFrame.from_dict(boot_std_errs, orient='index')
df.head()
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
      <th>Intercept</th>
      <th>income</th>
      <th>balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.455075</td>
      <td>0.000005</td>
      <td>0.000242</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.486832</td>
      <td>0.000005</td>
      <td>0.000253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.454962</td>
      <td>0.000005</td>
      <td>0.000236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.440095</td>
      <td>0.000005</td>
      <td>0.000230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.420974</td>
      <td>0.000005</td>
      <td>0.000220</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.std()
```




    Intercept    2.213157e-02
    income       1.415595e-07
    balance      1.109741e-05
    dtype: float64



These estimates are considerably smaller, and likely more precise.

For more details, see the chapter on bootstrapping in [Wasserman's All of Statistics](http://www.stat.cmu.edu/~larry/all-of-statistics/)

{% endkatexmm %}
