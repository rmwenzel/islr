---
layout: page
title: 8. Tree-based Methods
---

{% katexmm %}

# Exercise 7: Plotting test error for parameter values of random forest model in chapter 8 lab

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#grid-search-for-random-forest-model" data-toc-modified-id="Grid-search-for-random-forest-model-2">Grid search for random forest model</a></span></li></ul></div>

## Preparing the data


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

boston = pd.read_csv('../../datasets/Boston.csv', index_col=0)
boston = boston.reset_index(drop=True)
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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
    RangeIndex: 506 entries, 0 to 505
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
    memory usage: 55.4 KB


## Grid search for random forest model

We're using [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) - the parameters `n_estimators`, `max_features` correspond to the parameters `ntree` and `mtry` respectively, for the `R` function `randomForest()`.


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

n_estimators, max_features = np.arange(1, 51), np.arange(1, 14)
params = {'n_estimators': n_estimators, 'max_features': max_features}
rf_search = GridSearchCV(RandomForestRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5)
rf_search.fit(boston.drop(columns=['medv']), boston['medv'])
```

    /anaconda3/envs/islr/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'n_estimators': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]), 'max_features': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

rf_search_results = pd.DataFrame(rf_search.cv_results_)
rf_search_results
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_features</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>...</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003891</td>
      <td>0.000834</td>
      <td>0.001949</td>
      <td>0.000402</td>
      <td>1</td>
      <td>1</td>
      <td>{'max_features': 1, 'n_estimators': 1}</td>
      <td>-21.991667</td>
      <td>-85.189010</td>
      <td>-54.552079</td>
      <td>...</td>
      <td>-64.012530</td>
      <td>23.806851</td>
      <td>648</td>
      <td>-14.603713</td>
      <td>-14.161111</td>
      <td>-11.699358</td>
      <td>-16.489481</td>
      <td>-12.779235</td>
      <td>-13.946580</td>
      <td>1.634436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.007109</td>
      <td>0.000531</td>
      <td>0.002835</td>
      <td>0.000614</td>
      <td>1</td>
      <td>2</td>
      <td>{'max_features': 1, 'n_estimators': 2}</td>
      <td>-23.077500</td>
      <td>-40.479134</td>
      <td>-41.309505</td>
      <td>...</td>
      <td>-49.661359</td>
      <td>20.626230</td>
      <td>646</td>
      <td>-9.532983</td>
      <td>-8.897994</td>
      <td>-4.985086</td>
      <td>-8.101160</td>
      <td>-8.354475</td>
      <td>-7.974340</td>
      <td>1.573450</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005619</td>
      <td>0.001508</td>
      <td>0.001752</td>
      <td>0.000513</td>
      <td>1</td>
      <td>3</td>
      <td>{'max_features': 1, 'n_estimators': 3}</td>
      <td>-17.650196</td>
      <td>-41.920968</td>
      <td>-80.304345</td>
      <td>...</td>
      <td>-45.140823</td>
      <td>21.318752</td>
      <td>642</td>
      <td>-7.537261</td>
      <td>-7.789177</td>
      <td>-6.061572</td>
      <td>-4.895781</td>
      <td>-7.208291</td>
      <td>-6.698416</td>
      <td>1.077818</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.006459</td>
      <td>0.001157</td>
      <td>0.001699</td>
      <td>0.000412</td>
      <td>1</td>
      <td>4</td>
      <td>{'max_features': 1, 'n_estimators': 4}</td>
      <td>-28.594032</td>
      <td>-42.909629</td>
      <td>-56.985149</td>
      <td>...</td>
      <td>-46.777108</td>
      <td>12.584626</td>
      <td>643</td>
      <td>-5.815023</td>
      <td>-5.466340</td>
      <td>-3.049759</td>
      <td>-5.406645</td>
      <td>-4.841636</td>
      <td>-4.915881</td>
      <td>0.983894</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.006406</td>
      <td>0.001049</td>
      <td>0.001832</td>
      <td>0.000544</td>
      <td>1</td>
      <td>5</td>
      <td>{'max_features': 1, 'n_estimators': 5}</td>
      <td>-14.134204</td>
      <td>-33.097687</td>
      <td>-35.283347</td>
      <td>...</td>
      <td>-35.554827</td>
      <td>12.541295</td>
      <td>621</td>
      <td>-5.153073</td>
      <td>-4.932412</td>
      <td>-3.802281</td>
      <td>-3.102355</td>
      <td>-3.784279</td>
      <td>-4.154880</td>
      <td>0.770769</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.007777</td>
      <td>0.001909</td>
      <td>0.001566</td>
      <td>0.000146</td>
      <td>1</td>
      <td>6</td>
      <td>{'max_features': 1, 'n_estimators': 6}</td>
      <td>-20.835833</td>
      <td>-33.876378</td>
      <td>-57.023809</td>
      <td>...</td>
      <td>-36.389669</td>
      <td>15.985771</td>
      <td>626</td>
      <td>-5.226819</td>
      <td>-4.592859</td>
      <td>-5.367281</td>
      <td>-3.909785</td>
      <td>-4.322368</td>
      <td>-4.683823</td>
      <td>0.547726</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.008977</td>
      <td>0.002770</td>
      <td>0.001802</td>
      <td>0.000529</td>
      <td>1</td>
      <td>7</td>
      <td>{'max_features': 1, 'n_estimators': 7}</td>
      <td>-15.424894</td>
      <td>-35.388693</td>
      <td>-68.895278</td>
      <td>...</td>
      <td>-40.056006</td>
      <td>17.818470</td>
      <td>638</td>
      <td>-3.979481</td>
      <td>-3.187319</td>
      <td>-3.898648</td>
      <td>-2.693557</td>
      <td>-3.768051</td>
      <td>-3.505411</td>
      <td>0.491660</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.011081</td>
      <td>0.001878</td>
      <td>0.001943</td>
      <td>0.000419</td>
      <td>1</td>
      <td>8</td>
      <td>{'max_features': 1, 'n_estimators': 8}</td>
      <td>-12.485312</td>
      <td>-21.791623</td>
      <td>-55.193929</td>
      <td>...</td>
      <td>-36.435462</td>
      <td>21.621571</td>
      <td>627</td>
      <td>-4.376785</td>
      <td>-3.511429</td>
      <td>-3.452217</td>
      <td>-2.734615</td>
      <td>-3.731063</td>
      <td>-3.561222</td>
      <td>0.527566</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.010636</td>
      <td>0.002615</td>
      <td>0.002022</td>
      <td>0.000730</td>
      <td>1</td>
      <td>9</td>
      <td>{'max_features': 1, 'n_estimators': 9}</td>
      <td>-18.031892</td>
      <td>-31.122899</td>
      <td>-37.199493</td>
      <td>...</td>
      <td>-36.325122</td>
      <td>13.528382</td>
      <td>625</td>
      <td>-3.032920</td>
      <td>-2.808331</td>
      <td>-2.432588</td>
      <td>-3.095907</td>
      <td>-4.087000</td>
      <td>-3.091350</td>
      <td>0.549331</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.012460</td>
      <td>0.002050</td>
      <td>0.002123</td>
      <td>0.000522</td>
      <td>1</td>
      <td>10</td>
      <td>{'max_features': 1, 'n_estimators': 10}</td>
      <td>-17.399393</td>
      <td>-33.092368</td>
      <td>-32.873599</td>
      <td>...</td>
      <td>-31.259992</td>
      <td>14.477833</td>
      <td>582</td>
      <td>-3.891032</td>
      <td>-3.469017</td>
      <td>-2.679415</td>
      <td>-3.634116</td>
      <td>-2.866854</td>
      <td>-3.308087</td>
      <td>0.460854</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.019396</td>
      <td>0.002247</td>
      <td>0.002984</td>
      <td>0.001326</td>
      <td>1</td>
      <td>11</td>
      <td>{'max_features': 1, 'n_estimators': 11}</td>
      <td>-11.739315</td>
      <td>-22.884135</td>
      <td>-53.257725</td>
      <td>...</td>
      <td>-34.255825</td>
      <td>18.556652</td>
      <td>612</td>
      <td>-2.524618</td>
      <td>-2.831083</td>
      <td>-3.511232</td>
      <td>-2.794624</td>
      <td>-3.029749</td>
      <td>-2.938261</td>
      <td>0.328599</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.013650</td>
      <td>0.002062</td>
      <td>0.002338</td>
      <td>0.000655</td>
      <td>1</td>
      <td>12</td>
      <td>{'max_features': 1, 'n_estimators': 12}</td>
      <td>-20.765519</td>
      <td>-29.219963</td>
      <td>-43.909655</td>
      <td>...</td>
      <td>-35.303845</td>
      <td>14.835744</td>
      <td>620</td>
      <td>-3.728071</td>
      <td>-2.489337</td>
      <td>-2.629168</td>
      <td>-2.102650</td>
      <td>-3.578967</td>
      <td>-2.905638</td>
      <td>0.636286</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.013520</td>
      <td>0.002860</td>
      <td>0.002294</td>
      <td>0.000803</td>
      <td>1</td>
      <td>13</td>
      <td>{'max_features': 1, 'n_estimators': 13}</td>
      <td>-18.235791</td>
      <td>-41.704494</td>
      <td>-28.564638</td>
      <td>...</td>
      <td>-33.328750</td>
      <td>14.136039</td>
      <td>602</td>
      <td>-3.033637</td>
      <td>-2.431979</td>
      <td>-2.363831</td>
      <td>-2.386989</td>
      <td>-2.513514</td>
      <td>-2.545990</td>
      <td>0.249125</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.014610</td>
      <td>0.002561</td>
      <td>0.002329</td>
      <td>0.000904</td>
      <td>1</td>
      <td>14</td>
      <td>{'max_features': 1, 'n_estimators': 14}</td>
      <td>-13.594839</td>
      <td>-20.757757</td>
      <td>-55.323856</td>
      <td>...</td>
      <td>-31.949899</td>
      <td>17.840135</td>
      <td>589</td>
      <td>-3.856282</td>
      <td>-2.813479</td>
      <td>-2.660921</td>
      <td>-2.461808</td>
      <td>-3.621529</td>
      <td>-3.082804</td>
      <td>0.552205</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.016239</td>
      <td>0.003550</td>
      <td>0.002126</td>
      <td>0.000099</td>
      <td>1</td>
      <td>15</td>
      <td>{'max_features': 1, 'n_estimators': 15}</td>
      <td>-15.587993</td>
      <td>-37.151651</td>
      <td>-39.838389</td>
      <td>...</td>
      <td>-37.000578</td>
      <td>14.092232</td>
      <td>629</td>
      <td>-2.979421</td>
      <td>-2.745065</td>
      <td>-2.955570</td>
      <td>-2.302839</td>
      <td>-2.104006</td>
      <td>-2.617380</td>
      <td>0.353338</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.021542</td>
      <td>0.007437</td>
      <td>0.004869</td>
      <td>0.001939</td>
      <td>1</td>
      <td>16</td>
      <td>{'max_features': 1, 'n_estimators': 16}</td>
      <td>-18.313434</td>
      <td>-37.697324</td>
      <td>-50.224384</td>
      <td>...</td>
      <td>-39.177959</td>
      <td>13.554225</td>
      <td>636</td>
      <td>-2.383039</td>
      <td>-2.313836</td>
      <td>-2.564468</td>
      <td>-2.085711</td>
      <td>-2.971439</td>
      <td>-2.463699</td>
      <td>0.296579</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.031045</td>
      <td>0.007147</td>
      <td>0.004051</td>
      <td>0.001197</td>
      <td>1</td>
      <td>17</td>
      <td>{'max_features': 1, 'n_estimators': 17}</td>
      <td>-14.477579</td>
      <td>-38.545166</td>
      <td>-47.593614</td>
      <td>...</td>
      <td>-36.993035</td>
      <td>14.366971</td>
      <td>628</td>
      <td>-3.384781</td>
      <td>-2.466205</td>
      <td>-2.349847</td>
      <td>-2.080384</td>
      <td>-2.440984</td>
      <td>-2.544440</td>
      <td>0.441862</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.020159</td>
      <td>0.003701</td>
      <td>0.002337</td>
      <td>0.000288</td>
      <td>1</td>
      <td>18</td>
      <td>{'max_features': 1, 'n_estimators': 18}</td>
      <td>-13.351092</td>
      <td>-29.703280</td>
      <td>-41.840159</td>
      <td>...</td>
      <td>-32.092468</td>
      <td>12.389404</td>
      <td>591</td>
      <td>-2.381153</td>
      <td>-2.327830</td>
      <td>-2.322180</td>
      <td>-2.293605</td>
      <td>-3.171681</td>
      <td>-2.499290</td>
      <td>0.337384</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.023308</td>
      <td>0.001765</td>
      <td>0.003537</td>
      <td>0.001120</td>
      <td>1</td>
      <td>19</td>
      <td>{'max_features': 1, 'n_estimators': 19}</td>
      <td>-15.882930</td>
      <td>-31.164088</td>
      <td>-39.008378</td>
      <td>...</td>
      <td>-34.578963</td>
      <td>14.486152</td>
      <td>616</td>
      <td>-3.857120</td>
      <td>-2.922748</td>
      <td>-2.321944</td>
      <td>-2.458541</td>
      <td>-3.067876</td>
      <td>-2.925646</td>
      <td>0.542314</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.024203</td>
      <td>0.002189</td>
      <td>0.003143</td>
      <td>0.000700</td>
      <td>1</td>
      <td>20</td>
      <td>{'max_features': 1, 'n_estimators': 20}</td>
      <td>-17.061018</td>
      <td>-37.885674</td>
      <td>-44.421256</td>
      <td>...</td>
      <td>-34.235834</td>
      <td>11.267077</td>
      <td>611</td>
      <td>-3.016849</td>
      <td>-2.616610</td>
      <td>-2.271411</td>
      <td>-1.724738</td>
      <td>-2.350052</td>
      <td>-2.395932</td>
      <td>0.424817</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.022982</td>
      <td>0.002748</td>
      <td>0.002810</td>
      <td>0.001135</td>
      <td>1</td>
      <td>21</td>
      <td>{'max_features': 1, 'n_estimators': 21}</td>
      <td>-17.608000</td>
      <td>-29.764712</td>
      <td>-57.100468</td>
      <td>...</td>
      <td>-34.166448</td>
      <td>15.051979</td>
      <td>610</td>
      <td>-2.849463</td>
      <td>-2.687234</td>
      <td>-2.648921</td>
      <td>-2.026325</td>
      <td>-2.457962</td>
      <td>-2.533981</td>
      <td>0.282744</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.025036</td>
      <td>0.002274</td>
      <td>0.003434</td>
      <td>0.001300</td>
      <td>1</td>
      <td>22</td>
      <td>{'max_features': 1, 'n_estimators': 22}</td>
      <td>-16.536106</td>
      <td>-26.593730</td>
      <td>-42.356686</td>
      <td>...</td>
      <td>-33.135153</td>
      <td>14.229106</td>
      <td>600</td>
      <td>-2.971610</td>
      <td>-2.912313</td>
      <td>-2.576134</td>
      <td>-2.170917</td>
      <td>-3.137634</td>
      <td>-2.753722</td>
      <td>0.343865</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.027406</td>
      <td>0.005331</td>
      <td>0.004077</td>
      <td>0.001138</td>
      <td>1</td>
      <td>23</td>
      <td>{'max_features': 1, 'n_estimators': 23}</td>
      <td>-15.404804</td>
      <td>-26.790607</td>
      <td>-47.250104</td>
      <td>...</td>
      <td>-33.781914</td>
      <td>15.178863</td>
      <td>605</td>
      <td>-2.764676</td>
      <td>-2.730681</td>
      <td>-2.743073</td>
      <td>-2.136093</td>
      <td>-3.241368</td>
      <td>-2.723178</td>
      <td>0.350817</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.024246</td>
      <td>0.004053</td>
      <td>0.002846</td>
      <td>0.000299</td>
      <td>1</td>
      <td>24</td>
      <td>{'max_features': 1, 'n_estimators': 24}</td>
      <td>-13.008028</td>
      <td>-36.532921</td>
      <td>-36.781510</td>
      <td>...</td>
      <td>-34.381056</td>
      <td>15.122934</td>
      <td>614</td>
      <td>-3.566207</td>
      <td>-2.849794</td>
      <td>-2.161991</td>
      <td>-2.167744</td>
      <td>-2.510485</td>
      <td>-2.651244</td>
      <td>0.523361</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.024699</td>
      <td>0.001386</td>
      <td>0.003098</td>
      <td>0.000754</td>
      <td>1</td>
      <td>25</td>
      <td>{'max_features': 1, 'n_estimators': 25}</td>
      <td>-17.054358</td>
      <td>-37.102535</td>
      <td>-43.966166</td>
      <td>...</td>
      <td>-34.298870</td>
      <td>14.970548</td>
      <td>613</td>
      <td>-2.938494</td>
      <td>-2.180039</td>
      <td>-2.440997</td>
      <td>-1.917603</td>
      <td>-2.551470</td>
      <td>-2.405720</td>
      <td>0.345116</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.027446</td>
      <td>0.005452</td>
      <td>0.003953</td>
      <td>0.001620</td>
      <td>1</td>
      <td>26</td>
      <td>{'max_features': 1, 'n_estimators': 26}</td>
      <td>-16.987042</td>
      <td>-34.845853</td>
      <td>-51.564609</td>
      <td>...</td>
      <td>-34.968050</td>
      <td>15.036683</td>
      <td>618</td>
      <td>-3.129059</td>
      <td>-2.227953</td>
      <td>-3.506445</td>
      <td>-2.467797</td>
      <td>-1.866639</td>
      <td>-2.639579</td>
      <td>0.597901</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.027815</td>
      <td>0.003477</td>
      <td>0.002695</td>
      <td>0.000157</td>
      <td>1</td>
      <td>27</td>
      <td>{'max_features': 1, 'n_estimators': 27}</td>
      <td>-14.762697</td>
      <td>-28.462289</td>
      <td>-39.627764</td>
      <td>...</td>
      <td>-29.153360</td>
      <td>10.051608</td>
      <td>567</td>
      <td>-3.074188</td>
      <td>-1.793045</td>
      <td>-3.078800</td>
      <td>-1.674099</td>
      <td>-1.989327</td>
      <td>-2.321892</td>
      <td>0.624303</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.028738</td>
      <td>0.006258</td>
      <td>0.002989</td>
      <td>0.000205</td>
      <td>1</td>
      <td>28</td>
      <td>{'max_features': 1, 'n_estimators': 28}</td>
      <td>-16.580555</td>
      <td>-38.412260</td>
      <td>-39.808376</td>
      <td>...</td>
      <td>-35.772924</td>
      <td>13.948740</td>
      <td>622</td>
      <td>-2.825889</td>
      <td>-2.647394</td>
      <td>-2.635796</td>
      <td>-2.118854</td>
      <td>-2.340015</td>
      <td>-2.513589</td>
      <td>0.251521</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.027386</td>
      <td>0.003200</td>
      <td>0.003024</td>
      <td>0.000625</td>
      <td>1</td>
      <td>29</td>
      <td>{'max_features': 1, 'n_estimators': 29}</td>
      <td>-16.863019</td>
      <td>-29.274269</td>
      <td>-45.390146</td>
      <td>...</td>
      <td>-34.144240</td>
      <td>11.450715</td>
      <td>609</td>
      <td>-2.444111</td>
      <td>-2.820349</td>
      <td>-2.001195</td>
      <td>-2.113187</td>
      <td>-2.791126</td>
      <td>-2.433993</td>
      <td>0.336793</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.028264</td>
      <td>0.004045</td>
      <td>0.003585</td>
      <td>0.000949</td>
      <td>1</td>
      <td>30</td>
      <td>{'max_features': 1, 'n_estimators': 30}</td>
      <td>-13.082333</td>
      <td>-26.423729</td>
      <td>-36.927213</td>
      <td>...</td>
      <td>-34.760669</td>
      <td>14.209880</td>
      <td>617</td>
      <td>-2.995265</td>
      <td>-2.269125</td>
      <td>-2.131816</td>
      <td>-2.082101</td>
      <td>-2.156179</td>
      <td>-2.326897</td>
      <td>0.339759</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>620</th>
      <td>0.056998</td>
      <td>0.002920</td>
      <td>0.002737</td>
      <td>0.000033</td>
      <td>13</td>
      <td>21</td>
      <td>{'max_features': 13, 'n_estimators': 21}</td>
      <td>-8.214533</td>
      <td>-14.830912</td>
      <td>-22.534429</td>
      <td>...</td>
      <td>-22.231679</td>
      <td>14.579162</td>
      <td>391</td>
      <td>-1.754569</td>
      <td>-2.005698</td>
      <td>-1.794253</td>
      <td>-1.358432</td>
      <td>-1.487365</td>
      <td>-1.680063</td>
      <td>0.230307</td>
    </tr>
    <tr>
      <th>621</th>
      <td>0.060251</td>
      <td>0.001883</td>
      <td>0.003669</td>
      <td>0.001008</td>
      <td>13</td>
      <td>22</td>
      <td>{'max_features': 13, 'n_estimators': 22}</td>
      <td>-7.563142</td>
      <td>-12.218573</td>
      <td>-24.094603</td>
      <td>...</td>
      <td>-21.770627</td>
      <td>14.921198</td>
      <td>345</td>
      <td>-1.771726</td>
      <td>-1.696460</td>
      <td>-1.623146</td>
      <td>-1.287195</td>
      <td>-1.534081</td>
      <td>-1.582522</td>
      <td>0.167329</td>
    </tr>
    <tr>
      <th>622</th>
      <td>0.061005</td>
      <td>0.001704</td>
      <td>0.003103</td>
      <td>0.000323</td>
      <td>13</td>
      <td>23</td>
      <td>{'max_features': 13, 'n_estimators': 23}</td>
      <td>-8.105958</td>
      <td>-13.928286</td>
      <td>-18.885602</td>
      <td>...</td>
      <td>-20.635471</td>
      <td>13.060317</td>
      <td>199</td>
      <td>-1.891957</td>
      <td>-1.411495</td>
      <td>-1.325187</td>
      <td>-1.296137</td>
      <td>-2.003337</td>
      <td>-1.585623</td>
      <td>0.300092</td>
    </tr>
    <tr>
      <th>623</th>
      <td>0.063468</td>
      <td>0.001463</td>
      <td>0.002937</td>
      <td>0.000097</td>
      <td>13</td>
      <td>24</td>
      <td>{'max_features': 13, 'n_estimators': 24}</td>
      <td>-7.601690</td>
      <td>-15.970749</td>
      <td>-21.354981</td>
      <td>...</td>
      <td>-22.495433</td>
      <td>13.645004</td>
      <td>408</td>
      <td>-1.731364</td>
      <td>-1.863511</td>
      <td>-1.403066</td>
      <td>-1.260503</td>
      <td>-1.713268</td>
      <td>-1.594342</td>
      <td>0.225125</td>
    </tr>
    <tr>
      <th>624</th>
      <td>0.066116</td>
      <td>0.001926</td>
      <td>0.003166</td>
      <td>0.000096</td>
      <td>13</td>
      <td>25</td>
      <td>{'max_features': 13, 'n_estimators': 25}</td>
      <td>-8.235568</td>
      <td>-11.828849</td>
      <td>-21.175247</td>
      <td>...</td>
      <td>-21.176898</td>
      <td>12.580003</td>
      <td>261</td>
      <td>-1.836137</td>
      <td>-1.731191</td>
      <td>-1.778248</td>
      <td>-1.423419</td>
      <td>-1.764215</td>
      <td>-1.706642</td>
      <td>0.145622</td>
    </tr>
    <tr>
      <th>625</th>
      <td>0.071170</td>
      <td>0.005523</td>
      <td>0.003244</td>
      <td>0.000451</td>
      <td>13</td>
      <td>26</td>
      <td>{'max_features': 13, 'n_estimators': 26}</td>
      <td>-8.400417</td>
      <td>-12.670221</td>
      <td>-23.192632</td>
      <td>...</td>
      <td>-21.533916</td>
      <td>12.765942</td>
      <td>310</td>
      <td>-2.220595</td>
      <td>-2.052685</td>
      <td>-1.643289</td>
      <td>-1.201552</td>
      <td>-1.447342</td>
      <td>-1.713092</td>
      <td>0.376842</td>
    </tr>
    <tr>
      <th>626</th>
      <td>0.070533</td>
      <td>0.001690</td>
      <td>0.003245</td>
      <td>0.000122</td>
      <td>13</td>
      <td>27</td>
      <td>{'max_features': 13, 'n_estimators': 27}</td>
      <td>-8.244153</td>
      <td>-11.947600</td>
      <td>-19.265464</td>
      <td>...</td>
      <td>-21.861377</td>
      <td>13.279718</td>
      <td>356</td>
      <td>-1.534273</td>
      <td>-1.521596</td>
      <td>-1.813661</td>
      <td>-1.328404</td>
      <td>-2.152099</td>
      <td>-1.670007</td>
      <td>0.286423</td>
    </tr>
    <tr>
      <th>627</th>
      <td>0.074091</td>
      <td>0.003326</td>
      <td>0.003271</td>
      <td>0.000304</td>
      <td>13</td>
      <td>28</td>
      <td>{'max_features': 13, 'n_estimators': 28}</td>
      <td>-8.637694</td>
      <td>-14.060935</td>
      <td>-25.394871</td>
      <td>...</td>
      <td>-22.784116</td>
      <td>13.353693</td>
      <td>424</td>
      <td>-1.669801</td>
      <td>-1.748181</td>
      <td>-1.623929</td>
      <td>-1.392020</td>
      <td>-1.375407</td>
      <td>-1.561868</td>
      <td>0.150884</td>
    </tr>
    <tr>
      <th>628</th>
      <td>0.076364</td>
      <td>0.002998</td>
      <td>0.003175</td>
      <td>0.000052</td>
      <td>13</td>
      <td>29</td>
      <td>{'max_features': 13, 'n_estimators': 29}</td>
      <td>-8.121697</td>
      <td>-14.794659</td>
      <td>-19.478598</td>
      <td>...</td>
      <td>-21.329425</td>
      <td>12.810383</td>
      <td>278</td>
      <td>-1.664764</td>
      <td>-1.473133</td>
      <td>-1.515179</td>
      <td>-1.250855</td>
      <td>-1.580039</td>
      <td>-1.496794</td>
      <td>0.138944</td>
    </tr>
    <tr>
      <th>629</th>
      <td>0.080446</td>
      <td>0.002324</td>
      <td>0.003275</td>
      <td>0.000146</td>
      <td>13</td>
      <td>30</td>
      <td>{'max_features': 13, 'n_estimators': 30}</td>
      <td>-8.327223</td>
      <td>-13.938036</td>
      <td>-18.504142</td>
      <td>...</td>
      <td>-21.177455</td>
      <td>12.801173</td>
      <td>262</td>
      <td>-2.062341</td>
      <td>-1.929879</td>
      <td>-1.695927</td>
      <td>-1.258211</td>
      <td>-2.019631</td>
      <td>-1.793198</td>
      <td>0.295995</td>
    </tr>
    <tr>
      <th>630</th>
      <td>0.082867</td>
      <td>0.004171</td>
      <td>0.003718</td>
      <td>0.000470</td>
      <td>13</td>
      <td>31</td>
      <td>{'max_features': 13, 'n_estimators': 31}</td>
      <td>-7.962718</td>
      <td>-13.002553</td>
      <td>-23.942338</td>
      <td>...</td>
      <td>-22.044556</td>
      <td>12.910633</td>
      <td>375</td>
      <td>-1.644527</td>
      <td>-1.880164</td>
      <td>-1.529467</td>
      <td>-1.261535</td>
      <td>-1.878561</td>
      <td>-1.638851</td>
      <td>0.232404</td>
    </tr>
    <tr>
      <th>631</th>
      <td>0.094475</td>
      <td>0.003793</td>
      <td>0.003925</td>
      <td>0.000547</td>
      <td>13</td>
      <td>32</td>
      <td>{'max_features': 13, 'n_estimators': 32}</td>
      <td>-8.576633</td>
      <td>-12.183874</td>
      <td>-21.498011</td>
      <td>...</td>
      <td>-21.738878</td>
      <td>12.744928</td>
      <td>341</td>
      <td>-2.025376</td>
      <td>-1.560575</td>
      <td>-1.220283</td>
      <td>-1.242486</td>
      <td>-2.247338</td>
      <td>-1.659212</td>
      <td>0.413767</td>
    </tr>
    <tr>
      <th>632</th>
      <td>0.086008</td>
      <td>0.003725</td>
      <td>0.003554</td>
      <td>0.000156</td>
      <td>13</td>
      <td>33</td>
      <td>{'max_features': 13, 'n_estimators': 33}</td>
      <td>-8.437380</td>
      <td>-12.298403</td>
      <td>-20.587260</td>
      <td>...</td>
      <td>-22.134070</td>
      <td>13.272654</td>
      <td>382</td>
      <td>-1.808651</td>
      <td>-1.573553</td>
      <td>-1.464411</td>
      <td>-1.280631</td>
      <td>-1.754965</td>
      <td>-1.576442</td>
      <td>0.192798</td>
    </tr>
    <tr>
      <th>633</th>
      <td>0.088431</td>
      <td>0.003652</td>
      <td>0.003595</td>
      <td>0.000137</td>
      <td>13</td>
      <td>34</td>
      <td>{'max_features': 13, 'n_estimators': 34}</td>
      <td>-7.789550</td>
      <td>-13.488061</td>
      <td>-21.859775</td>
      <td>...</td>
      <td>-21.861308</td>
      <td>12.894215</td>
      <td>355</td>
      <td>-1.679846</td>
      <td>-1.667615</td>
      <td>-1.791897</td>
      <td>-1.321465</td>
      <td>-1.660280</td>
      <td>-1.624221</td>
      <td>0.158779</td>
    </tr>
    <tr>
      <th>634</th>
      <td>0.091344</td>
      <td>0.002211</td>
      <td>0.003711</td>
      <td>0.000252</td>
      <td>13</td>
      <td>35</td>
      <td>{'max_features': 13, 'n_estimators': 35}</td>
      <td>-8.359150</td>
      <td>-13.595726</td>
      <td>-22.272514</td>
      <td>...</td>
      <td>-22.603360</td>
      <td>14.050540</td>
      <td>412</td>
      <td>-1.726925</td>
      <td>-1.674344</td>
      <td>-1.427541</td>
      <td>-1.330239</td>
      <td>-1.826045</td>
      <td>-1.597019</td>
      <td>0.187191</td>
    </tr>
    <tr>
      <th>635</th>
      <td>0.093527</td>
      <td>0.001836</td>
      <td>0.003673</td>
      <td>0.000136</td>
      <td>13</td>
      <td>36</td>
      <td>{'max_features': 13, 'n_estimators': 36}</td>
      <td>-8.207619</td>
      <td>-11.886666</td>
      <td>-20.280017</td>
      <td>...</td>
      <td>-21.392191</td>
      <td>13.865106</td>
      <td>291</td>
      <td>-1.981380</td>
      <td>-1.686528</td>
      <td>-1.272086</td>
      <td>-1.248820</td>
      <td>-1.550914</td>
      <td>-1.547946</td>
      <td>0.273003</td>
    </tr>
    <tr>
      <th>636</th>
      <td>0.097891</td>
      <td>0.003279</td>
      <td>0.003842</td>
      <td>0.000363</td>
      <td>13</td>
      <td>37</td>
      <td>{'max_features': 13, 'n_estimators': 37}</td>
      <td>-8.413852</td>
      <td>-13.636418</td>
      <td>-18.461581</td>
      <td>...</td>
      <td>-21.609566</td>
      <td>14.287871</td>
      <td>318</td>
      <td>-1.748631</td>
      <td>-1.972837</td>
      <td>-1.519343</td>
      <td>-1.374962</td>
      <td>-1.448768</td>
      <td>-1.612908</td>
      <td>0.219219</td>
    </tr>
    <tr>
      <th>637</th>
      <td>0.104330</td>
      <td>0.014696</td>
      <td>0.003811</td>
      <td>0.000158</td>
      <td>13</td>
      <td>38</td>
      <td>{'max_features': 13, 'n_estimators': 38}</td>
      <td>-8.398189</td>
      <td>-12.856086</td>
      <td>-20.030461</td>
      <td>...</td>
      <td>-21.481719</td>
      <td>13.813047</td>
      <td>303</td>
      <td>-1.703596</td>
      <td>-1.638732</td>
      <td>-1.540393</td>
      <td>-1.363200</td>
      <td>-1.611878</td>
      <td>-1.571560</td>
      <td>0.116582</td>
    </tr>
    <tr>
      <th>638</th>
      <td>0.106177</td>
      <td>0.002897</td>
      <td>0.004407</td>
      <td>0.000388</td>
      <td>13</td>
      <td>39</td>
      <td>{'max_features': 13, 'n_estimators': 39}</td>
      <td>-8.954183</td>
      <td>-13.575304</td>
      <td>-21.071506</td>
      <td>...</td>
      <td>-21.889710</td>
      <td>12.731358</td>
      <td>358</td>
      <td>-1.499969</td>
      <td>-1.430730</td>
      <td>-1.587401</td>
      <td>-1.257869</td>
      <td>-1.530888</td>
      <td>-1.461371</td>
      <td>0.113629</td>
    </tr>
    <tr>
      <th>639</th>
      <td>0.139257</td>
      <td>0.021265</td>
      <td>0.005298</td>
      <td>0.000931</td>
      <td>13</td>
      <td>40</td>
      <td>{'max_features': 13, 'n_estimators': 40}</td>
      <td>-8.828245</td>
      <td>-15.956237</td>
      <td>-19.494517</td>
      <td>...</td>
      <td>-22.224991</td>
      <td>13.058084</td>
      <td>389</td>
      <td>-1.680124</td>
      <td>-1.453758</td>
      <td>-1.493468</td>
      <td>-1.233556</td>
      <td>-1.403723</td>
      <td>-1.452926</td>
      <td>0.144089</td>
    </tr>
    <tr>
      <th>640</th>
      <td>0.122522</td>
      <td>0.019294</td>
      <td>0.004169</td>
      <td>0.000311</td>
      <td>13</td>
      <td>41</td>
      <td>{'max_features': 13, 'n_estimators': 41}</td>
      <td>-8.139146</td>
      <td>-15.458479</td>
      <td>-21.882525</td>
      <td>...</td>
      <td>-21.569182</td>
      <td>13.141029</td>
      <td>313</td>
      <td>-1.855412</td>
      <td>-1.641777</td>
      <td>-1.475673</td>
      <td>-1.209853</td>
      <td>-1.872287</td>
      <td>-1.611001</td>
      <td>0.248269</td>
    </tr>
    <tr>
      <th>641</th>
      <td>0.113846</td>
      <td>0.005494</td>
      <td>0.004137</td>
      <td>0.000280</td>
      <td>13</td>
      <td>42</td>
      <td>{'max_features': 13, 'n_estimators': 42}</td>
      <td>-8.037270</td>
      <td>-12.633225</td>
      <td>-26.076467</td>
      <td>...</td>
      <td>-22.908325</td>
      <td>14.112196</td>
      <td>430</td>
      <td>-1.897440</td>
      <td>-1.608781</td>
      <td>-1.485819</td>
      <td>-1.123077</td>
      <td>-1.209502</td>
      <td>-1.464924</td>
      <td>0.279392</td>
    </tr>
    <tr>
      <th>642</th>
      <td>0.110486</td>
      <td>0.002739</td>
      <td>0.004201</td>
      <td>0.000132</td>
      <td>13</td>
      <td>43</td>
      <td>{'max_features': 13, 'n_estimators': 43}</td>
      <td>-7.738024</td>
      <td>-11.768356</td>
      <td>-23.203926</td>
      <td>...</td>
      <td>-22.395667</td>
      <td>13.987355</td>
      <td>399</td>
      <td>-1.503985</td>
      <td>-1.495877</td>
      <td>-1.375177</td>
      <td>-1.422425</td>
      <td>-1.366992</td>
      <td>-1.432891</td>
      <td>0.057973</td>
    </tr>
    <tr>
      <th>643</th>
      <td>0.111097</td>
      <td>0.002981</td>
      <td>0.004338</td>
      <td>0.000172</td>
      <td>13</td>
      <td>44</td>
      <td>{'max_features': 13, 'n_estimators': 44}</td>
      <td>-7.885614</td>
      <td>-13.315915</td>
      <td>-20.279256</td>
      <td>...</td>
      <td>-21.864941</td>
      <td>13.759772</td>
      <td>357</td>
      <td>-1.731070</td>
      <td>-1.665641</td>
      <td>-1.559994</td>
      <td>-1.225409</td>
      <td>-1.964766</td>
      <td>-1.629376</td>
      <td>0.241721</td>
    </tr>
    <tr>
      <th>644</th>
      <td>0.114695</td>
      <td>0.002069</td>
      <td>0.004231</td>
      <td>0.000199</td>
      <td>13</td>
      <td>45</td>
      <td>{'max_features': 13, 'n_estimators': 45}</td>
      <td>-8.522087</td>
      <td>-13.834266</td>
      <td>-19.959325</td>
      <td>...</td>
      <td>-21.711338</td>
      <td>13.139111</td>
      <td>336</td>
      <td>-1.752770</td>
      <td>-1.384300</td>
      <td>-1.472556</td>
      <td>-1.272192</td>
      <td>-1.593057</td>
      <td>-1.494975</td>
      <td>0.166411</td>
    </tr>
    <tr>
      <th>645</th>
      <td>0.124314</td>
      <td>0.010494</td>
      <td>0.004609</td>
      <td>0.000780</td>
      <td>13</td>
      <td>46</td>
      <td>{'max_features': 13, 'n_estimators': 46}</td>
      <td>-9.244762</td>
      <td>-13.533141</td>
      <td>-20.165793</td>
      <td>...</td>
      <td>-21.277579</td>
      <td>11.760141</td>
      <td>275</td>
      <td>-1.890338</td>
      <td>-1.523985</td>
      <td>-1.429138</td>
      <td>-1.260989</td>
      <td>-1.434726</td>
      <td>-1.507835</td>
      <td>0.209304</td>
    </tr>
    <tr>
      <th>646</th>
      <td>0.125519</td>
      <td>0.003614</td>
      <td>0.004486</td>
      <td>0.000513</td>
      <td>13</td>
      <td>47</td>
      <td>{'max_features': 13, 'n_estimators': 47}</td>
      <td>-7.523758</td>
      <td>-14.241435</td>
      <td>-23.102508</td>
      <td>...</td>
      <td>-22.140001</td>
      <td>13.534116</td>
      <td>383</td>
      <td>-1.688069</td>
      <td>-1.446108</td>
      <td>-1.646988</td>
      <td>-1.208258</td>
      <td>-1.689223</td>
      <td>-1.535729</td>
      <td>0.186770</td>
    </tr>
    <tr>
      <th>647</th>
      <td>0.134769</td>
      <td>0.006030</td>
      <td>0.004768</td>
      <td>0.000579</td>
      <td>13</td>
      <td>48</td>
      <td>{'max_features': 13, 'n_estimators': 48}</td>
      <td>-7.969312</td>
      <td>-12.855202</td>
      <td>-22.178862</td>
      <td>...</td>
      <td>-21.676694</td>
      <td>13.110346</td>
      <td>331</td>
      <td>-1.810569</td>
      <td>-1.547392</td>
      <td>-1.517920</td>
      <td>-1.351242</td>
      <td>-1.432992</td>
      <td>-1.532023</td>
      <td>0.155249</td>
    </tr>
    <tr>
      <th>648</th>
      <td>0.130817</td>
      <td>0.008154</td>
      <td>0.005051</td>
      <td>0.000525</td>
      <td>13</td>
      <td>49</td>
      <td>{'max_features': 13, 'n_estimators': 49}</td>
      <td>-8.560279</td>
      <td>-12.427586</td>
      <td>-21.365338</td>
      <td>...</td>
      <td>-21.529505</td>
      <td>12.947968</td>
      <td>307</td>
      <td>-1.643209</td>
      <td>-1.597696</td>
      <td>-1.413012</td>
      <td>-1.193954</td>
      <td>-1.778295</td>
      <td>-1.525233</td>
      <td>0.202755</td>
    </tr>
    <tr>
      <th>649</th>
      <td>0.137780</td>
      <td>0.003193</td>
      <td>0.004835</td>
      <td>0.000229</td>
      <td>13</td>
      <td>50</td>
      <td>{'max_features': 13, 'n_estimators': 50}</td>
      <td>-8.213808</td>
      <td>-13.037796</td>
      <td>-18.630593</td>
      <td>...</td>
      <td>-21.511986</td>
      <td>13.648508</td>
      <td>306</td>
      <td>-1.693073</td>
      <td>-1.399460</td>
      <td>-1.560297</td>
      <td>-1.264475</td>
      <td>-1.516853</td>
      <td>-1.486832</td>
      <td>0.145540</td>
    </tr>
  </tbody>
</table>
<p>650 rows × 22 columns</p>
</div>




```python
from mpl_toolkits import mplot3d

n_estimators, max_features = np.arange(1, 51), np.arange(1, 14)
X, Y = np.meshgrid(max_features, n_estimators)
Z = scores = np.sqrt(-rf_search_results[['mean_test_score']].values).reshape(50, 13)

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='Greys', edgecolor='none')
ax.set_xlabel('n_estimators')
ax.set_ylabel('max_features')
ax.set_zlabel('rmse');
ax.view_init(60, 35)
```


![png]({{site.baseurl}}/assets/images/ch08_exercise_07_8_0.png)



```python
rf_search.best_params_
```




    {'max_features': 8, 'n_estimators': 19}




```python
np.sqrt(-rf_search.best_score_)
```




    4.198781714064752


{% endkatexmm %}