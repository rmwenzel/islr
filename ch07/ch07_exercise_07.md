
# Using non-linear multiple regression to predict `wage` in `Wage` dataset

We're modifying the exercise a bit to consider multiple regression (as opposed to considering different predictors individually). It's not hard to see how the techniques of this chapter generalize to the multiple regression setting.

## Preparing the data

### Loading


```python
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

wage = pd.read_csv("../../datasets/Wage.csv")
wage.head()
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
      <th>Unnamed: 0</th>
      <th>year</th>
      <th>age</th>
      <th>maritl</th>
      <th>race</th>
      <th>education</th>
      <th>region</th>
      <th>jobclass</th>
      <th>health</th>
      <th>health_ins</th>
      <th>logwage</th>
      <th>wage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>231655</td>
      <td>2006</td>
      <td>18</td>
      <td>1. Never Married</td>
      <td>1. White</td>
      <td>1. &lt; HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>2. No</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86582</td>
      <td>2004</td>
      <td>24</td>
      <td>1. Never Married</td>
      <td>1. White</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>2. No</td>
      <td>4.255273</td>
      <td>70.476020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>161300</td>
      <td>2003</td>
      <td>45</td>
      <td>2. Married</td>
      <td>1. White</td>
      <td>3. Some College</td>
      <td>2. Middle Atlantic</td>
      <td>1. Industrial</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.875061</td>
      <td>130.982177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>155159</td>
      <td>2003</td>
      <td>43</td>
      <td>2. Married</td>
      <td>3. Asian</td>
      <td>4. College Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>2. &gt;=Very Good</td>
      <td>1. Yes</td>
      <td>5.041393</td>
      <td>154.685293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11443</td>
      <td>2005</td>
      <td>50</td>
      <td>4. Divorced</td>
      <td>1. White</td>
      <td>2. HS Grad</td>
      <td>2. Middle Atlantic</td>
      <td>2. Information</td>
      <td>1. &lt;=Good</td>
      <td>1. Yes</td>
      <td>4.318063</td>
      <td>75.043154</td>
    </tr>
  </tbody>
</table>
</div>




```python
wage.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000 entries, 0 to 2999
    Data columns (total 12 columns):
    Unnamed: 0    3000 non-null int64
    year          3000 non-null int64
    age           3000 non-null int64
    maritl        3000 non-null object
    race          3000 non-null object
    education     3000 non-null object
    region        3000 non-null object
    jobclass      3000 non-null object
    health        3000 non-null object
    health_ins    3000 non-null object
    logwage       3000 non-null float64
    wage          3000 non-null float64
    dtypes: float64(2), int64(3), object(7)
    memory usage: 281.3+ KB


### Cleaning

#### Drop columns

The unnamed column appears to be some sort of id number, which is useless for our purposes. We can also drop `logwage` since it's redundant


```python
wage = wage.drop(columns=['Unnamed: 0', 'logwage'])
```

#### Convert to numerical dtypes


```python
wage_num = pd.get_dummies(wage)
wage_num.head()
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
      <th>year</th>
      <th>age</th>
      <th>wage</th>
      <th>maritl_1. Never Married</th>
      <th>maritl_2. Married</th>
      <th>maritl_3. Widowed</th>
      <th>maritl_4. Divorced</th>
      <th>maritl_5. Separated</th>
      <th>race_1. White</th>
      <th>race_2. Black</th>
      <th>...</th>
      <th>education_3. Some College</th>
      <th>education_4. College Grad</th>
      <th>education_5. Advanced Degree</th>
      <th>region_2. Middle Atlantic</th>
      <th>jobclass_1. Industrial</th>
      <th>jobclass_2. Information</th>
      <th>health_1. &lt;=Good</th>
      <th>health_2. &gt;=Very Good</th>
      <th>health_ins_1. Yes</th>
      <th>health_ins_2. No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006</td>
      <td>18</td>
      <td>75.043154</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004</td>
      <td>24</td>
      <td>70.476020</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>45</td>
      <td>130.982177</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>43</td>
      <td>154.685293</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>50</td>
      <td>75.043154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Preprocessing

#### Scaling the numerical variables


```python
df = wage_num[['year', 'age', 'wage']]
wage_num_std = wage_num.copy()
wage_num_std.loc[:, ['year', 'age', 'wage']] = (df - df.mean())/df.std()
wage_num_std.head()
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
      <th>year</th>
      <th>age</th>
      <th>wage</th>
      <th>maritl_1. Never Married</th>
      <th>maritl_2. Married</th>
      <th>maritl_3. Widowed</th>
      <th>maritl_4. Divorced</th>
      <th>maritl_5. Separated</th>
      <th>race_1. White</th>
      <th>race_2. Black</th>
      <th>...</th>
      <th>education_3. Some College</th>
      <th>education_4. College Grad</th>
      <th>education_5. Advanced Degree</th>
      <th>region_2. Middle Atlantic</th>
      <th>jobclass_1. Industrial</th>
      <th>jobclass_2. Information</th>
      <th>health_1. &lt;=Good</th>
      <th>health_2. &gt;=Very Good</th>
      <th>health_ins_1. Yes</th>
      <th>health_ins_2. No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.103150</td>
      <td>-2.115215</td>
      <td>-0.878545</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.883935</td>
      <td>-1.595392</td>
      <td>-0.987994</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.377478</td>
      <td>0.223986</td>
      <td>0.461999</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.377478</td>
      <td>0.050712</td>
      <td>1.030030</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.390392</td>
      <td>0.657171</td>
      <td>-0.878545</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## Fitting some nonlinear models


```python
X_sc, y_sc = wage_num_std.drop(columns=['wage']).values, wage_num_std['wage'].values
```


```python
X_sc.shape, y_sc.shape
```




    ((3000, 23), (3000,))



### Polynomial Ridge Regression

We don't need a special module for this model - we can use a `scikit-learn` pipeline.

We'll use 10-fold cross validation to pick the polynomial degree and L2 penalty.


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pr_pipe = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
pr_param_grid = dict(poly__degree=np.arange(1, 5), ridge__alpha=np.logspace(-4, 4, 5))
pr_search = GridSearchCV(pr_pipe, pr_param_grid, cv=5, scoring='neg_mean_squared_error')
pr_search.fit(X_sc, y_sc)
```




    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001))]),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'poly__degree': array([1, 2, 3, 4]), 'ridge__alpha': array([1.e-04, 1.e-02, 1.e+00, 1.e+02, 1.e+04])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
pr_search.best_params_
```




    {'poly__degree': 2, 'ridge__alpha': 100.0}



### Local Regression

`scikit-learn` has [support for local regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)


```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

lr_param_grid = dict(n_neighbors=np.arange(1,7), weights=['uniform', 'distance'], 
                     p=np.arange(1, 7))
lr_search = GridSearchCV(KNeighborsRegressor(), lr_param_grid, cv=10, 
                         scoring='neg_mean_squared_error')
lr_search.fit(X_sc, y_sc)
```




    GridSearchCV(cv=10, error_score='raise-deprecating',
           estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=5, p=2,
              weights='uniform'),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'n_neighbors': array([1, 2, 3, 4, 5, 6]), 'weights': ['uniform', 'distance'], 'p': array([1, 2, 3, 4, 5, 6])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
lr_search.best_params_
```




    {'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}



### GAMs

GAMs are quite general. There exists python modules that implement specific choices for the nonlinear component functions $f_i(X_i)$. Here we'll explore two modules that seem relatively mature/well-maintained.

#### GAMs with `pyGAM`

The module [`pyGAM`](https://pygam.readthedocs.io/en/latest/?badge=latest) implements [P-splines](https://en.wikipedia.org/wiki/B-spline#P-spline).


```python
from pygam import GAM, s, f

# generate string for terms
spline_terms = ' + '.join(['s(' + stri. + ')' for i in range(0,3)])
factor_terms = ' + '.join(['f(' + stri. + ')' 
                           for i in range(3,X_sc.shape[1])])
terms = spline_terms + ' + ' + factor_terms
terms

```




    's(0) + s(1) + s(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) + f(9) + f(10) + f(11) + f(12) + f(13) + f(14) + f(15) + f(16) + f(17) + f(18) + f(19) + f(20) + f(21) + f(22)'




```python
pygam_gam = GAM(s(0) + s(1) + s(2) + f(3) + f(4) + f(5) + f(6) + f(7) 
                + f(8) + f(9) + f(10) + f(11) + f(12) + f(13) + f(14) 
                + f(15) + f(16) + f(17) + f(18) + f(19) + f(20) + f(21) 
                + f(22))
```


```python
ps_search = pygam_gam.gridsearch(X_sc, y_sc, progress=True, 
                     lam=np.exp(np.random.rand(100, 23) * 6 - 3))
```

    100% (100 of 100) |######################| Elapsed Time: 0:00:13 Time:  0:00:13


## Model Selection

As in exercise 6, we'll select a model on the basis of mean squared test error. 


```python
mse_test_df = pd.DataFrame({'mse_test':np.zeros(3)}, index=['poly_ridge', 'local_reg', 'p_spline'])

# polynomial ridge and local regression models already have CV estimates of test mse
mse_test_df.at['poly_ridge', 'mse_test'] = -pr_search.best_score_
mse_test_df.at['local_reg', 'mse_test'] = -lr_search.best_score_
```


```python
from sklearn.model_selection import cross_val_score

# get p-spline CV estimate of test mse
mse_test_df.at['p_spline', 'mse_test'] = -np.mean(cross_val_score(ps_search,
                                                                  X_sc, y_sc, scoring='neg_mean_squared_error',
                                                                  cv=10))
```


```python
mse_test_df
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
      <th>mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>poly_ridge</th>
      <td>0.653614</td>
    </tr>
    <tr>
      <th>local_reg</th>
      <td>0.741645</td>
    </tr>
    <tr>
      <th>p_spline</th>
      <td>1.000513</td>
    </tr>
  </tbody>
</table>
</div>



Polynomial ridge regression has won out. Since this CV mse estimate was calculated on scaled data, let's get the estimate for the original data



```python
%%capture
X, y = wage_num.drop(columns=['wage']).values, wage_num['wage'].values
cv_score = cross_val_score(pr_search.best_estimator_, X, y, scoring='neg_mean_squared_error', cv=10)
```


```python
mse = -np.mean(cv_score)
mse
```




    1137.2123862552992




```python
me = np.sqrt(mse)
me
```




    33.722579768684646




```python
sns.distplot(wage['wage'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a20d1e668>




![png](ch07_exercise_07_files/ch07_exercise_07_42_1.png)



```python
wage['wage'].describe()
```




    count    3000.000000
    mean      111.703608
    std        41.728595
    min        20.085537
    25%        85.383940
    50%       104.921507
    75%       128.680488
    max       318.342430
    Name: wage, dtype: float64



This model predicts a mean (absolute) error of $~\approx 33.7$


```python
print('{}'.format(round(me/wage['wage'].std(), 2)))
```

    0.81


which is 0.81 standard deviations.

## Improvements

After inspecting the distribution of `wage`, it's fairly clear there is a group of outliers that are no doubt affecting the prediction accuracy of the model. Let's try to separate that group.


```python
sns.scatterplot(x=wage.index, y=wage['wage'].sort_values(), alpha=0.4, color='grey')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2469d8d0>




![png](ch07_exercise_07_files/ch07_exercise_07_49_1.png)


There appears to be a break point around 250. Let's take all rows with wage less than this


```python
wage_num_low = wage_num[wage_num['wage'] < 250]
wage_num_low_sc = wage_num_low.copy()

df = wage_num_low_sc[['year', 'age', 'wage']]
wage_num_low_sc.loc[:, ['year', 'age', 'wage']] = (df - df.mean())/df.std()
```

Let train the same models again


```python
X_low_sc, y_low_sc = wage_num_low_sc.drop(columns=['wage']).values, wage_num_low_sc['wage']

# polynomial ridge model
pr_search.fit(X_low_sc, y_low_sc)
```

    /Users/home/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001))]),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'poly__degree': array([1, 2, 3, 4]), 'ridge__alpha': array([1.e-04, 1.e-02, 1.e+00, 1.e+02, 1.e+04])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
pr_search.best_params_
```




    {'poly__degree': 2, 'ridge__alpha': 100.0}




```python
# local regression
lr_search.fit(X_low_sc, y_low_sc)
```




    GridSearchCV(cv=10, error_score='raise-deprecating',
           estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=5, p=2,
              weights='uniform'),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'n_neighbors': array([1, 2, 3, 4, 5, 6]), 'weights': ['uniform', 'distance'], 'p': array([1, 2, 3, 4, 5, 6])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
lr_search.best_params_
```




    {'n_neighbors': 6, 'p': 5, 'weights': 'uniform'}




```python
# p spline 
ps_search = pygam_gam.gridsearch(X_low_sc, y_low_sc, progress=True, 
                     lam=np.exp(np.random.rand(100, 23) * 6 - 3))
```

    100% (100 of 100) |######################| Elapsed Time: 0:00:25 Time:  0:00:25



```python
ps_search.summary()
```

    GAM                                                                                                       
    =============================================== ==========================================================
    Distribution:                        NormalDist Effective DoF:                                     28.9895
    Link Function:                     IdentityLink Log Likelihood:                                 -3606.1899
    Number of Samples:                         2921 AIC:                                             7272.3587
                                                    AICc:                                            7273.0018
                                                    GCV:                                                0.6157
                                                    Scale:                                              0.6047
                                                    Pseudo R-Squared:                                   0.4011
    ==========================================================================================================
    Feature Function                  Lambda               Rank         EDoF         P > x        Sig. Code   
    ================================= ==================== ============ ============ ============ ============
    s(0)                              [13.8473]            20           7.0          1.38e-05     ***         
    s(1)                              [17.046]             20           8.3          3.52e-13     ***         
    s(2)                              [0.1439]             20           1.1          5.30e-04     ***         
    f(3)                              [11.8899]            2            1.0          3.09e-02     *           
    f(4)                              [3.5507]             2            0.9          2.96e-01                 
    f(5)                              [0.1322]             2            0.9          1.79e-02     *           
    f(6)                              [14.0907]            2            0.0          3.22e-01                 
    f(7)                              [8.8878]             2            1.0          2.87e-01                 
    f(8)                              [0.2954]             2            1.0          3.01e-01                 
    f(9)                              [2.0563]             2            0.9          6.03e-01                 
    f(10)                             [8.6475]             2            0.0          3.51e-01                 
    f(11)                             [8.7695]             2            1.0          1.11e-16     ***         
    f(12)                             [0.1363]             2            1.0          2.69e-02     *           
    f(13)                             [0.9965]             2            1.0          1.11e-16     ***         
    f(14)                             [0.1233]             2            1.0          1.11e-16     ***         
    f(15)                             [0.3011]             2            0.0          1.11e-16     ***         
    f(16)                             [2.094]              1            0.0          1.00e+00                 
    f(17)                             [0.6317]             2            1.0          2.06e-01                 
    f(18)                             [16.422]             2            0.0          2.07e-01                 
    f(19)                             [0.3865]             2            1.0          1.33e-05     ***         
    f(20)                             [4.8438]             2            0.0          1.38e-05     ***         
    f(21)                             [0.1626]             2            1.0          1.11e-16     ***         
    f(22)                             [0.3707]             2            0.0          1.11e-16     ***         
    intercept                                              1            0.0          2.88e-01                 
    ==========================================================================================================
    Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    WARNING: Fitting splines and a linear function to a feature introduces a model identifiability problem
             which can cause p-values to appear significant when they are not.
    
    WARNING: p-values calculated in this manner behave correctly for un-penalized models or models with
             known smoothing parameters, but when smoothing parameters have been estimated, the p-values
             are typically lower than they should be, meaning that the tests reject the null too readily.


    /Users/home/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning: KNOWN BUG: p-values computed in this summary are likely much smaller than they should be. 
     
    Please do not make inferences based on these values! 
    
    Collaborate on a solution, and stay up to date at: 
    github.com/dswah/pyGAM/issues/163 
    
      """Entry point for launching an IPython kernel.



```python
low_mse_test_df = pd.DataFrame({'low_mse_test':np.zeros(3)}, index=['poly_ridge', 'local_reg', 'p_spline'])
low_mse_test_df.at['poly_ridge', 'low_mse_test'] = -pr_search.best_score_
low_mse_test_df.at['local_reg', 'low_mse_test'] = -lr_search.best_score_
low_mse_test_df.at['p_spline', 'low_mse_test'] = -np.mean(cross_val_score(ps_search,
                                                                 X_low_sc, y_low_sc, scoring='neg_mean_squared_error',
                                                                 cv=10))
mse_df = pd.concat([mse_test_df, low_mse_test_df], axis=1)
mse_df
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
      <th>mse_test</th>
      <th>low_mse_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>poly_ridge</th>
      <td>0.653614</td>
      <td>0.613098</td>
    </tr>
    <tr>
      <th>local_reg</th>
      <td>0.741645</td>
      <td>0.701615</td>
    </tr>
    <tr>
      <th>p_spline</th>
      <td>1.000513</td>
      <td>1.000513</td>
    </tr>
  </tbody>
</table>
</div>



There was a considerable improvement for the polynomial ridge and local regression models.
