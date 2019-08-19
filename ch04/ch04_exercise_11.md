---
layout: page
title: 4. Logistic Regression
---

{% katexmm %}

# Exercise 11: Classify high/low `mpg` cars in `Auto` dataset

<div class="toc">
  <ul class="toc-item">
    <li><span><a href="#prepare-the-dataset" data-toc-modified-id="Prepare-the-dataset-1">Prepare the dataset</a></span></li>
    <li><span><a href="#a-create-high-and-low-mpg-classes" data-toc-modified-id="a.-Create-high-and-low-mpg-classes-2">a. Create high and low mpg classes</a></span></li>
    <li><span><a href="#b-visual-feature-selection" data-toc-modified-id="b.-Visual-feature-selection-3">b. Visual feature selection</a></span>
      <ul class="toc-item">
        <li><span><a href="#quantitative-features" data-toc-modified-id="Quantitative-features-3.1">Quantitative features</a></span></li>
        <li><span><a href="qualitative-features" data-toc-modified-id="Qualitative-features-3.2">Qualitative features</a></span></li>
      </ul>
    </li>
    <li><span><a href="#c-train-test-split" data-toc-modified-id="c.-Train-test-split-4">c. Train-test split</a></span></li>
    <li><span><a href="#d-lda-model" data-toc-modified-id="d.-LDA-model-5">d. LDA model</a></span></li>
    <li><span><a href="#e-qda-model" data-toc-modified-id="e.-QDA-model-6">e. QDA model</a></span></li>
    <li><span><a href="#f-logit-model" data-toc-modified-id="f.-Logit-model-7">f. Logit model</a></span></li>
    <li><span><a href="#g-knn-model" data-toc-modified-id="g.-KNN-model-8">g. KNN model</a></span></li>
  </ul>
</div>

## Prepare the dataset


```python
import pandas as pd

auto = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Auto.csv')
```


```python
### 
## impute missing horsepower values with mean
#

# replace `?` with 0 so means can be calculated
for index in auto.index:
    if auto.loc[index, 'horsepower'] == '?':
        auto.loc[index, 'horsepower'] = 0

# cast horsepower to numeric dtype
auto.loc[ : , 'horsepower'] = pd.to_numeric(auto.horsepower)

# now impute values
for index in auto.index:
    if auto.loc[index, 'horsepower'] == 0:
        auto.loc[index, 'horsepower'] = auto[auto.cylinders == auto.loc[index, 'cylinders']].horsepower.mean()
```

## a. Create high and low mpg classes


```python
# represent high mpg as mpg above the median
auto['mpg01'] = (auto.mpg > auto.mpg.median()).astype('int32')
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
      <th>mpg01</th>
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
      <td>chevrolet chevelle malibu</td>
      <td>0</td>
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
      <td>buick skylark 320</td>
      <td>0</td>
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
      <td>plymouth satellite</td>
      <td>0</td>
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
      <td>amc rebel sst</td>
      <td>0</td>
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
      <td>ford torino</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
auto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 397 entries, 0 to 396
    Data columns (total 10 columns):
    mpg             397 non-null float64
    cylinders       397 non-null int64
    displacement    397 non-null float64
    horsepower      397 non-null float64
    weight          397 non-null int64
    acceleration    397 non-null float64
    year            397 non-null int64
    origin          397 non-null int64
    name            397 non-null object
    mpg01           397 non-null int32
    dtypes: float64(4), int32(1), int64(4), object(1)
    memory usage: 29.5+ KB


Note high `mpg` is represented by class 1

## b. Visual feature selection


```python
import seaborn as sns
sns.set_style('white')

import warnings  
warnings.filterwarnings('ignore')
```

### Quantitative features

We'll inspect some plots of the quantitative variables against the high/low classes first.

Of course, `mpg` will completely separate classes, which is unsurprising. We won't use this feature in our models, since it seems like cheating and/or makes the exercise uninteresting.


```python
ax = sns.stripplot(x="mpg01", y="mpg", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_13_0.png)


Now let's look at the other quanitative features


```python
ax = sns.violinplot(x="mpg01", y="displacement", data=auto)
ax = sns.stripplot(x="mpg01", y="displacement", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_15_0.png)



```python
ax = sns.violinplot(x="mpg01", y="horsepower", data=auto)
ax = sns.stripplot(x="mpg01", y="horsepower", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_16_0.png)



```python
ax = sns.catplot(x="mpg01", y="weight", data=auto, kind='violin')
ax = sns.stripplot(x="mpg01", y="weight", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_17_0.png)



```python
ax = sns.violinplot(x="mpg01", y="acceleration", data=auto)
ax = sns.stripplot(x="mpg01", y="acceleration", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_18_0.png)


Since the number of unique values for `year` is small


```python
auto.year.unique()
```




    array([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82])



one might argue it should be trasted as a qualitative variable. However, since there it has a natural (time) ordering we treat it as quantitative 


```python
ax = sns.catplot(x="mpg01", y="year", data=auto, kind='violin')
ax = sns.stripplot(x="mpg01", y="year", data=auto)
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_22_0.png)


With the exception of `acceleration` and possibly year, all these plots show a good separation of the distributions across the `mpg` classes. Based on these plots, all the quantitative variable except `acceleration` look like useful features for predicting `mpg` class.

### Qualitative features


```python
ax = sns.catplot(x="cylinders", hue='mpg01', data=auto, kind='count')
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_25_0.png)



```python
ax = sns.catplot(x="origin", hue='mpg01', data=auto, kind='count')
```


![png]({{site.baseurl}}/assets/images/ch04_exercise_11_26_0.png)


Since `cylinders` and `origin` separate the `mpg` classes well they both should be useful predictors. 

Note: we're going to ignore `name` for now. This is a categorical variable but it has a lot of levels


```python
len(auto.name.unique())
```




    304



and analysis is likely a bit complicated

## c. Train-test split


```python
import sklearn.model_selection as skl_model_selection
import statsmodels.api as sm
```


```python
X, y = sm.add_constant(auto.drop(['acceleration', 'mpg01', 'name'], axis=1).values), auto.mpg01.values
X_train, X_test, y_train, y_test = skl_model_selection.train_test_split(X, y)
```

## d. LDA model


```python
import sklearn.discriminant_analysis as skl_discriminant_analysis

LDA_model = skl_discriminant_analysis.LinearDiscriminantAnalysis()
```


```python
import sklearn.metrics as skl_metrics

skl_metrics.accuracy_score(y_test, LDA_model.fit(X_train, y_train).predict(X_test))
```




    0.93



Impressive!

## e. QDA model


```python
QDA_model = skl_discriminant_analysis.QuadraticDiscriminantAnalysis()
```


```python
skl_metrics.accuracy_score(y_test, QDA_model.fit(X_train, y_train).predict(X_test))
```




    0.48



Much worse :(

### f. Logit model


```python
import sklearn.linear_model as skl_linear_model

Logit_model = skl_linear_model.LogisticRegression()
```


```python
skl_metrics.accuracy_score(y_test, Logit_model.fit(X_train, y_train).predict(X_test))
```




    0.94



Better!

### g. KNN model


```python
import sklearn.neighbors as skl_neighbors
```


```python
models = {}
accuracies = {}

for i in range(1, 11):
    name = 'KNN' + str(i)
    models[name] = skl_neighbors.KNeighborsClassifier(n_neighbors=i)
    accuracies[name] = skl_metrics.accuracy_score(y_test, models[name].fit(X_train, y_train).predict(X_test))
```


```python
pd.DataFrame(accuracies, index=[0]).sort_values(by=[0], axis='columns', ascending=False)
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
      <th>KNN5</th>
      <th>KNN9</th>
      <th>KNN10</th>
      <th>KNN3</th>
      <th>KNN7</th>
      <th>KNN1</th>
      <th>KNN6</th>
      <th>KNN8</th>
      <th>KNN4</th>
      <th>KNN2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.83</td>
      <td>0.83</td>
      <td>0.83</td>
      <td>0.82</td>
      <td>0.82</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.76</td>
      <td>0.7</td>
    </tr>
  </tbody>
</table>
</div>



These values are all really close, except perhaps $N=2,4$. Given the bias-variance tradeoff, we'd probably want to select $N=3$ or $N=5$ based on these results


{% endkatexmm %}
