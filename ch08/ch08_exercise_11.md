---
layout: page
title: 8. Tree-based Methods
---

{% katexmm %}

# Exercise 11: Predicting `Purchase` in `Caravan` dataset with a boosted tree classifier

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-train-test-split" data-toc-modified-id="a.-Train-test-split-2">a. Train test split</a></span></li><li><span><a href="#b-fit-boosted-tree-model" data-toc-modified-id="b.-Fit-boosted-tree-model-3">b. Fit boosted tree model</a></span></li><li><span><a href="#c-predict-purchase-and-compare-with-knn-logistic-regression" data-toc-modified-id="c.-Predict-Purchase-and-compare-with-KNN,-Logistic-Regression-4">c. Predict <code>Purchase</code> and compare with KNN, Logistic Regression</a></span><ul class="toc-item"><li><span><a href="#confusion-matrix-and-precision-for-boosted-tree-model" data-toc-modified-id="Confusion-Matrix-and-precision-for-Boosted-Tree-model-4.1">Confusion Matrix and precision for Boosted Tree model</a></span></li><li><span><a href="#confusion-matrix-and-precision-for-knn-model" data-toc-modified-id="Confusion-matrix-and-precision-for-KNN-model-4.2">Confusion matrix and precision for KNN model</a></span></li><li><span><a href="#confusion-matrix-and-precision-for-logistic-regression" data-toc-modified-id="Confusion-matrix-and-precision-for-Logistic-Regression-4.3">Confusion matrix and precision for Logistic Regression</a></span></li></ul></li></ul></div>

## Preparing the data

Information on the dataset can be [found here](https://rdrr.io/cran/ISLR/man/Caravan.html)


```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
caravan = pd.read_csv('../../datasets/Caravan.csv', index_col=0)
caravan.reset_index(inplace=True, drop=True)
caravan.head()
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

<div style="overflow:auto;">
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>MOSTYPE</th>
        <th>MAANTHUI</th>
        <th>MGEMOMV</th>
        <th>MGEMLEEF</th>
        <th>MOSHOOFD</th>
        <th>MGODRK</th>
        <th>MGODPR</th>
        <th>MGODOV</th>
        <th>MGODGE</th>
        <th>MRELGE</th>
        <th>...</th>
        <th>APERSONG</th>
        <th>AGEZONG</th>
        <th>AWAOREG</th>
        <th>ABRAND</th>
        <th>AZEILPL</th>
        <th>APLEZIER</th>
        <th>AFIETS</th>
        <th>AINBOED</th>
        <th>ABYSTAND</th>
        <th>Purchase</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>33</td>
        <td>1</td>
        <td>3</td>
        <td>2</td>
        <td>8</td>
        <td>0</td>
        <td>5</td>
        <td>1</td>
        <td>3</td>
        <td>7</td>
        <td>...</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>No</td>
      </tr>
      <tr>
        <th>1</th>
        <td>37</td>
        <td>1</td>
        <td>2</td>
        <td>2</td>
        <td>8</td>
        <td>1</td>
        <td>4</td>
        <td>1</td>
        <td>4</td>
        <td>6</td>
        <td>...</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>No</td>
      </tr>
      <tr>
        <th>2</th>
        <td>37</td>
        <td>1</td>
        <td>2</td>
        <td>2</td>
        <td>8</td>
        <td>0</td>
        <td>4</td>
        <td>2</td>
        <td>4</td>
        <td>3</td>
        <td>...</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>No</td>
      </tr>
      <tr>
        <th>3</th>
        <td>9</td>
        <td>1</td>
        <td>3</td>
        <td>3</td>
        <td>3</td>
        <td>2</td>
        <td>3</td>
        <td>2</td>
        <td>4</td>
        <td>5</td>
        <td>...</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>No</td>
      </tr>
      <tr>
        <th>4</th>
        <td>40</td>
        <td>1</td>
        <td>4</td>
        <td>2</td>
        <td>10</td>
        <td>1</td>
        <td>4</td>
        <td>1</td>
        <td>4</td>
        <td>7</td>
        <td>...</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>0</td>
        <td>No</td>
      </tr>
    </tbody>
  </table>
</div>
<p>5 rows × 86 columns</p>
</div>




```python
caravan.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5822 entries, 0 to 5821
    Data columns (total 86 columns):
    MOSTYPE     5822 non-null int64
    MAANTHUI    5822 non-null int64
    MGEMOMV     5822 non-null int64
    MGEMLEEF    5822 non-null int64
    MOSHOOFD    5822 non-null int64
    MGODRK      5822 non-null int64
    MGODPR      5822 non-null int64
    MGODOV      5822 non-null int64
    MGODGE      5822 non-null int64
    MRELGE      5822 non-null int64
    MRELSA      5822 non-null int64
    MRELOV      5822 non-null int64
    MFALLEEN    5822 non-null int64
    MFGEKIND    5822 non-null int64
    MFWEKIND    5822 non-null int64
    MOPLHOOG    5822 non-null int64
    MOPLMIDD    5822 non-null int64
    MOPLLAAG    5822 non-null int64
    MBERHOOG    5822 non-null int64
    MBERZELF    5822 non-null int64
    MBERBOER    5822 non-null int64
    MBERMIDD    5822 non-null int64
    MBERARBG    5822 non-null int64
    MBERARBO    5822 non-null int64
    MSKA        5822 non-null int64
    MSKB1       5822 non-null int64
    MSKB2       5822 non-null int64
    MSKC        5822 non-null int64
    MSKD        5822 non-null int64
    MHHUUR      5822 non-null int64
    MHKOOP      5822 non-null int64
    MAUT1       5822 non-null int64
    MAUT2       5822 non-null int64
    MAUT0       5822 non-null int64
    MZFONDS     5822 non-null int64
    MZPART      5822 non-null int64
    MINKM30     5822 non-null int64
    MINK3045    5822 non-null int64
    MINK4575    5822 non-null int64
    MINK7512    5822 non-null int64
    MINK123M    5822 non-null int64
    MINKGEM     5822 non-null int64
    MKOOPKLA    5822 non-null int64
    PWAPART     5822 non-null int64
    PWABEDR     5822 non-null int64
    PWALAND     5822 non-null int64
    PPERSAUT    5822 non-null int64
    PBESAUT     5822 non-null int64
    PMOTSCO     5822 non-null int64
    PVRAAUT     5822 non-null int64
    PAANHANG    5822 non-null int64
    PTRACTOR    5822 non-null int64
    PWERKT      5822 non-null int64
    PBROM       5822 non-null int64
    PLEVEN      5822 non-null int64
    PPERSONG    5822 non-null int64
    PGEZONG     5822 non-null int64
    PWAOREG     5822 non-null int64
    PBRAND      5822 non-null int64
    PZEILPL     5822 non-null int64
    PPLEZIER    5822 non-null int64
    PFIETS      5822 non-null int64
    PINBOED     5822 non-null int64
    PBYSTAND    5822 non-null int64
    AWAPART     5822 non-null int64
    AWABEDR     5822 non-null int64
    AWALAND     5822 non-null int64
    APERSAUT    5822 non-null int64
    ABESAUT     5822 non-null int64
    AMOTSCO     5822 non-null int64
    AVRAAUT     5822 non-null int64
    AAANHANG    5822 non-null int64
    ATRACTOR    5822 non-null int64
    AWERKT      5822 non-null int64
    ABROM       5822 non-null int64
    ALEVEN      5822 non-null int64
    APERSONG    5822 non-null int64
    AGEZONG     5822 non-null int64
    AWAOREG     5822 non-null int64
    ABRAND      5822 non-null int64
    AZEILPL     5822 non-null int64
    APLEZIER    5822 non-null int64
    AFIETS      5822 non-null int64
    AINBOED     5822 non-null int64
    ABYSTAND    5822 non-null int64
    Purchase    5822 non-null object
    dtypes: int64(85), object(1)
    memory usage: 3.8+ MB



```python
caravan = pd.get_dummies(caravan, drop_first=True)
```

## a. Train test split


```python
from sklearn.model_selection import train_test_split

X, y = caravan.drop(columns=['Purchase_Yes']), caravan['Purchase_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, random_state=27)
X_train.shape
```




    (1000, 85)



## b. Fit boosted tree model


```python
from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01)
boost_clf.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.01, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=1000,
                  n_iter_no_change=None, presort='auto', random_state=None,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)




```python
feat_imp = pd.DataFrame({'Feature Importance': boost_clf.feature_importances_},
                        index=X.columns).sort_values(by='Feature Importance', ascending=False)

feat_imp
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
      <th>Feature Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PBRAND</th>
      <td>7.989783e-02</td>
    </tr>
    <tr>
      <th>MOPLLAAG</th>
      <td>7.138105e-02</td>
    </tr>
    <tr>
      <th>MKOOPKLA</th>
      <td>6.779442e-02</td>
    </tr>
    <tr>
      <th>MBERARBG</th>
      <td>6.553880e-02</td>
    </tr>
    <tr>
      <th>PPERSAUT</th>
      <td>5.104349e-02</td>
    </tr>
    <tr>
      <th>MSKD</th>
      <td>4.747047e-02</td>
    </tr>
    <tr>
      <th>MINK7512</th>
      <td>4.596157e-02</td>
    </tr>
    <tr>
      <th>PPLEZIER</th>
      <td>3.968977e-02</td>
    </tr>
    <tr>
      <th>MGODOV</th>
      <td>3.914403e-02</td>
    </tr>
    <tr>
      <th>MOPLMIDD</th>
      <td>3.819071e-02</td>
    </tr>
    <tr>
      <th>APLEZIER</th>
      <td>3.810555e-02</td>
    </tr>
    <tr>
      <th>APERSAUT</th>
      <td>3.343816e-02</td>
    </tr>
    <tr>
      <th>MOSTYPE</th>
      <td>2.926208e-02</td>
    </tr>
    <tr>
      <th>PTRACTOR</th>
      <td>2.795271e-02</td>
    </tr>
    <tr>
      <th>PBYSTAND</th>
      <td>2.634395e-02</td>
    </tr>
    <tr>
      <th>ALEVEN</th>
      <td>2.426373e-02</td>
    </tr>
    <tr>
      <th>MSKC</th>
      <td>2.391492e-02</td>
    </tr>
    <tr>
      <th>MINK4575</th>
      <td>2.262974e-02</td>
    </tr>
    <tr>
      <th>MBERHOOG</th>
      <td>2.139981e-02</td>
    </tr>
    <tr>
      <th>MBERARBO</th>
      <td>1.534339e-02</td>
    </tr>
    <tr>
      <th>MFALLEEN</th>
      <td>1.528145e-02</td>
    </tr>
    <tr>
      <th>MINKGEM</th>
      <td>1.474515e-02</td>
    </tr>
    <tr>
      <th>MGEMOMV</th>
      <td>1.381983e-02</td>
    </tr>
    <tr>
      <th>MGODPR</th>
      <td>1.217382e-02</td>
    </tr>
    <tr>
      <th>MSKB2</th>
      <td>1.195765e-02</td>
    </tr>
    <tr>
      <th>MINK3045</th>
      <td>9.918341e-03</td>
    </tr>
    <tr>
      <th>MBERMIDD</th>
      <td>9.052936e-03</td>
    </tr>
    <tr>
      <th>PLEVEN</th>
      <td>8.742057e-03</td>
    </tr>
    <tr>
      <th>MOPLHOOG</th>
      <td>8.205575e-03</td>
    </tr>
    <tr>
      <th>MFWEKIND</th>
      <td>7.063705e-03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>MHKOOP</th>
      <td>6.941976e-04</td>
    </tr>
    <tr>
      <th>MSKB1</th>
      <td>6.914181e-04</td>
    </tr>
    <tr>
      <th>MRELOV</th>
      <td>5.480189e-04</td>
    </tr>
    <tr>
      <th>ATRACTOR</th>
      <td>2.495046e-04</td>
    </tr>
    <tr>
      <th>PMOTSCO</th>
      <td>2.387955e-04</td>
    </tr>
    <tr>
      <th>MBERBOER</th>
      <td>2.193347e-04</td>
    </tr>
    <tr>
      <th>AMOTSCO</th>
      <td>2.157301e-04</td>
    </tr>
    <tr>
      <th>MGODRK</th>
      <td>2.125493e-04</td>
    </tr>
    <tr>
      <th>PWABEDR</th>
      <td>1.929167e-04</td>
    </tr>
    <tr>
      <th>PWERKT</th>
      <td>1.205401e-04</td>
    </tr>
    <tr>
      <th>AWERKT</th>
      <td>9.615065e-05</td>
    </tr>
    <tr>
      <th>AFIETS</th>
      <td>8.984330e-05</td>
    </tr>
    <tr>
      <th>PFIETS</th>
      <td>8.910239e-05</td>
    </tr>
    <tr>
      <th>AWABEDR</th>
      <td>3.331575e-05</td>
    </tr>
    <tr>
      <th>APERSONG</th>
      <td>2.180892e-05</td>
    </tr>
    <tr>
      <th>PPERSONG</th>
      <td>1.190193e-05</td>
    </tr>
    <tr>
      <th>PWALAND</th>
      <td>1.866996e-07</td>
    </tr>
    <tr>
      <th>AWALAND</th>
      <td>1.618010e-07</td>
    </tr>
    <tr>
      <th>AWAOREG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>AGEZONG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>AZEILPL</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>AVRAAUT</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>AAANHANG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PGEZONG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>ABESAUT</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PAANHANG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PBESAUT</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PZEILPL</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PWAOREG</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>PVRAAUT</th>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 1 columns</p>
</div>



## c. Predict `Purchase` and compare with KNN, Logistic Regression

### Confusion Matrix and precision for Boosted Tree model


```python
from sklearn.metrics import confusion_matrix

y_act = pd.Series(['Yes' if entry == 1 else 'No' for entry in y_test],
                  name='Actual')
y_pred = pd.Series(['Yes' if prob[1] > 0.2 else 'No' for prob in boost_clf.predict_proba(X_test)],
                        name='Predicted')
boost_tree_conf = pd.crosstab(y_act, y_pred, margins=True)
boost_tree_conf
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
      <th>Predicted</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>4328</td>
      <td>204</td>
      <td>4532</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>242</td>
      <td>48</td>
      <td>290</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4570</td>
      <td>252</td>
      <td>4822</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fraction of people predicted to make a purchase that actually do - this is the "precision"
boost_tree_conf.at['Yes', 'Yes']/(boost_tree_conf.at['Yes', 'No'] + boost_tree_conf.at['Yes', 'Yes'])
```




    0.16551724137931034



### Confusion matrix and precision for KNN model


```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

y_act = pd.Series(['Yes' if entry == 1 else 'No' for entry in y_test],
                  name='Actual')
y_pred = pd.Series(['Yes' if prob[1] > 0.2 else 'No' for prob in knn_clf.predict_proba(X_test)],
                        name='Predicted')
knn_conf = pd.crosstab(y_act, y_pred, margins=True)
knn_conf
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
      <th>Predicted</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>4340</td>
      <td>192</td>
      <td>4532</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>259</td>
      <td>31</td>
      <td>290</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4599</td>
      <td>223</td>
      <td>4822</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fraction of people predicted to make a purchase that actually do - this is the "precision"
knn_conf.at['Yes', 'Yes']/(knn_conf.at['Yes', 'No'] + knn_conf.at['Yes', 'Yes'])
```




    0.10689655172413794



### Confusion matrix and precision for Logistic Regression


```python
from sklearn.linear_model import LogisticRegression

logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)

y_act = pd.Series(['Yes' if entry == 1 else 'No' for entry in y_test],
                  name='Actual')
y_pred = pd.Series(['Yes' if prob[1] > 0.2 else 'No' for prob in logreg_clf.predict_proba(X_test)],
                        name='Predicted')
logreg_conf = pd.crosstab(y_act, y_pred, margins=True)
logreg_conf
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
      <th>Predicted</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>4275</td>
      <td>257</td>
      <td>4532</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>254</td>
      <td>36</td>
      <td>290</td>
    </tr>
    <tr>
      <th>All</th>
      <td>4529</td>
      <td>293</td>
      <td>4822</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fraction of people predicted to make a purchase that actually do - this is the "precision"
logreg_conf.at['Yes', 'Yes']/(logreg_conf.at['Yes', 'No'] + logreg_conf.at['Yes', 'Yes'])
```




    0.12413793103448276


{% endkatexmm %}