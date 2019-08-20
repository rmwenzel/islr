---
layout: page
title: 8. Tree-based Methods
---

{% katexmm %}

# Exercise 9: Using tree based methods on the `OJ` dataset

<div class="toc"><ul class="toc-item"><li><span><a href="#preparing-the-data" data-toc-modified-id="Preparing-the-data-1">Preparing the data</a></span></li><li><span><a href="#a-train-test-split" data-toc-modified-id="a.-Train-test-split-2">a. Train-test split</a></span></li><li><span><a href="#b-classification-Tree-for-predicting-Purchase" data-toc-modified-id="b.-Classification-Tree-for-predicting-Purchase-3">b. Classification Tree for predicting <code>Purchase</code></a></span></li><li><span><a href="#c-classification-tree-feature-importances" data-toc-modified-id="c.-Classification-tree-feature-importances-4">c. Classification tree feature importances</a></span></li><li><span><a href="#d-classification-tree-plot" data-toc-modified-id="d.-Classification-tree-plot-5">d. Classification tree plot</a></span></li><li><span><a href="#e-confusion-matrix-for-test-data" data-toc-modified-id="e.-Confusion-matrix-for-test-data-6">e. Confusion matrix for test data</a></span></li><li><span><a href="#f-cross-validation-for-optimal-tree-size" data-toc-modified-id="f.-Cross-validation-for-optimal-tree-size-7">f. Cross-validation for optimal tree size</a></span></li><li><span><a href="#g-plot-of-cv-error-vs-tree-size" data-toc-modified-id="g.-Plot-of-CV-error-vs.-tree-size-8">g. Plot of CV error vs. tree size</a></span></li><li><span><a href="#j-comparing-training-error-rates" data-toc-modified-id="j.-Comparing-training-error-rates-9">j. Comparing training error rates</a></span></li></ul></div>

## Preparing the data

Information on the dataset can be [found here](https://rdrr.io/cran/ISLR/man/OJ.html)


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style('whitegrid')
oj = pd.read_csv('../../datasets/OJ.csv', index_col=0)
oj = oj.reset_index(drop=True)
oj.head()
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
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH</td>
      <td>237</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CH</td>
      <td>239</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CH</td>
      <td>245</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MM</td>
      <td>227</td>
      <td>1</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CH</td>
      <td>228</td>
      <td>7</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
oj.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1070 entries, 0 to 1069
    Data columns (total 18 columns):
    Purchase          1070 non-null object
    WeekofPurchase    1070 non-null int64
    StoreID           1070 non-null int64
    PriceCH           1070 non-null float64
    PriceMM           1070 non-null float64
    DiscCH            1070 non-null float64
    DiscMM            1070 non-null float64
    SpecialCH         1070 non-null int64
    SpecialMM         1070 non-null int64
    LoyalCH           1070 non-null float64
    SalePriceMM       1070 non-null float64
    SalePriceCH       1070 non-null float64
    PriceDiff         1070 non-null float64
    Store7            1070 non-null object
    PctDiscMM         1070 non-null float64
    PctDiscCH         1070 non-null float64
    ListPriceDiff     1070 non-null float64
    STORE             1070 non-null int64
    dtypes: float64(11), int64(5), object(2)
    memory usage: 150.5+ KB



```python
# drop superfluous variables
oj = oj.drop(columns=['STORE', 'Store7'])
```


```python
oj.columns
```




    Index(['Purchase', 'WeekofPurchase', 'StoreID', 'PriceCH', 'PriceMM', 'DiscCH',
           'DiscMM', 'SpecialCH', 'SpecialMM', 'LoyalCH', 'SalePriceMM',
           'SalePriceCH', 'PriceDiff', 'PctDiscMM', 'PctDiscCH', 'ListPriceDiff'],
          dtype='object')




```python
# one hot encode categoricals
cat_vars = ['StoreID', 'Purchase']

oj = pd.get_dummies(data=oj, columns=cat_vars, drop_first=True)
oj.head()
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
      <th>WeekofPurchase</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>StoreID_2</th>
      <th>StoreID_3</th>
      <th>StoreID_4</th>
      <th>StoreID_7</th>
      <th>Purchase_MM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>239</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>227</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
oj.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1070 entries, 0 to 1069
    Data columns (total 18 columns):
    Purchase          1070 non-null object
    WeekofPurchase    1070 non-null int64
    StoreID           1070 non-null int64
    PriceCH           1070 non-null float64
    PriceMM           1070 non-null float64
    DiscCH            1070 non-null float64
    DiscMM            1070 non-null float64
    SpecialCH         1070 non-null int64
    SpecialMM         1070 non-null int64
    LoyalCH           1070 non-null float64
    SalePriceMM       1070 non-null float64
    SalePriceCH       1070 non-null float64
    PriceDiff         1070 non-null float64
    Store7            1070 non-null object
    PctDiscMM         1070 non-null float64
    PctDiscCH         1070 non-null float64
    ListPriceDiff     1070 non-null float64
    STORE             1070 non-null int64
    dtypes: float64(11), int64(5), object(2)
    memory usage: 150.5+ KB


## a. Train-test split


```python
from sklearn.model_selection import train_test_split

X, y = oj.drop(columns=['Purchase_MM']), oj['Purchase_MM']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=27)
X_train.shape
```

    (800, 18)



## b. Classification Tree for predicting `Purchase`


```python
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(random_state=27)
clf_tree.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=27,
                splitter='best')




```python
# training error rate
clf_tree.score(X_train, y_train)
```




    0.98875




```python
# test error rate
clf_tree.score(X_test, y_test)
```




    0.7777777777777778



The following is lifted straight from the [`sklearn` docs](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py):


```python
estimator = clf_tree
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

# number of leaves = number of terminal nodes
np.sum(is_leaves)
```




    165



## c. Classification tree feature importances


```python
# feature importances
clf_tree_feat_imp = pd.DataFrame({'feature': X_train.columns, 
                            'importance': clf_tree.feature_importances_})
clf_tree_feat_imp.sort_values(by='importance', ascending=False)
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
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>LoyalCH</td>
      <td>0.659534</td>
    </tr>
    <tr>
      <th>0</th>
      <td>WeekofPurchase</td>
      <td>0.098166</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PriceDiff</td>
      <td>0.085502</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ListPriceDiff</td>
      <td>0.026363</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SalePriceMM</td>
      <td>0.024777</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PriceCH</td>
      <td>0.016979</td>
    </tr>
    <tr>
      <th>14</th>
      <td>StoreID_2</td>
      <td>0.013228</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SpecialCH</td>
      <td>0.010388</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SalePriceCH</td>
      <td>0.010037</td>
    </tr>
    <tr>
      <th>15</th>
      <td>StoreID_3</td>
      <td>0.009939</td>
    </tr>
    <tr>
      <th>17</th>
      <td>StoreID_7</td>
      <td>0.009578</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PriceMM</td>
      <td>0.007906</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PctDiscCH</td>
      <td>0.007804</td>
    </tr>
    <tr>
      <th>16</th>
      <td>StoreID_4</td>
      <td>0.006839</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SpecialMM</td>
      <td>0.004914</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PctDiscMM</td>
      <td>0.004139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DiscMM</td>
      <td>0.003906</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DiscCH</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## d. Classification tree plot


```python
from graphviz import Source
from sklearn import tree
from IPython.display import SVG
graph = Source(tree.export_graphviz(clf_tree, out_file=None, filled=True, 
                                    feature_names=X.columns))
display(SVG(graph.pipe(format='svg')))
```


[![svg]({{site.baseurl}}/assets/images/ch08_exercise_09_21_0.svg)]({{site.baseurl}}/assets/images/ch08_exercise_09_21_0.svg)


## e. Confusion matrix for test data

This [Stack Overflow post](https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python) was helpful


```python
from sklearn.metrics import confusion_matrix

y_act = pd.Series(['Purchase_CH' if entry == 0 else "Purchase_MM" for entry in y_test],
                  name='Actual')
y_pred = pd.Series(['Purchase_CH' if entry == 0 else "Purchase_MM" for entry in clf_tree.predict(X_test)],
                  name='Predicted')
clf_tree_test_conf = pd.crosstab(y_act, y_pred, margins=True)
clf_tree_test_conf

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
      <th>Purchase_CH</th>
      <th>Purchase_MM</th>
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
      <th>Purchase_CH</th>
      <td>136</td>
      <td>34</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Purchase_MM</th>
      <td>26</td>
      <td>74</td>
      <td>100</td>
    </tr>
    <tr>
      <th>All</th>
      <td>162</td>
      <td>108</td>
      <td>270</td>
    </tr>
  </tbody>
</table>
</div>



## f. Cross-validation for optimal tree size


```python
from sklearn.model_selection import GridSearchCV

params = {'max_depth': list(range(1, 800)) + [None]}
clf_tree = DecisionTreeClassifier(random_state=27)
clf_tree_search = GridSearchCV(estimator=clf_tree, 
                               param_grid=params,
                               cv=8,
                               scoring='accuracy')

%timeit -n1 -r1 clf_tree_search.fit(X_train, y_train)
```

    44.9 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)



```python
clf_tree_search.best_params_
```




    {'max_depth': 2}




```python
clf_tree_search.best_score_
```




    0.8075



## g. Plot of CV error vs. tree size


```python
clf_tree_search_df = pd.DataFrame(clf_tree_search.cv_results_)
clf_tree_search_df.head()
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
      <th>param_max_depth</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>...</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>split5_train_score</th>
      <th>split6_train_score</th>
      <th>split7_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003523</td>
      <td>0.000421</td>
      <td>0.001491</td>
      <td>0.000361</td>
      <td>1</td>
      <td>{'max_depth': 1}</td>
      <td>0.801980</td>
      <td>0.841584</td>
      <td>0.811881</td>
      <td>0.76</td>
      <td>...</td>
      <td>0.816881</td>
      <td>0.811159</td>
      <td>0.809728</td>
      <td>0.817143</td>
      <td>0.817143</td>
      <td>0.814551</td>
      <td>0.810271</td>
      <td>0.803138</td>
      <td>0.812502</td>
      <td>0.004591</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002702</td>
      <td>0.000291</td>
      <td>0.001097</td>
      <td>0.000200</td>
      <td>2</td>
      <td>{'max_depth': 2}</td>
      <td>0.801980</td>
      <td>0.841584</td>
      <td>0.811881</td>
      <td>0.78</td>
      <td>...</td>
      <td>0.816881</td>
      <td>0.811159</td>
      <td>0.816881</td>
      <td>0.827143</td>
      <td>0.821429</td>
      <td>0.818830</td>
      <td>0.817404</td>
      <td>0.810271</td>
      <td>0.817500</td>
      <td>0.005043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003065</td>
      <td>0.000534</td>
      <td>0.001211</td>
      <td>0.000264</td>
      <td>3</td>
      <td>{'max_depth': 3}</td>
      <td>0.811881</td>
      <td>0.801980</td>
      <td>0.831683</td>
      <td>0.73</td>
      <td>...</td>
      <td>0.826896</td>
      <td>0.831187</td>
      <td>0.836910</td>
      <td>0.851429</td>
      <td>0.844286</td>
      <td>0.841655</td>
      <td>0.841655</td>
      <td>0.830243</td>
      <td>0.838032</td>
      <td>0.007727</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002972</td>
      <td>0.000107</td>
      <td>0.000967</td>
      <td>0.000055</td>
      <td>4</td>
      <td>{'max_depth': 4}</td>
      <td>0.801980</td>
      <td>0.811881</td>
      <td>0.831683</td>
      <td>0.73</td>
      <td>...</td>
      <td>0.856938</td>
      <td>0.841202</td>
      <td>0.841202</td>
      <td>0.868571</td>
      <td>0.847143</td>
      <td>0.861626</td>
      <td>0.858773</td>
      <td>0.844508</td>
      <td>0.852495</td>
      <td>0.009673</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.003204</td>
      <td>0.000121</td>
      <td>0.000971</td>
      <td>0.000060</td>
      <td>5</td>
      <td>{'max_depth': 5}</td>
      <td>0.782178</td>
      <td>0.821782</td>
      <td>0.801980</td>
      <td>0.75</td>
      <td>...</td>
      <td>0.868383</td>
      <td>0.852647</td>
      <td>0.856938</td>
      <td>0.880000</td>
      <td>0.865714</td>
      <td>0.873039</td>
      <td>0.881598</td>
      <td>0.864479</td>
      <td>0.867850</td>
      <td>0.009552</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
x, y = clv_tree_search_df['param_max_depth'], clv_tree_search_df['mean_test_score']
plt.plot(x, y)
```




    [<matplotlib.lines.Line2D at 0x1a19ea4908>]




![png]({{site.baseurl}}/assets/images/ch08_exercise_09_31_1.png)


We chose an upper limit of 799 for the maximum tree depth (in a worst case scenario, the decision tree partitions into unique regions for each observation). This is clearly overkill


```python
plt.plot(x.loc[:30, ], y.loc[:30, ])
```




    [<matplotlib.lines.Line2D at 0x1a1a67e6d8>]




![png]({{site.baseurl}}/assets/images/ch08_exercise_09_33_1.png)



```python
plt.plot(np.arange(y))
```

A maximum tree depth of 2 leads to the best cv test error!

# i. Pruning a tree with depth 2


```python
from sklearn.model_selection import RandomizedSearchCV

params = {'max_features': np.arange(1, 18),
          'max_leaf_nodes': np.append(np.arange(2, 21), None),
          'min_impurity_decrease': np.linspace(0, 1, 10),
          'min_samples_leaf': np.arange(1, 11)
          }
clf_tree2 = DecisionTreeClassifier(random_state=27, max_depth=2)

# Randomized search to cover a large region of parameter space
clf_tree2_rvsearch = RandomizedSearchCV(estimator=clf_tree2,
                                        param_distributions=params,
                                        cv=8,
                                        scoring='accuracy',
                                        n_iter=1000,
                                        random_state=27)
%timeit -n1 -r1 clf_tree2_rvsearch.fit(X_train, y_train)
```

    39.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


    /anaconda3/envs/islr/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
clf_tree2_rvsearch.best_params_
```




    {'min_samples_leaf': 6,
     'min_impurity_decrease': 0.0,
     'max_leaf_nodes': None,
     'max_features': 16}




```python
# Grid search nearby randomized search results
params = {'min_samples_leaf': np.arange(4, 8),
          'max_features': np.arange(10, 16)
         }
clf_tree3 = DecisionTreeClassifier(random_state=27, 
                                   max_depth=2,
                                   max_leaf_nodes=None)
clf_tree2_gridsearch = GridSearchCV(estimator=clf_tree3,
                                    param_grid=params,
                                    cv=8,
                                    scoring='accuracy')
%timeit -n1 -r1 clf_tree2_gridsearch.fit(X_train, y_train)
```

    1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


    /anaconda3/envs/islr/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
clf_tree2_gridsearch.best_params_
```




    {'max_features': 15, 'min_samples_leaf': 4}




```python
clf_tree2_gridsearch.best_score_
```




    0.8075



## j. Comparing training error rates


```python
from sklearn.metrics import accuracy_score

# trees for comparison
clf_tree = clf_tree_search.best_estimator_
pruned_clf_tree = clf_tree2_gridsearch.best_estimator_

# train and test errors
train = [accuracy_score(clf_tree.predict(X_train), y_train), 
         accuracy_score(pruned_clf_tree.predict(X_train), y_train)]
test = [accuracy_score(clf_tree.predict(X_test), y_test), 
         accuracy_score(pruned_clf_tree.predict(X_test), y_test)]

# df for results
comp_df = pd.DataFrame({'train_error': train, 'test_error': test}, 
                       index=['clf_tree', 'pruned_clf_tree'])
comp_df
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
      <th>train_error</th>
      <th>test_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>clf_tree</th>
      <td>0.81625</td>
      <td>0.785185</td>
    </tr>
    <tr>
      <th>pruned_clf_tree</th>
      <td>0.81625</td>
      <td>0.785185</td>
    </tr>
  </tbody>
</table>
</div>

{% endkatexmm %}
