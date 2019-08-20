---
layout: page
title: 8. Tree-based Methods
---

{% katexmm %}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#tree-based-methods" data-toc-modified-id="Tree-Based-Methods-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Tree-Based Methods</a></span><ul class="toc-item"><li><span><a href="#the-basics-of-decision-trees" data-toc-modified-id="The-Basics-of-Decision-Trees-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>The Basics of Decision Trees</a></span><ul class="toc-item"><li><span><a href="#regression-trees" data-toc-modified-id="Regression-Trees-8.1.1"><span class="toc-item-num">8.1.1&nbsp;&nbsp;</span>Regression Trees</a></span></li><li><span><a href="#classification-trees" data-toc-modified-id="Classification-Trees-8.1.2"><span class="toc-item-num">8.1.2&nbsp;&nbsp;</span>Classification Trees</a></span></li><li><span><a href="#trees-versus-linear-models" data-toc-modified-id="Trees-Versus-Linear-Models-8.1.3"><span class="toc-item-num">8.1.3&nbsp;&nbsp;</span>Trees Versus Linear Models</a></span></li><li><span><a href="#advantages-and-disadvantages-of-trees" data-toc-modified-id="Advantages-and-Disadvantages-of-Trees-8.1.4"><span class="toc-item-num">8.1.4&nbsp;&nbsp;</span>Advantages and Disadvantages of Trees</a></span></li></ul></li><li><span><a href="#bagging-random-forests-boosting" data-toc-modified-id="Bagging,-Random-Forests,-Boosting-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Bagging, Random Forests, Boosting</a></span><ul class="toc-item"><li><span><a href="#bagging" data-toc-modified-id="Bagging-8.2.1"><span class="toc-item-num">8.2.1&nbsp;&nbsp;</span>Bagging</a></span></li><li><span><a href="#random-forests" data-toc-modified-id="Random-Forests-8.2.2"><span class="toc-item-num">8.2.2&nbsp;&nbsp;</span>Random Forests</a></span></li><li><span><a href="#boosting" data-toc-modified-id="Boosting-8.2.3"><span class="toc-item-num">8.2.3&nbsp;&nbsp;</span>Boosting</a></span></li></ul></li><li><span><a href="#footnotes" data-toc-modified-id="Footnotes-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Footnotes</a></span><ul class="toc-item"><li><span><a href="#blah" data-toc-modified-id="blah-8.3.1"><span class="toc-item-num">8.3.1&nbsp;&nbsp;</span>blah</a></span></li></ul></li></ul></li></ul></div>


## The Basics of Decision Trees

### Regression Trees

##### Overview

- There are two main steps:
    - Partition predictor space $\mathbb{R}^p$ into regions $R_1, \dots, R_M$.
    - For all $X = (X_1, \dots, X_p) \in R_m$, predict the average over the responses in $R_m$
    $$\hat{f}(X) = \hat{y}_{R_m} := \frac{1}{N_m}\sum_{i: y_i\in R_m} y_i$$
    where $N_m = |\{y_i\ |\ y_i \in R_m\}|$
- In practice, we take the regions of the partition to be rectangular for simplicity and ease of interpretation.
- We choose the partition to minimize the RSS
$$ \sum_{m = 1}^M \sum_{i: y_i \in R_m} (y_i - \hat{y}_{R_m})^2 $$
- We search the space of partitions using a ***recursive binary splitting***[^1] strategy.

##### Algorithm: Recursive Binary Decision Tree for Linear Regression

1. Start with top node $\mathbb{R}^p$
2. While a stopping criterion is unsatisfied:
    1. Let

      $$(\hat{i}, \hat{j}) = \underset{(i, j)}{\text{argmin}}\left(
                                 \sum_{i: x_i\in R_1} (y_i - \hat{y}_{R_1})^ 2 + 
                                 \sum_{i: x_i\in R_2} (y_i - \hat{y}_{R_2})^ 2\right)$$
       where 

      $$R_{1} = \{X| X_j < x_{i,j}\}$$
      $$R_{2} = \{X| X_j \geqslant x_{i,j}\}$$

    2. Add nodes 

        $$\hat{R}_{1} = \{X| X_{\hat{j}} < x_{\hat{i},\hat{j}}\}$$
        $$\hat{R}_{2} = \{X| X_{\hat{j}} \geqslant x_{\hat{i},\hat{j}}\}$$

        to the partition, and recurse on one of the nodes

##### Tree-pruning

- Complex trees can overfit, but simpler trees may avoid it [^2]
- To get a simpler tree, we can grow a large tree $T_0$ and ***prune*** it to obtain a subtree.
- ***Cost complexity*** or ***weakest link*** pruning is a method for finding an optimal subtree [^3]. For $\alpha > 0$, we obtain a subtree
    $$ T_\alpha = \underset{T\ \subset T_0}{\text{argmin}}
                  \left(\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} \left(y_i - \hat{y}_{R_m}\right)^2 + \alpha|T|\right)$$
  where
  $|T|$ is the number of terminal nodes of $T$, $R_m$ is the rectangle corresponding to the $m$-th terminal node, and $\hat{y}_{R_m}$ is the predicted response (average of $y_i\in R_m$) [^4].

##### Algorithm: Weakest link regression tree with $K$-fold cross-validation

1. For each [^5] $\alpha > 0$:
   
   1. For $k = 1, \dots K$:
       1. Let $\mathcal{D}_k = \mathcal{D} \backslash \{k-\text{th fold}\}$
       2. Use recursive binary splitting to grow a tree $T_{k}$, stopping when each node has fewer than some minimum number of observations $M$ is reached [^6]
       3. Use weakest link pruning to find a subtree $T_{k, \alpha}$
   2. Let $CV_{(k)}(\alpha)$ be the $K$-fold cross-validation estimate of the mean squared test error
2. Choose
    $$ \hat{\alpha} = \underset{\alpha}{\text{argmin}}\ CV_{(k)}(\alpha) $$
3. Return $\hat{T} = T_{\hat{\alpha}}$

### Classification Trees

- ***Classification trees*** are very similar to regression trees, but they predict qualitative responses. The predicted class for an observation $(x_i, y_i)$ in $R_m$ is [^7] is
$$ \hat{k}_m = \underset{k}{\text{argmax}}\ \hat{p}_{m,k} $$
    where $\hat{p}_{m,k}$ is the fraction of observations $(x_i, y_i)$ in the region $R_m$ such that $y_i = k$.
- One performance measure is the ***Classification error rate*** [^8] for the region $R_m$ is
    $$ E_m = 1 - \hat{p}_{m, \hat{k}} $$
- A better performance measure is the  ***Gini index*** for the region $R_m$, a measure of total variance [^9] across classes
    $$ G_m = \sum_{k = 1}^K \hat{p}_{m,k}(1 - \hat{p}_{m,k})$$
- Another better performance measure is the ***entropy*** for the region $R_m$ [^10]
    $$ D_m = \sum_{k = 1}^K - \hat{p}_{m,k}\log(\hat{p}_{m,k}) $$
- Typically the Gini index or entropy is used to prune, due to their sensitivity to node purity. However, if prediction accuracy is the goal then classification error rate is preferable.


### Trees Versus Linear Models

- A linear regression model is of the form

    $$f(X) = \beta_0 + \sum_{j = 1}^p \beta_j X_j$$
    
    while a regression tree model is of the form
    
    $$ f(X) = \sum_{m = 1}^M c_m I(X \in R_m)$$
- Linear regression will tend to perform better if the relationship between features and response is well-approximated by a linear function, whereas the regression tree will tend perform better if the relationship is non-linear or complex. 

### Advantages and Disadvantages of Trees

##### Advantages

- Conceptual simplicity
- May mirror human decision-making better than previous regression and classification methods
- Readily visualizable and easily interpreted
- Can handle qualitative predictors without the need for dummy variables

##### Disadvantages

- Less accurate prediction than previous regression and classification methods
- Non-robust to changes in data -- small changes in data lead to large changes in estimated tree.

## Bagging, Random Forests, Boosting

These are methods for improving the prediction accuracy of decision trees.

### Bagging

- The decision trees in [§ 8.1](#The-Basics-of-Decision-Trees) suffer from high variance. 
- ***Bagging*** is a method of reducing the variance of a statistical learning process [^11].  The bagging estimate of the target function of the process with dataset $\mathcal{D}$ is

    $$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b = 1}^B \hat{f}^{*b}(x) $$

   where $\hat{f^*}^b(x)$ is the estimate of target function on the boostrap dataset        $\mathcal{D}_b$ [^12].
- Bagging can be used for any statistical learning method but it is particularly useful for decision trees [^13]

##### Out-of-bag Error Estimation

- On average, a bagged tree uses about 2/3 of the data -- the remaining 1/3 is the ***out-of-bag*** (OOB) data. 
- We can predict the response for each observation using the trees for which it was OOB, yielding about B/3 prediction.
- If we're doing regression, we can average these predicted responses, or if we're doing classification, take a majority vote, to get a single OOB prediction for each observation.
- Test error can be estimated using these predictions.

##### Variable Importance Measures

- Bagging typically results in improved prediction accuracy over single trees, at the expense of interpretability
- The RSS (for bagging regression trees) and Gini index (for bagging classification trees) can provide measures of variable importance.
- For both loss functions (RSS/Gini) the amount the loss is decreases due to a split over a given predictor, averaged over the B bagged trees. The greater the decrease, the more important the predictor

### Random Forests

- Random forests works as follows: at each split in the tree, choose a predictor from among a new random sample of $1 \leqslant m \leqslant p$ predictors.
- The random predictor sampling overcomes the tendency of bagged trees to look similar given strong predictors (e.g. the strongest predictor will be at the top of most or all of the bagged trees). 
- On average, $\frac{p-m}{p}$ of the splits will not consider a given predictor, giving other predictors a chance to be chosen. This decorrelation of the trees improves the reduction in variance achieved by bagged.
- $m=p$ corresponds to bagging. $m << p$ is useful when there is a large number of correlated predictors. Typically we choose $m \approx \sqrt{p}$

### Boosting

- Boosting is another method of improving prediction accuracy that can be applied to many statistical learning methods.
- In decision trees, each tree is build using information from the previous trees. Instead of bootstrapped datasets, the datasets are modified based on the previously grown trees.
- The boosting approach learns slowly, by slowly improving in areas where it underperforms. It has 3 parameters:
    - Number of trees $B$. Unlike bagging and random forests, boosting can overfit if $B$ is too big, although this happens slowly.
    - The shrinkage parameter $\lambda > 0$. Typical values are $\lambda = 0.01, 0.001$. Very small $\lambda$ can require very large $B$ to get good performance.
    - The number of tree splits $d$, which controls the complexity of the boosted ensemble. Often $d$ works well (the resulting tree is called a ***stump***) [^14]

##### Algorithm: Boosting for Regression Trees

1. Set $\hat{f}(x) = 0$ and $r_i = y_i$, $1 \leqslant i \leqslant n$
2. For $b  = 1, 2, \dots, B$:
    1. Fit a tree $\hat{f}^b$ with $d$ splits to $(X, r)$
    2. Update the model $\hat{f}$ by adding a shrunk version of the new tree:
    $$ \hat{f}(x) \leftarrow \hat{f}(x) + \lambda \hat{f}^b(x)$$
    3. Update the residuals:
    $$ r_i \leftarrow r_i - \lambda \hat{f}^b(x)$$
3. Output the boosted model

    $$ \hat{f}(x) = \sum_{b = 1}^B \lambda \hat{f}^b(x)$$

___
## Footnotes

[^1]: This strategy results in a binary tree with the partition regions as leaves, and binary splits as nodes. It is "top-down" because it starts at the top of the partition tree (with a single region), "binary" because it splits the predictor space into two regions at each node in the tree, "recursive" because it calls itself at each node, and "greedy" because at each node, it chooses the optimal split at that node

[^2]: That is, it may lead to lower variance and better interpretation at the cost of a higher bias

[^3]: We want a subtree with minimal estimated test error but it's infeasible to compute this for all subtrees.

[^4]: This is the RSS for the partition given by the nodes of the tree $T$, with a weighted penalty $\alpha|T|$ for the number of nodes (hence the complexity).

[^5]: Even though $\alpha \in [0, \infty)$ is a continuous parameter here, in practice it will be selected from a finite set of values. In fact (cf. comment on pg 309 of the text), as $\alpha$ increases,"branches get pruned in a nested and predictable fashion", resulting in a sequence of subtrees as a function of $\alpha$. One can then find a sequence $\alpha_1, \dots, \alpha_N$ such that at each $\alpha_i$, a branch is removed, and since the tree is finite, the algorithm is guaranteed to terminate.

[^6]: The smallest possible number of observations per node is $M=1$, which results in a partition with only one point in each region. This is clearly a maximal complexity tree, so we probably take $M >> 1$ in practice.

[^7]: That is, the predicted class for observations in $R_m$ is the most frequently occuring class in $R_m$.

[^8]: The classification error rate isn't sufficiently sensitive to "node purity", that is degree to which a node contains observations from a single class.

[^9]: The Gini index is a measure of "node purity" -- it is minimized when all $\hat{p}_{m, k} \in \{0, 1\}$, that is, when all nodes contain observations from a single class.

[^10]: The $\hat{p}_{m,k}$ are the empirical pmf estimates of the conditional probabilities $p_{m, k} = P(Y = k | X \in R_m)$, so $D$ is an estimate of the conditional entropy, i.e. the entropy of $Y\ |\ X \in R_m$. Thus $D$ is a measure of information that the empirical pmf, and hence the corresponding tree provides, that is, of its average suprisal. As with the Gini index, $D$ is minimized when all $\hat{p}_{m, k} \in \{0, 1\}$. An average surprisal of zero means the tree provides all information, that is, it perfectly separates the classes.

[^11]: Bagging is another name for <a href='#the-bootstrap'>bootstrapping</a>. It appears that the latter is usually used in the context of estimating the standard error of a statistic, while the former is used in the context of a statistical learning process (even though these are essentially the same).

[^12]: Really this is the bootstrap estimate of the average of the target function estimate over many datasets. For a given dataset $\mathcal{D}$,  the function $\hat{f}(x)$ produced by the learning process is an estimate of the target function $f(x)$. Repeating the process $1 \leqslant b \leqslant B$ times over datasets $\mathcal{D}_b$, we get estimates $\hat{f}^b(x)$. Assuming these are iid, they have common variance $\sigma^2$, but their average $$\hat{f}_{\text{avg}} = \frac{1}{B} \sum_{b = 1}^B \hat{f}^{b}(x)$$ has variance $\frac{\sigma^2}{B}$. Given $B$ large enough, this variance is low. Bagging/bootstrapping gets around the lack of separate datasets $\mathcal{D}_b$ in practice by repeated sampling with replacement from a single dataset $\mathcal{D}$.


[^13]: For regression, one grows $B$ deep (unpruned) regression trees on $B$ bootstrapped datasets, each of which has [low bias but high variance](#The-Bias-Variance-Tradeoff), then averages them to get a bootstrap estimate which has the same low bias, but much lower variance. For classification (since we can't average over the classes of the bootstrapped trees) a simple approach is to predict the majority class over the bootstrapped trees.

[^14]: In the case of a stump, the boosted ensemble is fitting an additive model, since each term is a single variable. More generally, $d$ is the interaction depth -- since $d$ splits can involve at most $d$ variables, this controls the interaction order of the boosted model (i.e. the model can fit interaction terms up to degree $d$.

{% endkatexmm %}